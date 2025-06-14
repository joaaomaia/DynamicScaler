# -*- coding: utf-8 -*-
import logging
import pathlib
import hashlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sns = None
from scipy import stats
from scipy.stats import shapiro, skew, kurtosis
import sklearn
from sklearn.utils.validation import check_is_fitted
from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    QuantileTransformer,
    PowerTransformer,
)

__version__ = "0.4.0"


class DynamicScaler(BaseEstimator, TransformerMixin):
    """
    Seleciona e aplica dinamicamente o scaler adequado para cada feature numérica.

    Parâmetros
    ----------
    strategy : {'auto', 'standard', 'robust', 'minmax', 'quantile', None}, default='auto'
        - 'auto'     → decide por coluna com base em normalidade, skew e outliers.
        - demais     → aplica o scaler escolhido a **todas** as colunas.
        - None       → passthrough (sem escalonamento).

    serialize : bool, default=False
        Se True, salva automaticamente o dict de scalers em `save_path` após o fit.

    save_path : str | Path | None
        Caminho do arquivo .pkl a ser salvo (ou sobreposto). Só usado se
        `serialize=True`. Padrão: 'scalers.pkl'.

    random_state : int, default=0
        Usado no QuantileTransformer e em amostragens internas.

    shapiro_p_val : float, default=0.01
        Valor-p mínimo para considerar a variável normal e habilitar o
        ``StandardScaler`` no modo ``auto``.

    shapiro_n : int, default=5000
        Tamanho máximo da amostra usada no teste de Shapiro-Wilk.

    ignore_cols : list[str] | None
        Colunas numéricas a serem preservadas sem escalonamento.

    logger : logging.Logger | None
        Logger customizado; se None, cria logger básico.
    """

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
    def __init__(
        self,
        strategy: str = "auto",
        shapiro_p_val: float = 0.01,
        serialize: bool = False,
        save_path: str | pathlib.Path | None = None,
        random_state: int = 0,
        missing_strategy: str = "none",
        plot_backend: str = "matplotlib",
        power_skew_thr: float = 1.0,
        power_kurt_thr: float = 10.0,
        power_method: str = "auto",
        profile: str = "default",
        shapiro_n: int = 5000,
        n_jobs: int = 1,
        ignore_cols: list[str] | None = None,
        logger: logging.Logger | None = None,
        min_post_std: float = 1e-3,
        min_post_iqr: float = 1e-3,
        min_post_unique: int = 2,
        validation_fraction: float = 0.1,
        scoring: Callable | None = None,
        ignore_scalers: list[str] | None = None,
        extra_scalers: list[BaseEstimator] | None = None,
        extra_validation: bool = False,
        allow_minmax: bool = True,
        kurtosis_thr: float = 10.0,
        cv_gain_thr: float = 0.002,
    ):
        self.strategy = strategy.lower() if strategy else None
        self.serialize = serialize
        self.save_path = pathlib.Path(save_path or "scalers.pkl")
        self.shapiro_p_val = shapiro_p_val
        self.random_state = random_state
        self.missing_strategy = missing_strategy
        self.plot_backend = plot_backend
        self.power_skew_thr = power_skew_thr
        self.power_kurt_thr = power_kurt_thr
        self.power_method = power_method
        self.profile = profile
        self.shapiro_n = shapiro_n
        self.n_jobs = n_jobs
        self.ignore_cols = set(ignore_cols or [])
        self.min_post_std = min_post_std
        self.min_post_iqr = min_post_iqr
        self.min_post_unique = min_post_unique
        self.validation_fraction = validation_fraction
        if scoring is None:
            self.scoring = lambda _y, arr: stats.skew(np.abs(arr), nan_policy="omit")
        else:
            self.scoring = scoring
        self.ignore_scalers = set(ignore_scalers or [])
        self.extra_scalers = extra_scalers or []
        self.extra_validation = extra_validation
        self.allow_minmax = allow_minmax
        self.kurtosis_thr = kurtosis_thr
        self.cv_gain_thr = cv_gain_thr

        self.scalers_: dict[str, BaseEstimator] | None = None
        self.report_: dict[str, dict] = {}  # estatísticas por coluna
        self.stats_: dict[str, dict] = {}

        # logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                logging.basicConfig(
                    level=logging.INFO, format="%(levelname)s: %(message)s"
                )

        self.tags_ = {"allow_nan": True, "X_types": ["2darray", "dataframe"]}

    # ------------------------------------------------------------------
    # MÉTODO INTERNO PARA ESTRATÉGIA AUTO
    # ------------------------------------------------------------------
    def _choose_auto(
        self, x: pd.Series, y: pd.Series | None = None, *, stats_callback: Callable | None = None
    ):
        """Avalia uma fila curta de scalers e retorna o primeiro que passa na validação."""

        sample = x.dropna().astype(float)
        if sample.nunique() == 1:
            return None, {
                "chosen_scaler": "None",
                "reason": "constante",
                "validation_stats": {},
                "ignored": list(self.ignore_scalers),
                "candidates_tried": [],
            }

        n_val = max(1, int(len(sample) * self.validation_fraction))
        val = sample.sample(n_val, random_state=self.random_state)
        train = sample.drop(index=val.index)
        baseline_score = self.scoring(None, val.values)
        baseline_kurt = float(kurtosis(val.values, nan_policy="omit"))
        baseline_cv: float | None = None

        # --------------------------------------------------------------
        # Teste de normalidade (Shapiro-Wilk)
        # --------------------------------------------------------------
        shapiro_sample = sample.sample(
            min(len(sample), self.shapiro_n), random_state=self.random_state
        )
        try:
            shapiro_p = float(shapiro(shapiro_sample)[1])
        except Exception:
            shapiro_p = 0.0
        normal = shapiro_p >= self.shapiro_p_val

        queue: list[BaseEstimator] = []
        if normal and "StandardScaler" not in self.ignore_scalers:
            queue.append(StandardScaler())
        pt_args = {}
        if self.power_method != "auto":
            pt_args["method"] = self.power_method
        default_q = [
            PowerTransformer(**pt_args),
            QuantileTransformer(
                output_distribution="normal", random_state=self.random_state
            ),
            RobustScaler(),
        ]
        if self.allow_minmax:
            default_q.append(MinMaxScaler())
        for sc in default_q:
            if sc.__class__.__name__ not in self.ignore_scalers:
                queue.append(sc)
        if self.extra_scalers:
            for sc in self.extra_scalers:
                if sc.__class__.__name__ not in self.ignore_scalers:
                    queue.append(sc)

        tried: list[str] = []
        for scaler in queue:
            name = scaler.__class__.__name__
            tried.append(name)
            scaler.fit(train.values.reshape(-1, 1))
            tr = scaler.transform(train.values.reshape(-1, 1)).ravel()
            cand_cv = float("nan")
            post_std = float(np.std(tr))
            post_iqr = float(np.percentile(tr, 75) - np.percentile(tr, 25))
            post_n_unique = int(len(np.unique(tr)))
            if (
                post_std < self.min_post_std
                or post_iqr < self.min_post_iqr
                or post_n_unique < self.min_post_unique
            ):
                continue
            val_tr = scaler.transform(val.values.reshape(-1, 1)).ravel()
            skew_test = float(self.scoring(None, val_tr))
            if abs(skew_test) >= abs(baseline_score):
                continue
            kurt_test = float(kurtosis(val_tr, nan_policy="omit"))
            if abs(kurt_test) >= abs(baseline_kurt) or abs(kurt_test) > self.kurtosis_thr:
                continue
            need_cv = self.extra_validation or name == "MinMaxScaler"
            if need_cv:
                if y is None:
                    continue
                y_train = y.loc[train.index]
                if baseline_cv is None:
                    baseline_cv = self._cv_score(train.values.reshape(-1, 1), y_train)
                cand_cv = self._cv_score(tr.reshape(-1, 1), y_train)
                if cand_cv < baseline_cv + self.cv_gain_thr:
                    continue
            report = {
                "chosen_scaler": name,
                "validation_stats": {
                    "post_std": post_std,
                    "post_iqr": post_iqr,
                    "post_n_unique": post_n_unique,
                    "skew_test": skew_test,
                    "kurtosis_test": kurt_test,
                    "cv_score": float(cand_cv) if need_cv else float("nan"),
                },
                "ignored": list(self.ignore_scalers),
                "candidates_tried": tried,
            }
            if stats_callback:
                stats_callback(x.name, report)
            return scaler, report

        return None, {
            "chosen_scaler": "None",
            "validation_stats": {
                "post_std": float("nan"),
                "post_iqr": float("nan"),
                "post_n_unique": 0,
                "skew_test": float(baseline_score),
                "kurtosis_test": float(baseline_kurt),
                "cv_score": float(baseline_cv) if baseline_cv is not None else float("nan"),
            },
            "ignored": list(self.ignore_scalers),
            "candidates_tried": tried,
            "reason": "all_rejected",
        }

    # ------------------------------------------------------------------
    # API FIT
    # ------------------------------------------------------------------
    def fit(self, X, y=None, *, stats_callback: Callable | None = None):
        X_df = pd.DataFrame(X)
        num_df = X_df.select_dtypes("number")
        y_series = None if y is None else pd.Series(y, index=X_df.index)

        ignore_in_data: list[str] = []
        if self.ignore_cols:
            ignore_in_data = list(set(self.ignore_cols) & set(num_df.columns))
            if ignore_in_data:
                self.logger.info("Ignoring columns (no scaling): %s", ignore_in_data)
                num_df = num_df.drop(columns=ignore_in_data)
        self.ignored_cols_ = ignore_in_data

        non_numeric = set(X_df.columns) - set(num_df.columns)
        if non_numeric:
            self.logger.warning("Ignoring non-numeric columns: %s", list(non_numeric))

        if self.missing_strategy == "drop":
            num_df = num_df.dropna()

        self.dtypes_ = num_df.dtypes.to_dict()

        if self.strategy not in {
            "auto",
            "standard",
            "robust",
            "minmax",
            "quantile",
            None,
        }:
            raise ValueError(f"strategy '{self.strategy}' não suportada.")

        self.scalers_ = {}

        for col in num_df.columns:
            if self.strategy == "auto":
                scaler, report = self._choose_auto(
                    num_df[col], y_series, stats_callback=stats_callback
                )
                if scaler is not None:
                    scaler.fit(num_df[[col]])
            elif self.strategy == "standard":
                scaler = StandardScaler().fit(num_df[[col]])
                report = {
                    "chosen_scaler": "StandardScaler",
                    "reason": "global-standard",
                }
            elif self.strategy == "robust":
                scaler = RobustScaler().fit(num_df[[col]])
                report = {"chosen_scaler": "RobustScaler", "reason": "global-robust"}
            elif self.strategy == "minmax":
                scaler = MinMaxScaler().fit(num_df[[col]])
                report = {"chosen_scaler": "MinMaxScaler", "reason": "global-minmax"}
            elif self.strategy == "quantile":
                scaler = QuantileTransformer(
                    output_distribution="normal", random_state=self.random_state
                ).fit(num_df[[col]])
                report = {
                    "chosen_scaler": "QuantileTransformer",
                    "reason": "global-quantile",
                }
            else:
                scaler = None
                report = {"chosen_scaler": "None", "reason": "passthrough"}

            self.scalers_[col] = scaler
            self.report_[col] = report

        self.feature_names_in_ = np.array(list(self.scalers_.keys()))
        self.n_features_in_ = len(self.feature_names_in_)
        self.columns_hash_ = hashlib.md5(",".join(self.scalers_).encode()).hexdigest()
        self.selected_cols_ = [c for c, r in self.report_.items() if r.get("chosen_scaler") != "None"]

        if self.serialize:
            self.save(self.save_path)

        return self

    # ------------------------------------------------------------------
    # PARTIAL FIT
    # ------------------------------------------------------------------
    def partial_fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        if not getattr(self, "scalers_", None):
            return self.fit(X_df)

        for col, scaler in self.scalers_.items():
            if col not in X_df.columns:
                self.logger.warning(
                    "Column '%s' missing in partial_fit input; skipping", col
                )
                continue
            if scaler is not None and hasattr(scaler, "partial_fit"):
                scaler.partial_fit(X_df[[col]])
            else:
                self.logger.info("Column '%s' scaler does not support partial_fit", col)
                if self.profile == "streaming" and isinstance(scaler, PowerTransformer):
                    self.logger.info("PowerTransformer lacks partial_fit support")
        return self

    # ------------------------------------------------------------------
    # TRANSFORM / INVERSE_TRANSFORM
    # ------------------------------------------------------------------
    def transform(
        self,
        X,
        *,
        return_df: bool = False,
        strict: bool = False,
        keep_other_cols: bool = True,
        log_level: str = "full",
    ):
        check_is_fitted(self, "feature_names_in_")
        X_df = pd.DataFrame(X).copy()

        if self.missing_strategy == "drop":
            X_df = X_df.dropna()

        missing = set(self.scalers_) - set(X_df.columns)
        if missing and strict:
            raise ValueError(f"Missing columns during transform: {missing}")

        for col, scaler in self.scalers_.items():
            if col in X_df.columns and scaler is not None:
                data_col = X_df[[col]].copy()
                fill = (
                    self.report_[col].get("fill_value")
                    if self.missing_strategy != "none"
                    else None
                )
                if fill is not None:
                    data_col[col] = data_col[col].fillna(fill)
                X_df[col] = scaler.transform(data_col)

        if log_level == "full":
            untouched = set(X_df.columns) - set(self.scalers_)
            if untouched:
                self.logger.info("Untouched columns preserved: %s", list(untouched))

        if keep_other_cols:
            return X_df if return_df else X_df.values
        else:
            X_scaled_only = X_df[list(self.scalers_)]
            return X_scaled_only if return_df else X_scaled_only.values

    def inverse_transform(
        self,
        X,
        *,
        return_df: bool = False,
        strict: bool = False,
        keep_other_cols: bool = True,
        log_level: str = "full",
    ):
        check_is_fitted(self, "feature_names_in_")
        X_df = pd.DataFrame(X).copy()
        missing = set(self.scalers_) - set(X_df.columns)
        if missing and strict:
            raise ValueError(f"Missing columns during transform: {missing}")

        for col, scaler in self.scalers_.items():
            if col in X_df.columns and scaler is not None:
                X_df[col] = scaler.inverse_transform(X_df[[col]])

        for col, dt in getattr(self, "dtypes_", {}).items():
            if col in X_df.columns:
                X_df[col] = X_df[col].astype(dt)

        if log_level == "full":
            untouched = set(X_df.columns) - set(self.scalers_)
            if untouched:
                self.logger.info("Untouched columns preserved: %s", list(untouched))

        if keep_other_cols:
            return X_df if return_df else X_df.values
        else:
            X_scaled_only = X_df[list(self.scalers_)]
            return X_scaled_only if return_df else X_scaled_only.values

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return getattr(self, "feature_names_in_", None)
        return np.array(input_features)

    def report_as_df(self) -> pd.DataFrame:
        """Devolve o relatório de métricas/decisões como DataFrame."""
        return pd.DataFrame.from_dict(self.report_, orient="index")

    def _cv_score(self, X: np.ndarray, y: pd.Series) -> float:
        """Calcula score de CV usando XGBoost."""
        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

        if y.dtype.kind in {"i", "u", "b"} and np.unique(y).size <= 20:
            model = xgb.XGBClassifier(
                random_state=self.random_state,
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
            )
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            return float(scores.mean())
        else:
            model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
            )
            cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
            return float(scores.mean())

    def plot_histograms(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        features: str | list[str],
        *,
        show_qq: bool = False,
    ):
        """
        Plota histogramas lado a lado (antes/depois do escalonamento) para uma ou mais variáveis.

        O título do histograma indica o scaler aplicado usando ``chosen_scaler``
        registrado no relatório.

        Parâmetros
        ----------
        original_df : pd.DataFrame
            DataFrame original (antes do transform).

        transformed_df : pd.DataFrame
            DataFrame escalonado (após o transform), com mesmas colunas que original_df.

        features : str ou list
            Nome de uma coluna ou lista de colunas a serem inspecionadas.
        """
        # Normalizar input
        if isinstance(features, str):
            features = [features]

        if self.plot_backend == "none":
            return

        for feature in features:
            if feature not in self.scalers_:
                self.logger.warning(
                    "Variável '%s' não foi tratada no fit. Pulando...", feature
                )
                continue

            scaler_nome = self.report_.get(feature, {}).get("chosen_scaler", "Nenhum")
            if scaler_nome is None or scaler_nome == "None":
                scaler_nome = "Nenhum"
            self.logger.info(
                "Plotando histograma de %s — scaler = %s", feature, scaler_nome
            )

            cols = 3 if show_qq and self.plot_backend == "matplotlib" else 2
            plt.figure(figsize=(6 * cols, 4))

            # Original
            plt.subplot(1, cols, 1)
            if self.plot_backend == "seaborn":
                sns.histplot(
                    original_df[feature].dropna(), bins=30, kde=True, color="steelblue"
                )
            else:
                plt.hist(
                    original_df[feature].dropna(), bins=30, color="steelblue", alpha=0.7
                )
            plt.title(f"{feature} — original")
            plt.xlabel(feature)

            # Transformada
            plt.subplot(1, cols, 2)
            if self.plot_backend == "seaborn":
                sns.histplot(
                    transformed_df[feature].dropna(),
                    bins=30,
                    kde=True,
                    color="darkorange",
                )
            else:
                plt.hist(
                    transformed_df[feature].dropna(),
                    bins=30,
                    color="darkorange",
                    alpha=0.7,
                )
            plt.title(f"{feature} — escalado com {scaler_nome}")
            plt.xlabel(feature)

            if show_qq and self.plot_backend == "matplotlib":
                plt.subplot(1, cols, 3)
                stats.probplot(transformed_df[feature].dropna(), dist="norm", plot=plt)
                plt.title(f"{feature} — QQ")

            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # SERIALIZAÇÃO
    # ------------------------------------------------------------------
    def save(self, path: str | pathlib.Path | None = None):
        """Serializa scalers + relatório + metadados."""
        path = pathlib.Path(path or self.save_path)
        scalers_to_save = {c: self.scalers_[c] for c in getattr(self, "selected_cols_", self.scalers_)}
        joblib.dump(
            {
                "scalers": scalers_to_save,
                "report": self.report_,
                "strategy": self.strategy,
                "random_state": self.random_state,
                "library_version": sklearn.__version__,
                "columns_hash": self.columns_hash_,
            },
            path,
            compress=("gzip", 3),
        )
        self.logger.info("Scalers salvos em %s", path)

    def load(self, path: str | pathlib.Path):
        """Restaura scalers + relatório + metadados já treinados."""
        data = joblib.load(path)
        self.scalers_ = data["scalers"]
        self.report_ = data.get("report", {})
        self.strategy = data.get("strategy", self.strategy)
        self.random_state = data.get("random_state", self.random_state)
        self.feature_names_in_ = np.array(list(self.scalers_.keys()))
        self.n_features_in_ = len(self.feature_names_in_)
        expected_hash = hashlib.md5(",".join(self.scalers_).encode()).hexdigest()
        if data.get("columns_hash") != expected_hash:
            raise ValueError("Columns hash mismatch")
        if data.get("library_version") != sklearn.__version__:
            self.logger.warning(
                "Library version mismatch: %s vs %s",
                data.get("library_version"),
                sklearn.__version__,
            )
        self.columns_hash_ = expected_hash
        self.selected_cols_ = [c for c, r in self.report_.items() if r.get("chosen_scaler") != "None"]
        self.logger.info("Scalers carregados de %s", path)
        return self
