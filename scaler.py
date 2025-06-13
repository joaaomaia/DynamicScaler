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
    StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer,
    PowerTransformer,
)

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

    ignore_cols : list[str] | None
        Colunas numéricas a serem preservadas sem escalonamento.

    logger : logging.Logger | None
        Logger customizado; se None, cria logger básico.
    """

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
    def __init__(self,
                 strategy: str = 'auto',
                 shapiro_p_val: float = 0.01,
                 serialize: bool = False,
                 save_path: str | pathlib.Path | None = None,
                 random_state: int = 0,
                 missing_strategy: str = 'none',
                 plot_backend: str = 'matplotlib',
                 power_skew_thr: float = 1.0,
                 power_kurt_thr: float = 10.0,
                 power_method: str = 'auto',
                 profile: str = 'default',
                 shapiro_n: int = 5000,
                 n_jobs: int = 1,
                 ignore_cols: list[str] | None = None,
                 logger: logging.Logger | None = None):
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

        self.scalers_: dict[str, BaseEstimator] | None = None
        self.report_:  dict[str, dict] = {}      # estatísticas por coluna
        self.stats_:   dict[str, dict] = {}

        # logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                logging.basicConfig(level=logging.INFO,
                                    format="%(levelname)s: %(message)s")

        self.tags_ = {"allow_nan": True, "X_types": ["2darray", "dataframe"]}

    # ------------------------------------------------------------------
    # MÉTODO INTERNO PARA ESTRATÉGIA AUTO
    # ------------------------------------------------------------------
    def _choose_auto(self, x: pd.Series, *, stats_callback: Callable | None = None):
        """
        Decide qual scaler empregar (ou nenhum) para a série x.

        Retorna
        -------
        scaler | None, dict
            Instância já criada (ainda não fitada) e dicionário de métricas.
        """
        sample = x.dropna().astype(float)

        # Coluna constante
        if sample.nunique() == 1:
            return None, dict(reason='constante', scaler='None')

        # ---------------- métricas básicas ----------------
        try:
            p_val = shapiro(sample.sample(min(self.shapiro_n, len(sample)),
                                          random_state=self.random_state))[1]
        except Exception:   # amostra minúscula ou erro numérico
            p_val = 0.0

        sk = skew(sample, nan_policy="omit")
        kt = kurtosis(sample, nan_policy="omit")        # Fisher (0 = normal)

        # ---------------- critérios de NÃO escalonar ----------------
        # (1) variável já em [0,1]
        if 0.95 <= sample.min() <= sample.max() <= 1.05:
            return None, dict(p_value=p_val, skew=sk, kurtosis=kt,
                              reason='já escalada [0-1]', scaler='None')

        # # (2) praticamente normal
        # if abs(sk) < 0.05 and abs(kt) < 0.1 and p_val > 0.90:
        #     return None, dict(p_value=p_val, skew=sk, kurtosis=kt,
        #                       reason='praticamente normal', scaler='None')
        
        # (3) praticamente normal (menos restritivo)
        if abs(sk) < 0.5 and abs(kt) < 1.0 and p_val > self.shapiro_p_val:
            return None, dict(p_value=p_val, skew=sk, kurtosis=kt,
                            reason='aproximadamente normal', scaler='None')


        # ---------------- escolha de scaler ----------------
        if (abs(sk) > self.power_skew_thr and kt <= self.power_kurt_thr and
                p_val < self.shapiro_p_val):
            method = self.power_method
            if method == 'auto':
                method = 'box-cox' if sample.min() > 0 else 'yeo-johnson'
            scaler = PowerTransformer(method=method, standardize=True)
            reason = f"{method} (high skew)"
        elif p_val >= 0.05 and abs(sk) <= 0.5:
            scaler = StandardScaler()
            reason = '≈normal'
        elif abs(sk) > 3 or kt > 20:
            scaler = QuantileTransformer(output_distribution='normal',
                                         random_state=self.random_state)
            reason = 'assimetria/kurtosis extrema'
        elif abs(sk) > 0.5:
            scaler = RobustScaler()
            reason = 'skew moderado/outliers'
        else:
            scaler = MinMaxScaler()
            reason = 'default'

        stats = dict(p_value=p_val, skew=sk, kurtosis=kt,
                     reason=reason, scaler=scaler.__class__.__name__)
        if stats_callback is not None:
            stats_callback(x.name, stats)
        return scaler, stats

    # ------------------------------------------------------------------
    # API FIT
    # ------------------------------------------------------------------
    def fit(self, X, y=None, *, stats_callback: Callable | None = None):
        X_df = pd.DataFrame(X)
        num_df = X_df.select_dtypes("number")
        ignore_in_data = []
        if self.ignore_cols:
            ignore_in_data = list(set(self.ignore_cols) & set(num_df.columns))
            if ignore_in_data:
                self.logger.info("Ignoring columns (no scaling): %s", ignore_in_data)
                num_df = num_df.drop(columns=ignore_in_data)
        self.ignored_cols_ = ignore_in_data
        non_numeric = set(X_df.columns) - set(num_df.columns)
        if non_numeric:
            self.logger.warning("Ignoring non-numeric columns: %s", list(non_numeric))
        X_df = num_df
        if self.missing_strategy == 'drop':
            X_df = X_df.dropna()
        self.dtypes_ = X_df.dtypes.to_dict()

        if self.strategy not in {'auto', 'standard', 'robust',
                                 'minmax', 'quantile', None}:
            raise ValueError(f"strategy '{self.strategy}' não suportada.")

        self.scalers_ = {}

        def _process(col):
            if self.strategy == 'auto':
                scaler, stats = self._choose_auto(X_df[col], stats_callback=stats_callback)
            elif self.strategy == 'standard':
                scaler = StandardScaler(); stats = dict(reason='global-standard', scaler='StandardScaler')
            elif self.strategy == 'robust':
                scaler = RobustScaler(); stats = dict(reason='global-robust', scaler='RobustScaler')
            elif self.strategy == 'minmax':
                scaler = MinMaxScaler(); stats = dict(reason='global-minmax', scaler='MinMaxScaler')
            elif self.strategy == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
                stats = dict(reason='global-quantile', scaler='QuantileTransformer')
            else:
                scaler = None; stats = dict(reason='passthrough', scaler='None')

            fill_value = None
            if self.missing_strategy == 'median':
                fill_value = X_df[col].median()
            elif self.missing_strategy == 'mean':
                fill_value = X_df[col].mean()
            elif self.missing_strategy == 'constant':
                fill_value = 0
            if fill_value is not None:
                X_df[col] = X_df[col].fillna(fill_value)
                stats['fill_value'] = fill_value

            if scaler is not None:
                scaler.fit(X_df[[col]])
                if isinstance(scaler, PowerTransformer):
                    transformed = scaler.transform(X_df[[col]])
                    post_sk = skew(transformed.ravel(), nan_policy="omit")
                    post_kt = kurtosis(transformed.ravel(), nan_policy="omit")
                    stats['post_skew'] = post_sk
                    stats['post_kurtosis'] = post_kt
                    self.stats_[col] = {
                        'pre_skew': stats.get('skew'),
                        'pre_kurt': stats.get('kurtosis'),
                        'post_skew': post_sk,
                        'post_kurt': post_kt,
                    }
            return col, scaler, stats

        if self.n_jobs == 1:
            results = [_process(col) for col in X_df.columns]
        else:
            results = Parallel(n_jobs=self.n_jobs)(delayed(_process)(col) for col in X_df.columns)

        for col, scaler, stats in results:
            self.scalers_[col] = scaler
            self.report_[col] = stats

            self.logger.info(
                "Coluna '%s' → %s (p=%.3f, skew=%.2f, kurt=%.1f) | motivo: %s",
                col, stats.get('scaler'),
                stats.get('p_value', np.nan),
                stats.get('skew',     np.nan),
                stats.get('kurtosis', np.nan),
                stats['reason'],
            )
            # --- seleção do scaler -----------------------------------
            if self.strategy == 'auto':
                scaler, stats = self._choose_auto(X_df[col], stats_callback=stats_callback)
            elif self.strategy == 'standard':
                scaler = StandardScaler()
                stats  = dict(reason='global-standard', scaler='StandardScaler')
            elif self.strategy == 'robust':
                scaler = RobustScaler()
                stats  = dict(reason='global-robust', scaler='RobustScaler')
            elif self.strategy == 'minmax':
                scaler = MinMaxScaler()
                stats  = dict(reason='global-minmax', scaler='MinMaxScaler')
            elif self.strategy == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal',
                                             random_state=self.random_state)
                stats  = dict(reason='global-quantile', scaler='QuantileTransformer')
            else:              # None
                scaler = None
                stats  = dict(reason='passthrough', scaler='None')

            # --- tratamento de missing values -----------------------
            fill_value = None
            if self.missing_strategy == 'median':
                fill_value = X_df[col].median()
            elif self.missing_strategy == 'mean':
                fill_value = X_df[col].mean()
            elif self.missing_strategy == 'constant':
                fill_value = 0
            if fill_value is not None:
                X_df[col] = X_df[col].fillna(fill_value)
                stats['fill_value'] = fill_value

            # --- ajuste ---------------------------------------------
            if scaler is not None:
                scaler.fit(X_df[[col]])
                if isinstance(scaler, PowerTransformer):
                    transformed = scaler.transform(X_df[[col]])
                    post_sk = skew(transformed.ravel(), nan_policy="omit")
                    post_kt = kurtosis(transformed.ravel(), nan_policy="omit")
                    stats['post_skew'] = post_sk
                    stats['post_kurtosis'] = post_kt
                    self.stats_[col] = {
                        'pre_skew': stats.get('skew'),
                        'pre_kurt': stats.get('kurtosis'),
                        'post_skew': post_sk,
                        'post_kurt': post_kt,
                    }

            self.scalers_[col] = scaler
            self.report_[col]  = stats

            # --- log -------------------------------------------------
            self.logger.info(
                "Coluna '%s' → %s (p=%.3f, skew=%.2f, kurt=%.1f) | motivo: %s",
                col, stats.get('scaler'),
                stats.get('p_value', np.nan),
                stats.get('skew',     np.nan),
                stats.get('kurtosis', np.nan),
                stats['reason']
            )

        self.feature_names_in_ = np.array(list(self.scalers_.keys()))
        self.n_features_in_ = len(self.feature_names_in_)
        self.columns_hash_ = hashlib.md5(",".join(self.scalers_).encode()).hexdigest()

        # serialização opcional
        if self.serialize:
            self.save(self.save_path)

        return self

    # ------------------------------------------------------------------
    # PARTIAL FIT
    # ------------------------------------------------------------------
    def partial_fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        if not getattr(self, 'scalers_', None):
            return self.fit(X_df)

        for col, scaler in self.scalers_.items():
            if col not in X_df.columns:
                self.logger.warning("Column '%s' missing in partial_fit input; skipping", col)
                continue
            if scaler is not None and hasattr(scaler, 'partial_fit'):
                scaler.partial_fit(X_df[[col]])
            else:
                self.logger.info("Column '%s' scaler does not support partial_fit", col)
                if self.profile == 'streaming' and isinstance(scaler, PowerTransformer):
                    self.logger.info("PowerTransformer lacks partial_fit support")
        return self

    # ------------------------------------------------------------------
    # TRANSFORM / INVERSE_TRANSFORM
    # ------------------------------------------------------------------
    def transform(self, X, *, return_df: bool = False, strict: bool = False,
                  keep_other_cols: bool = True, log_level: str = "full"):
        check_is_fitted(self, "feature_names_in_")
        X_df = pd.DataFrame(X).copy()

        if self.missing_strategy == 'drop':
            X_df = X_df.dropna()

        missing = set(self.scalers_) - set(X_df.columns)
        if missing and strict:
            raise ValueError(f"Missing columns during transform: {missing}")

        for col, scaler in self.scalers_.items():
            if col in X_df.columns and scaler is not None:
                data_col = X_df[[col]].copy()
                fill = self.report_[col].get('fill_value') if self.missing_strategy != 'none' else None
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

    def inverse_transform(self, X, *, return_df: bool = False, strict: bool = False,
                           keep_other_cols: bool = True, log_level: str = "full"):
        check_is_fitted(self, "feature_names_in_")
        X_df = pd.DataFrame(X).copy()
        missing = set(self.scalers_) - set(X_df.columns)
        if missing and strict:
            raise ValueError(f"Missing columns during transform: {missing}")

        for col, scaler in self.scalers_.items():
            if col in X_df.columns and scaler is not None:
                X_df[col] = scaler.inverse_transform(X_df[[col]])

        for col, dt in getattr(self, 'dtypes_', {}).items():
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
        return pd.DataFrame.from_dict(self.report_, orient='index')

    def plot_histograms(self, original_df: pd.DataFrame, transformed_df: pd.DataFrame, features: str | list[str], *, show_qq: bool = False):
        """
        Plota histogramas lado a lado (antes/depois do escalonamento) para uma ou mais variáveis.

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
                self.logger.warning("Variável '%s' não foi tratada no fit. Pulando...", feature)
                continue

            scaler_nome = self.report_.get(feature, {}).get("scaler", "Desconhecido")

            cols = 3 if show_qq and self.plot_backend == "matplotlib" else 2
            plt.figure(figsize=(6 * cols, 4))

            # Original
            plt.subplot(1, cols, 1)
            if self.plot_backend == "seaborn":
                sns.histplot(original_df[feature].dropna(), bins=30, kde=True, color="steelblue")
            else:
                plt.hist(original_df[feature].dropna(), bins=30, color="steelblue", alpha=0.7)
            plt.title(f"{feature} — original")
            plt.xlabel(feature)

            # Transformada
            plt.subplot(1, cols, 2)
            if self.plot_backend == "seaborn":
                sns.histplot(transformed_df[feature].dropna(), bins=30, kde=True, color="darkorange")
            else:
                plt.hist(transformed_df[feature].dropna(), bins=30, color="darkorange", alpha=0.7)
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
        joblib.dump({
            'scalers': self.scalers_,
            'report':  self.report_,
            'strategy': self.strategy,
            'random_state': self.random_state,
            'library_version': sklearn.__version__,
            'columns_hash': self.columns_hash_
        }, path, compress=('gzip', 3))
        self.logger.info("Scalers salvos em %s", path)

    def load(self, path: str | pathlib.Path):
        """Restaura scalers + relatório + metadados já treinados."""
        data = joblib.load(path)
        self.scalers_  = data['scalers']
        self.report_   = data.get('report', {})
        self.strategy  = data.get('strategy', self.strategy)
        self.random_state = data.get('random_state', self.random_state)
        self.feature_names_in_ = np.array(list(self.scalers_.keys()))
        self.n_features_in_ = len(self.feature_names_in_)
        expected_hash = hashlib.md5(",".join(self.scalers_).encode()).hexdigest()
        if data.get('columns_hash') != expected_hash:
            raise ValueError('Columns hash mismatch')
        if data.get('library_version') != sklearn.__version__:
            self.logger.warning(
                "Library version mismatch: %s vs %s",
                data.get('library_version'), sklearn.__version__)
        self.columns_hash_ = expected_hash
        self.logger.info("Scalers carregados de %s", path)
        return self
