# -*- coding: utf-8 -*-
import logging
import pathlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, skew, kurtosis

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
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

    logger : logging.Logger | None
        Logger customizado; se None, cria logger básico.
    """

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
    def __init__(self,
                 strategy: str = 'auto',
                 shapiro_p_val: float = 0.01, # se aumentar fica mais restritiva a escolha de StandardScaler()
                 serialize: bool = False,
                 save_path: str | pathlib.Path | None = None,
                 random_state: int = 0,
                 logger: logging.Logger | None = None):
        self.strategy = strategy.lower() if strategy else None
        self.serialize = serialize
        self.save_path = pathlib.Path(save_path or "scalers.pkl")
        self.shapiro_p_val = shapiro_p_val
        self.random_state = random_state

        self.scalers_: dict[str, BaseEstimator] = {}
        self.report_:  dict[str, dict] = {}      # estatísticas por coluna

        # logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.handlers:
                logging.basicConfig(level=logging.INFO,
                                    format="%(levelname)s: %(message)s")

    # ------------------------------------------------------------------
    # MÉTODO INTERNO PARA ESTRATÉGIA AUTO
    # ------------------------------------------------------------------
    def _choose_auto(self, x: pd.Series):
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
            p_val = shapiro(sample.sample(min(5000, len(sample)),
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
        if p_val >= 0.05 and abs(sk) <= 0.5:
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
        return scaler, stats

    # ------------------------------------------------------------------
    # API FIT
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)

        if self.strategy not in {'auto', 'standard', 'robust',
                                 'minmax', 'quantile', None}:
            raise ValueError(f"strategy '{self.strategy}' não suportada.")

        for col in X_df.columns:
            # --- seleção do scaler -----------------------------------
            if self.strategy == 'auto':
                scaler, stats = self._choose_auto(X_df[col])
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

            # --- ajuste ---------------------------------------------
            if scaler is not None:
                scaler.fit(X_df[[col]])

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

        # serialização opcional
        if self.serialize:
            self.save(self.save_path)

        return self

    # ------------------------------------------------------------------
    # TRANSFORM / INVERSE_TRANSFORM
    # ------------------------------------------------------------------
    def transform(self, X, return_df: bool = False):
        X_df = pd.DataFrame(X).copy()

        # Verifica se todas as colunas esperadas estão presentes
        missing = set(self.scalers_) - set(X_df.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no transform: {missing}")

        for col, scaler in self.scalers_.items():
            if scaler is not None:
                X_df[col] = scaler.transform(X_df[[col]])

        return X_df if return_df else X_df.values

    def inverse_transform(self, X, return_df: bool = False):
        X_df = pd.DataFrame(X, columns=self.scalers_.keys()).copy()
        for col, scaler in self.scalers_.items():
            if scaler is not None:
                X_df[col] = scaler.inverse_transform(X_df[[col]])
        return X_df if return_df else X_df.values

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)

    def report_as_df(self) -> pd.DataFrame:
        """Devolve o relatório de métricas/decisões como DataFrame."""
        return pd.DataFrame.from_dict(self.report_, orient='index')

    def plot_histograms(self, original_df: pd.DataFrame, transformed_df: pd.DataFrame, features: str | list[str]):
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

        for feature in features:
            if feature not in self.scalers_:
                self.logger.warning("Variável '%s' não foi tratada no fit. Pulando...", feature)
                continue

            scaler_nome = self.report_.get(feature, {}).get("scaler", "Desconhecido")

            plt.figure(figsize=(12, 4))

            # Original
            plt.subplot(1, 2, 1)
            sns.histplot(original_df[feature].dropna(), bins=30, kde=True, color="steelblue")
            plt.title(f"{feature} — original")
            plt.xlabel(feature)

            # Transformada
            plt.subplot(1, 2, 2)
            sns.histplot(transformed_df[feature].dropna(), bins=30, kde=True, color="darkorange")
            plt.title(f"{feature} — escalado com {scaler_nome}")
            plt.xlabel(feature)

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
            'random_state': self.random_state
        }, path)
        self.logger.info("Scalers salvos em %s", path)

    def load(self, path: str | pathlib.Path):
        """Restaura scalers + relatório + metadados já treinados."""
        data = joblib.load(path)
        self.scalers_  = data['scalers']
        self.report_   = data.get('report', {})
        self.strategy  = data.get('strategy', self.strategy)
        self.random_state = data.get('random_state', self.random_state)
        self.logger.info("Scalers carregados de %s", path)
        return self
