import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================
# MODEL FITTING (ADAPTIVE)
# ============================

def fit_structural_model(df, entity_column, entity_value):
    """
    Ajusta modelo log-log para entidade específica.
    Especificação adaptativa baseada no tamanho da amostra.
    """

    # Filtra produto
    df_entity = df[df[entity_column] == entity_value].copy()

    # Remove valores inválidos antes do log
    df_entity = df_entity[
        (df_entity["price"] > 0) &
        (df_entity["quantity"] > 0)
    ]

    n_obs = len(df_entity)

    if n_obs < 6:
        raise ValueError("Not enough observations to estimate elasticity.")

    # Log-transform
    df_entity["log_price"] = np.log(df_entity["price"])
    df_entity["log_quantity"] = np.log(df_entity["quantity"])

    # ============================
    # MODEL SPECIFICATION SELECTION
    # ============================

    base_features = ["log_price", "cac", "promotion_flag"]

    if n_obs < 18:
        # 🔹 Base pequena → modelo simples
        X = df_entity[base_features]

    elif 18 <= n_obs < 36:
        # 🔹 Base média → tendência temporal contínua
        df_entity["month_index"] = (
            pd.to_datetime(df_entity["month"])
              .rank(method="dense")
        )
        X = df_entity[base_features + ["month_index"]]

    else:
        # 🔹 Base grande → fixed effects
        month_dummies = pd.get_dummies(
            df_entity["month"],
            prefix="month",
            drop_first=True
        )
        X = pd.concat(
            [df_entity[base_features], month_dummies],
            axis=1
        )

    # ============================
    # NUMERIC SAFETY
    # ============================

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df_entity["log_quantity"], errors="coerce")

    data = pd.concat([X, y], axis=1).dropna()

    X = data[X.columns]
    y = data["log_quantity"]

    # Segurança adicional contra singularidade
    if len(X.columns) >= len(X):
        # fallback automático para modelo simples
        X = df_entity[base_features]
        X = X.apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(df_entity["log_quantity"], errors="coerce")
        data = pd.concat([X, y], axis=1).dropna()
        X = data[X.columns]
        y = data["log_quantity"]

    # Adiciona constante
    X = sm.add_constant(X)

    X = X.astype(float)
    y = y.astype(float)

    model = sm.OLS(y, X).fit()

    return model


# ============================
# ELASTICITY DIAGNOSTICS (ROBUST)
# ============================

def elasticity_stat_diagnostics(model):
    """
    Retorna diagnóstico estatístico da elasticidade.
    Protegido contra NaN e singularidade.
    """

    if "log_price" not in model.params:
        return {
            "elasticity": None,
            "p_value": None,
            "reliability": "invalid",
            "economic_regime": "invalid"
        }

    beta = model.params.get("log_price", np.nan)
    p_value = model.pvalues.get("log_price", np.nan)

    if beta is None or np.isnan(beta):
        return {
            "elasticity": None,
            "p_value": None,
            "reliability": "invalid",
            "economic_regime": "invalid"
        }

    # Significance
    if p_value is not None and not np.isnan(p_value):
        reliability = "significant" if p_value < 0.05 else "not_significant"
    else:
        reliability = "invalid"

    # Economic regime
    if beta < -1:
        regime = "elastic"
    elif -1 <= beta < 0:
        regime = "inelastic"
    else:
        regime = "invalid"

    return {
        "elasticity": beta,
        "p_value": p_value,
        "reliability": reliability,
        "economic_regime": regime
    }


# ============================
# TRUST REGION CHECK
# ============================

def trust_region_check(price_candidate, observed_min, observed_max, tolerance=0.3):
    """
    Verifica se preço está extrapolando além da região confiável.
    """

    allowed_upper_bound = observed_max + tolerance * (observed_max - observed_min)

    if price_candidate > allowed_upper_bound:
        return {
            "status": "excessive_extrapolation",
            "allowed_upper_bound": allowed_upper_bound
        }
    else:
        return {
            "status": "within_trust_region",
            "allowed_upper_bound": allowed_upper_bound
        }