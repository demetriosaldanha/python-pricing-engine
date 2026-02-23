import numpy as np
import statsmodels.api as sm


# ============================
# MODEL FITTING
# ============================

def fit_structural_model(df, entity_column, entity_value):
    """
    Ajusta modelo log-log para entidade específica.
    """

    df_entity = df[df[entity_column] == entity_value].copy()

    df_entity["log_price"] = np.log(df_entity["price"])
    df_entity["log_quantity"] = np.log(df_entity["quantity"])

    X = df_entity[["log_price", "cac", "promotion_flag", "month"]]
    X = sm.add_constant(X)

    y = df_entity["log_quantity"]

    model = sm.OLS(y, X).fit()

    return model


# ============================
# ELASTICITY DIAGNOSTICS
# ============================

def elasticity_stat_diagnostics(model):
    """
    Retorna diagnóstico estatístico da elasticidade.
    """

    beta = model.params["log_price"]
    p_value = model.pvalues["log_price"]

    if p_value < 0.05:
        reliability = "significant"
    else:
        reliability = "not_significant"

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