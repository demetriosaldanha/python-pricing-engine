import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


# ============================
# WTP CALIBRATION
# ============================

def calibrate_wtp_distribution(mean_price, elasticity):
    """
    Calibra uma distribuição Normal de WTP consistente
    com a elasticidade estimada no preço médio observado.
    """

    def objective(params):
        mu, sigma = params

        # Penalização se sigma inválido
        if sigma <= 0:
            return 1e6

        # Demanda = proporção com WTP >= preço médio
        demand = 1 - norm.cdf(mean_price, mu, sigma)

        # Densidade no ponto
        pdf = norm.pdf(mean_price, mu, sigma)

        # Elasticidade estrutural teórica
        elasticity_model = -(mean_price / demand) * pdf

        return (elasticity_model - elasticity) ** 2

    result = minimize(
        objective,
        x0=[mean_price, mean_price * 0.2],
        method="Nelder-Mead"
    )

    mu_est, sigma_est = result.x

    return mu_est, sigma_est


# ============================
# POPULATION GENERATION
# ============================

def generate_calibrated_population(mu, sigma, n=10000, seed=42):
    """
    Gera população sintética de WTP baseada na distribuição calibrada.
    """
    np.random.seed(seed)
    return np.random.normal(mu, sigma, n)


# ============================
# DEMAND FUNCTIONS
# ============================

def demand_individual(wtp, price):
    """
    Demanda individual como proporção de consumidores
    com WTP >= preço.
    """
    return np.mean(wtp >= price)


def demand_bundle(wtp_A, wtp_B, price_bundle):
    """
    Demanda do bundle como proporção de consumidores
    cuja soma dos WTPs excede o preço do bundle.
    """
    return np.mean((wtp_A + wtp_B) >= price_bundle)


# ============================
# PROFIT FUNCTIONS
# ============================

def profit_individual(price, wtp, cost):
    """
    Lucro individual = (preço - custo) * demanda
    """
    demand = demand_individual(wtp, price)
    return (price - cost) * demand


def profit_bundle(price_bundle, wtp_A, wtp_B, cost_bundle):
    """
    Lucro do bundle = (preço_bundle - custo_bundle) * demanda_bundle
    """
    demand = demand_bundle(wtp_A, wtp_B, price_bundle)
    return (price_bundle - cost_bundle) * demand