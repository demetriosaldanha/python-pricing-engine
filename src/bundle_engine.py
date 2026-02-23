import numpy as np
from .structural_market import (
    profit_individual,
    profit_bundle
)


def optimal_individual_price(wtp, cost, price_range):
    profits = [
        profit_individual(p, wtp, cost)
        for p in price_range
    ]
    
    idx = np.argmax(profits)
    return price_range[idx], profits[idx]


def optimal_bundle_price(wtp_A, wtp_B, cost_bundle, price_range):
    profits = [
        profit_bundle(p, wtp_A, wtp_B, cost_bundle)
        for p in price_range
    ]
    
    idx = np.argmax(profits)
    return price_range[idx], profits[idx]