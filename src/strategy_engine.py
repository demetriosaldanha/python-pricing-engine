import numpy as np
from src.bundle_engine import optimal_individual_price, optimal_bundle_price


def enterprise_market_strategy(wtp_A, wtp_B, cost_A, cost_B, prices):

    # =========================
    # INDIVIDUAL OPTIMIZATION
    # =========================

    opt_A_price, opt_A_profit = optimal_individual_price(
        wtp_A, cost_A, prices
    )

    opt_B_price, opt_B_profit = optimal_individual_price(
        wtp_B, cost_B, prices
    )

    total_separate_profit = opt_A_profit + opt_B_profit

    # =========================
    # BUNDLE OPTIMIZATION
    # =========================

    opt_bundle_price, opt_bundle_profit = optimal_bundle_price(
        wtp_A,
        wtp_B,
        cost_A + cost_B,
        prices
    )

    incremental_gain = opt_bundle_profit - total_separate_profit

    if total_separate_profit != 0:
        gain_pct = incremental_gain / total_separate_profit
    else:
        gain_pct = 0

    # =========================
    # REGIME CLASSIFICATION
    # =========================

    if incremental_gain > 0:
        regime = "bundle_dominates"
    else:
        regime = "separate_dominates"

    # =========================
    # PROFIT CURVES
    # =========================

    profit_separate_curve = []
    profit_bundle_curve = []

    for p in prices:

        demand_A = np.sum(wtp_A >= p)
        demand_B = np.sum(wtp_B >= p)

        profit_A = (p - cost_A) * demand_A
        profit_B = (p - cost_B) * demand_B

        profit_separate_curve.append(profit_A + profit_B)

        bundle_demand = np.sum((wtp_A + wtp_B) >= p)
        bundle_profit_at_p = (p - (cost_A + cost_B)) * bundle_demand

        profit_bundle_curve.append(bundle_profit_at_p)

    # =========================
    # FINAL RETURN
    # =========================

    return {
        # diagnóstico
        "regime": regime,

        # ótimos
        "optimal_A_price": opt_A_price,
        "optimal_B_price": opt_B_price,
        "bundle_price": opt_bundle_price,

        # lucros
        "separate_profit": total_separate_profit,
        "bundle_profit": opt_bundle_profit,

        # incremental
        "incremental_gain": incremental_gain,
        "incremental_gain_pct": gain_pct,

        # curvas
        "profit_separate_curve": profit_separate_curve,
        "profit_bundle_curve": profit_bundle_curve
    }