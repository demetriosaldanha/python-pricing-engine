import numpy as np


def enterprise_market_strategy(
    wtp_dict,
    cost_dict,
    prices,
    bundle_products=None
):

    results = {
        "individual": {},
        "bundle": {
            "enabled": False
        },
        "regime": None
    }

    # ====================================================
    # INDIVIDUAL OPTIMIZATION (FOR N PRODUCTS)
    # ====================================================

    total_separate_profit = 0
    aggregate_separate_curve = np.zeros(len(prices))

    for product_id, wtp in wtp_dict.items():

        cost = cost_dict[product_id]

        profit_curve = []

        for p in prices:
            demand = np.sum(wtp >= p)
            profit = (p - cost) * demand
            profit_curve.append(profit)

        profit_curve = np.array(profit_curve)

        optimal_idx = np.argmax(profit_curve)
        optimal_price = prices[optimal_idx]
        optimal_profit = profit_curve[optimal_idx]

        total_separate_profit += optimal_profit
        aggregate_separate_curve += profit_curve

        results["individual"][product_id] = {
            "optimal_price": optimal_price,
            "optimal_profit": optimal_profit,
            "profit_curve": profit_curve
        }

    results["separate_total_profit"] = total_separate_profit
    results["profit_separate_curve"] = aggregate_separate_curve

    # ====================================================
    # BUNDLE OPTIMIZATION (OPTIONAL)
    # ====================================================

    if bundle_products and len(bundle_products) >= 2:

        results["bundle"]["enabled"] = True
        results["bundle"]["products"] = bundle_products

        # Aggregate WTP
        base_array = next(iter(wtp_dict.values()))
        bundle_wtp = np.zeros_like(base_array)

        for product in bundle_products:
            bundle_wtp += wtp_dict[product]

        bundle_cost = sum(cost_dict[p] for p in bundle_products)

        bundle_profit_curve = []

        for p in prices:
            demand = np.sum(bundle_wtp >= p)
            profit = (p - bundle_cost) * demand
            bundle_profit_curve.append(profit)

        bundle_profit_curve = np.array(bundle_profit_curve)

        optimal_idx = np.argmax(bundle_profit_curve)
        optimal_price = prices[optimal_idx]
        optimal_profit = bundle_profit_curve[optimal_idx]

        incremental_gain = optimal_profit - total_separate_profit

        if total_separate_profit != 0:
            incremental_gain_pct = incremental_gain / total_separate_profit
        else:
            incremental_gain_pct = 0

        # Regime
        if incremental_gain > 0:
            regime = "bundle_dominates"
        else:
            regime = "separate_dominates"

        results["bundle"].update({
            "optimal_price": optimal_price,
            "optimal_profit": optimal_profit,
            "profit_curve": bundle_profit_curve,
            "incremental_gain": incremental_gain,
            "incremental_gain_pct": incremental_gain_pct
        })

        results["regime"] = regime
        results["profit_bundle_curve"] = bundle_profit_curve

    else:
        # Single product or no bundle
        results["regime"] = "single_product"
        results["profit_bundle_curve"] = None

    return results