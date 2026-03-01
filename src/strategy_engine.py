import numpy as np


def enterprise_market_strategy(
    wtp_dict,
    cost_dict,
    prices,
    market_size_dict,
    bundle_products=None
):

    if not wtp_dict:
        raise ValueError("wtp_dict cannot be empty.")

    results = {
        "individual": {},
        "bundle": {
            "enabled": False
        },
        "regime": None
    }

    total_separate_profit = 0
    aggregate_separate_curve = np.zeros(len(prices))

    # ====================================================
    # INDIVIDUAL OPTIMIZATION
    # ====================================================

    for product_id, wtp in wtp_dict.items():

        cost = cost_dict[product_id]
        market_size = market_size_dict[product_id]

        population_size = len(wtp)
        scale_factor = market_size / population_size

        profit_curve = []

        for p in prices:
            demand = np.sum(wtp >= p)
            profit = (p - cost) * demand * scale_factor
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
            "profit_curve": profit_curve,
            "market_size": market_size
        }

    results["separate_total_profit"] = total_separate_profit
    results["profit_separate_curve"] = aggregate_separate_curve

    # ====================================================
    # BUNDLE OPTIMIZATION
    # ====================================================

    if bundle_products and len(bundle_products) >= 2:

        results["bundle"]["enabled"] = True
        results["bundle"]["products"] = bundle_products

        base_array = next(iter(wtp_dict.values()))
        bundle_wtp = np.zeros_like(base_array)

        bundle_cost = 0
        bundle_market_sizes = []

        for product in bundle_products:
            bundle_wtp += wtp_dict[product]
            bundle_cost += cost_dict[product]
            bundle_market_sizes.append(market_size_dict[product])

        # Conservador: bundle TAM = menor TAM entre produtos
        bundle_market_size = min(bundle_market_sizes)

        population_size = len(bundle_wtp)
        scale_factor = bundle_market_size / population_size

        bundle_profit_curve = []

        for p in prices:
            demand = np.sum(bundle_wtp >= p)
            profit = (p - bundle_cost) * demand * scale_factor
            bundle_profit_curve.append(profit)

        bundle_profit_curve = np.array(bundle_profit_curve)

        optimal_idx = np.argmax(bundle_profit_curve)
        optimal_price = prices[optimal_idx]
        optimal_profit = bundle_profit_curve[optimal_idx]

        incremental_gain = optimal_profit - total_separate_profit

        incremental_gain_pct = (
            incremental_gain / total_separate_profit
            if total_separate_profit != 0
            else 0
        )

        regime = (
            "bundle_dominates"
            if incremental_gain > 0
            else "separate_dominates"
        )

        results["bundle"].update({
            "optimal_price": optimal_price,
            "optimal_profit": optimal_profit,
            "profit_curve": bundle_profit_curve,
            "incremental_gain": incremental_gain,
            "incremental_gain_pct": incremental_gain_pct,
            "market_size": bundle_market_size
        })

        results["regime"] = regime
        results["profit_bundle_curve"] = bundle_profit_curve

    else:
        results["regime"] = "single_product"
        results["profit_bundle_curve"] = None

    return results