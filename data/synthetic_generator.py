import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_data(
    n_products=5,
    n_days=180,
    seed=42
):
    np.random.seed(seed)

    start_date = datetime.today() - timedelta(days=n_days)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    data = []

    for product_id in range(1, n_products + 1):

        # Características fixas do produto
        base_price = np.random.uniform(50, 300)
        cost = base_price * np.random.uniform(0.3, 0.6)
        elasticity = np.random.uniform(-2.5, -0.8)
        base_demand = np.random.uniform(50, 200)

        for date in dates:

            price = base_price * np.random.uniform(0.8, 1.2)
            cac = np.random.uniform(5, 30)

            seasonal_factor = 1 + 0.2 * np.sin(
                date.timetuple().tm_yday / 365 * 2 * np.pi
            )

            promotion_flag = np.random.choice([0, 1], p=[0.85, 0.15])
            promo_effect = 1.3 if promotion_flag == 1 else 1.0

            quantity = (
                base_demand
                * (price / base_price) ** elasticity
                * seasonal_factor
                * promo_effect
                * np.random.normal(1, 0.1)
            )

            quantity = max(0, quantity)

            data.append([
                date,
                product_id,
                price,
                cost,
                cac,
                quantity,
                promotion_flag
            ])

    df = pd.DataFrame(data, columns=[
        "date",
        "product_id",
        "price",
        "cost",
        "cac",
        "quantity",
        "promotion_flag"
    ])

    return df