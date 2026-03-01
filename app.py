import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.elasticity_engine import (
    fit_structural_model,
    elasticity_stat_diagnostics
)

from src.structural_market import (
    calibrate_wtp_distribution,
    generate_calibrated_population
)

from src.strategy_engine import enterprise_market_strategy

from matplotlib.ticker import FuncFormatter

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #2ECC71;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    div.stButton > button:first-child:hover {
        background-color: #27AE60;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
st.title("Product & Bundle Pricing Optimizer")

mode = st.sidebar.radio(
    "Select Mode",
    ["Upload Dataset (Econometric Estimation)", "Quick Structural Simulation"],
    index=0
)

# =====================================================
# MODE 1 — QUICK STRUCTURAL SIMULATION (REFATORADO)
# =====================================================

if mode == "Quick Structural Simulation":

    if "result" not in st.session_state:
        st.session_state["result"] = None
        st.session_state["prices"] = None

    st.sidebar.header("Quick Simulation Setup")

    # ==============================
    # Número de Produtos
    # ==============================

    num_products = st.sidebar.number_input(
        "Number of Products",
        min_value=1,
        max_value=5,
        value=2,
        step=1
    )

    wtp_dict = {}
    cost_dict = {}
    market_size_dict = {}
    for i in range(num_products):

        st.sidebar.subheader(f"Product {i+1}")

        mean_price = st.sidebar.number_input(
            "Mean Price",
            value=150.0 + (i * 20),
            key=f"mean_price_{i}"
        )

        elasticity = st.sidebar.number_input(
            "Elasticity",
            value=-1.5,
            key=f"elasticity_{i}"
        )

        cost = st.sidebar.number_input(
            "Cost",
            value=60.0,
            key=f"cost_{i}"
        )
        market_size = st.sidebar.slider(
            "Addressable Market",
            min_value=1_000,
            max_value=10_000_000,
            value=100_000,
            step=10_000,
            key=f"tam_{i}"
        )

        st.sidebar.markdown(
            f"<div style='font-size:14px; color:#BBBBBB;'>"
            f"Selected TAM: <b>{market_size:,.0f}</b> consumers"
            f"</div>",
            unsafe_allow_html=True
        )

        mu, sigma = calibrate_wtp_distribution(mean_price, elasticity)
        wtp = generate_calibrated_population(mu, sigma)

        product_id = f"P{i+1}"

        wtp_dict[product_id] = wtp
        cost_dict[product_id] = cost
        market_size_dict[product_id] = market_size

    # ==============================
    # Bundle Option
    # ==============================

    bundle_enabled = st.sidebar.checkbox("Enable Bundle Analysis")

    bundle_products = None

    if bundle_enabled and num_products >= 2:
        bundle_products = st.sidebar.multiselect(
            "Select Products for Bundle",
            list(wtp_dict.keys()),
            default=list(wtp_dict.keys())[:2]
        )

        if len(bundle_products) < 2:
            st.sidebar.warning("Select at least 2 products for bundle.")
            bundle_products = None

    # ==============================
    # Run Simulation
    # ==============================

    if st.button("Run Simulation"):

        min_price = min([150.0]) * 0.3
        max_price = max([150.0 + (num_products * 20)]) * 2.5
        mean_prices = [
            st.session_state.get(f"mean_price_{i}", 150.0)
            for i in range(num_products)
        ]

        costs = [
            st.session_state.get(f"cost_{i}", 60.0)
            for i in range(num_products)
        ]

        min_reference = min(min(mean_prices), min(costs))
        max_reference = max(max(mean_prices), max(costs))

        min_price = max(1, min_reference * 0.1)
        max_price = max_reference * 3

        prices = np.linspace(min_price, max_price, 400)

        result = enterprise_market_strategy(
            wtp_dict,
            cost_dict,
            prices,
            market_size_dict=market_size_dict,
            bundle_products=bundle_products
        )

        st.session_state["result"] = result
        st.session_state["prices"] = prices

    # =====================================================
    # RENDER RESULTS
    # =====================================================

    if st.session_state.get("result") is not None:

        result = st.session_state["result"]
        prices = st.session_state["prices"]

        col1, col2 = st.columns([1.2, 2])

        # ==============================
        # LEFT COLUMN (Metrics)
        # ==============================

        with col1:

            st.subheader("Strategy Recommendation")

            # INDIVIDUAL PRODUCTS
            for product_id, data in result["individual"].items():
                st.markdown(f"### {product_id}")
                st.metric(
                    "Optimal Price",
                    f"${data['optimal_price']:,.2f}"
                )
                st.metric(
                    "Optimal Profit",
                    f"${data['optimal_profit']:,.2f}"
                )
                st.markdown("---")

            # BUNDLE
            if result["bundle"]["enabled"]:

                st.markdown("### Bundle")

                st.metric(
                    "Bundle Optimal Price",
                    f"${result['bundle']['optimal_price']:,.2f}"
                )

                st.metric(
                    "Bundle Profit",
                    f"${result['bundle']['optimal_profit']:,.2f}"
                )

                st.metric(
                    "Incremental Gain",
                    f"${result['bundle']['incremental_gain']:,.2f}"
                )

            # REGIME
            regime = result["regime"]

            regime_map = {
                "bundle_dominates": "Bundle Dominates",
                "separate_dominates": "Separate Pricing Dominates",
                "single_product": "Single Product Analysis"
            }

        # ==============================
        # RIGHT COLUMN (Chart)
        # ==============================

        with col2:

            fig, ax = plt.subplots(figsize=(8, 4.5))

            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")

            colors = ["#4DA3FF", "#2ECC71", "#E74C3C", "#9B59B6", "#F1C40F"]

            # INDIVIDUAL CURVES
            for idx, (product_id, data) in enumerate(result["individual"].items()):

                curve = data["profit_curve"]

                ax.plot(
                    prices,
                    curve,
                    label=f"{product_id} Profit",
                    color=colors[idx % len(colors)],
                    linewidth=2.2
                )

                opt_price = data["optimal_price"]
                opt_profit = data["optimal_profit"]

                ax.scatter(
                    opt_price,
                    opt_profit,
                    color=colors[idx % len(colors)],
                    s=80,
                    zorder=5
                )
                ax.annotate(
                    f"${opt_price:,.2f}",
                    (opt_price, opt_profit),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color=colors[idx % len(colors)],
                    fontsize=9,
                    fontweight="bold"
                )

            # BUNDLE CURVE
            if result["bundle"]["enabled"]:

                bundle_curve = result["bundle"]["profit_curve"]

                ax.plot(
                    prices,
                    bundle_curve,
                    label="Bundle Profit",
                    color="#FFFFFF",
                    linewidth=2.5,
                    linestyle="--"
                )

                ax.scatter(
                    result["bundle"]["optimal_price"],
                    result["bundle"]["optimal_profit"],
                    color="#FFFFFF",
                    s=100,
                    zorder=6
                )
                ax.annotate(
                    f"${result['bundle']['optimal_price']:,.2f}",
                    (
                        result["bundle"]["optimal_price"],
                        result["bundle"]["optimal_profit"]
                    ),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color="#FFFFFF",
                    fontsize=9,
                    fontweight="bold"
                )
            ax.axhline(
                0,
                linestyle="--",
                linewidth=1,
                alpha=0.3,
                color="white"
            )
            ax.set_xlabel("Price", color="white")
            ax.set_ylabel("Profit", color="white")
            ax.tick_params(colors="white")

            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"${x:,.0f}")
            )

            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, pos: f"${x:,.0f}")
            )

            ax.legend(
                facecolor="#0E1117",
                edgecolor="white",
                labelcolor="white"
            )

            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")

            st.pyplot(fig, use_container_width=False)

            friendly_regime = regime_map.get(regime, regime)

            st.info(f"Regime: {friendly_regime}")
# =====================================================
# MODE 2 — DATA UPLOAD
# =====================================================

if mode == "Upload Dataset (Econometric Estimation)":
# =====================================================
# TEMPLATE DOWNLOAD SECTION
# =====================================================

    st.markdown("📄 Download Data Template")

    template_df = pd.DataFrame({
        "product_id": ["A", "A", "B", "B"],
        "product_name": ["Product A", "Product A", "Product B", "Product B"],
        "price": [150, 160, 180, 170],
        "quantity": [100, 95, 80, 85],
        "cac": [20, 22, 25, 24],
        "promotion_flag": [0, 1, 0, 1],
        "month": ["2024-01", "2024-02", "2024-01", "2024-02"]
    })

    csv_template = template_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV Template",
        data=csv_template,
        file_name="pricing_optimizer_template.csv",
        mime="text/csv"
    )

    st.markdown(
        """
        **Required Columns Explanation:**
        
        - `product_id`: Unique identifier for each product
        - `product_name`: Unique description for each product 
        - `price`: Selling price  
        - `quantity`: Units sold  
        - `cac`: Customer acquisition cost  
        - `promotion_flag`: 1 if promotion active, 0 otherwise  
        - `month`: Time period (YYYY-MM format recommended)
        """
    )

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")

        # Normaliza nomes das colunas
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
        )

        # Colunas que precisam ser numéricas
        numeric_cols = [
            "price",
            "quantity",
            "cac",
            "promotion_flag"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)  # corrige vírgula decimal
                    .str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove linhas inválidas
        df = df.dropna(subset=numeric_cols)

        required_cols = [
            "product_id",
            "product_name",
            "price",
            "quantity",
            "cac",
            "promotion_flag",
            "month"
        ]

        if not all(col in df.columns for col in required_cols):
            st.error("Dataset missing required columns.")
        else:

            # Cria mapping ID -> Name
            product_mapping = (
                df[["product_id", "product_name"]]
                .drop_duplicates()
                .set_index("product_name")["product_id"]
                .to_dict()
            )
            id_to_name = {v: k for k, v in product_mapping.items()}
            product_names = list(product_mapping.keys())

            st.markdown("## Product Selection")

            selected_product_names = st.multiselect(
                "Select Products for Analysis",
                options=product_names
            )

            # Converte nomes selecionados para IDs
            selected_products = [
                product_mapping[name] for name in selected_product_names
            ]

            if len(selected_products) > 0:

                st.markdown("---")
                st.markdown("## Product Configuration")

                wtp_dict = {}
                cost_dict = {}
                market_size_dict = {}

                diagnostics = {}

                prices = np.linspace(
                    df["price"].min() * 0.5,
                    df["price"].max() * 1.5,
                    200
                )

                for product in selected_products:

                    st.markdown(f"### {id_to_name.get(product, product)}")

                    cost = st.number_input(
                        "Unit Cost",
                        value=50.0,
                        key=f"cost_upload_{product}"
                    )

                    tam_scale = st.selectbox(
                        "Addressable Market Scale",
                        ["Small Market (<50k)", "Medium Market (50k–500k)", "Large Market (500k+)"],
                        key=f"tam_scale_{product}"
                    )

                    if tam_scale == "Small Market (<50k)":
                        tam = st.slider(
                            "Addressable Market",
                            min_value=1_000,
                            max_value=50_000,
                            value=10_000,
                            step=1_000,
                            key=f"tam_small_{product}"
                        )

                    elif tam_scale == "Medium Market (50k–500k)":
                        tam = st.slider(
                            "Addressable Market",
                            min_value=50_000,
                            max_value=500_000,
                            value=100_000,
                            step=10_000,
                            key=f"tam_medium_{product}"
                        )

                    else:
                        tam = st.slider(
                            "Addressable Market",
                            min_value=500_000,
                            max_value=10_000_000,
                            value=1_000_000,
                            step=100_000,
                            key=f"tam_large_{product}"
                        )

                    st.caption(f"Selected TAM: {tam:,.0f} consumers")

                    model = fit_structural_model(df, "product_id", product)
                    diag = elasticity_stat_diagnostics(model)

                    diagnostics[product] = diag

                    mean_price = df[df["product_id"] == product]["price"].mean()

                    mu, sigma = calibrate_wtp_distribution(
                        mean_price,
                        diag["elasticity"]
                    )

                    wtp = generate_calibrated_population(mu, sigma)

                    wtp_dict[product] = wtp
                    cost_dict[product] = cost
                    market_size_dict[product] = tam

                    st.markdown("---")

                # ----------------------------
                # BUNDLE
                # ----------------------------

                enable_bundle = st.checkbox("Enable Bundle Analysis")

                bundle_products = None

                if enable_bundle:

                    bundle_names = st.multiselect(
                        "Select Products for Bundle",
                        options=selected_product_names  # apenas os já selecionados
                    )

                    bundle_products = [
                        product_mapping[name]
                        for name in bundle_names
                    ]

                # ----------------------------
                # RUN ENGINE
                # ----------------------------

                if st.button("Run Econometric Structural Analysis"):

                    result = enterprise_market_strategy(
                        wtp_dict,
                        cost_dict,
                        prices,
                        market_size_dict=market_size_dict,
                        bundle_products=bundle_products
                    )

                    # =====================================================
                    # ELASTICITY DIAGNOSTICS
                    # =====================================================

                    st.markdown("## Elasticity Diagnostics")

                    for product, diag in diagnostics.items():

                        confidence = (1 - diag["p_value"]) * 100

                        st.markdown(f"### {id_to_name.get(product, product)}")

                        c1, c2, c3 = st.columns(3)

                        c1.metric("Elasticity", f"{diag['elasticity']:.2f}")
                        c2.metric("Confidence", f"{confidence:.2f}%")
                        c3.metric("Regime", diag["economic_regime"].capitalize())

                        if abs(diag["elasticity"]) > 1:
                            st.info("Demand is elastic — price changes strongly impact sales.")
                        else:
                            st.info("Demand is inelastic — price changes have limited impact.")
                        st.markdown("---")

                    # =====================================================
                    # STRATEGY RECOMMENDATION
                    # =====================================================

                    col1, col2 = st.columns([1, 1.6])

                    # LEFT COLUMN
                    with col1:

                        st.markdown("## Strategy Recommendation")

                        for product_id, data in result["individual"].items():

                            product_name = id_to_name.get(product_id, product_id)

                            st.markdown(f"### {product_name}")

                            st.metric(
                                "Optimal Price",
                                f"${data['optimal_price']:,.2f}"
                            )

                            st.metric(
                                "Optimal Profit",
                                f"${data['optimal_profit']:,.2f}"
                            )

                            st.markdown("---")

                        if result["bundle"]["enabled"]:

                            st.markdown("### Bundle")

                            st.metric(
                                "Bundle Optimal Price",
                                f"${result['bundle']['optimal_price']:,.2f}"
                            )

                            st.metric(
                                "Bundle Profit",
                                f"${result['bundle']['optimal_profit']:,.2f}"
                            )

                            st.metric(
                                "Incremental Gain",
                                f"${result['bundle']['incremental_gain']:,.2f}"
                            )

                    # RIGHT COLUMN — GRAPH
                    with col2:

                        fig, ax = plt.subplots(figsize=(8, 4.5))

                        fig.patch.set_facecolor("#0E1117")
                        ax.set_facecolor("#0E1117")

                        colors = ["#4DA3FF", "#2ECC71", "#FF5733", "#9B59B6"]

                        for i, (product_id, data) in enumerate(result["individual"].items()):
                            product_name = id_to_name.get(product_id, product_id)
                            curve = data["profit_curve"]
                            color = colors[i % len(colors)]

                            ax.plot(
                                prices,
                                curve,
                                label=f"{product_name} Profit",
                                color=color,
                                linewidth=2.5
                            )

                            idx = np.argmax(curve)
                            price_opt = prices[idx]
                            profit_opt = curve[idx]

                            ax.scatter(price_opt, profit_opt, color=color, s=90, zorder=5)

                            ax.annotate(
                                f"${price_opt:,.2f}",
                                (price_opt, profit_opt),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha='center',
                                color=color,
                                fontsize=9,
                                fontweight="bold"
                            )

                        # Bundle curve
                        if result["bundle"]["enabled"]:

                            bundle_curve = result["bundle"]["profit_curve"]

                            ax.plot(
                                prices,
                                bundle_curve,
                                linestyle="--",
                                linewidth=2.5,
                                label="Bundle Profit",
                                color="white"
                            )

                            idx = np.argmax(bundle_curve)
                            price_opt = prices[idx]
                            profit_opt = bundle_curve[idx]

                            ax.scatter(price_opt, profit_opt, color="white", s=90, zorder=5)

                            ax.annotate(
                                f"${price_opt:,.2f}",
                                (price_opt, profit_opt),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha='center',
                                color="white",
                                fontsize=9,
                                fontweight="bold"
                            )

                        ax.axhline(
                            0,
                            linestyle="--",
                            alpha=0.3,
                            color="#AAAAAA"
                        )

                        ax.set_xlabel("Price", color="white")
                        ax.set_ylabel("Profit", color="white")
                        ax.tick_params(colors="white")

                        ax.yaxis.set_major_formatter(
                            FuncFormatter(lambda x, pos: f"${x:,.0f}")
                        )

                        ax.xaxis.set_major_formatter(
                            FuncFormatter(lambda x, pos: f"${x:,.0f}")
                        )

                        ax.legend(
                            facecolor="#0E1117",
                            edgecolor="white",
                            labelcolor="white"
                        )

                        ax.spines["bottom"].set_color("white")
                        ax.spines["left"].set_color("white")

                        st.pyplot(fig, use_container_width=False)

                    # ----------------------------
                    # REGIME (Below Graph)
                    # ----------------------------

                        st.markdown("---")

                        regime_map = {
                            "bundle_dominates": "Bundle Dominates",
                            "separate_dominates": "Separate Pricing Dominates",
                            "single_product": "Single Product Analysis"
                        }

                        friendly_regime = regime_map.get(result["regime"], result["regime"])

                        st.info(f"Regime: {friendly_regime}")