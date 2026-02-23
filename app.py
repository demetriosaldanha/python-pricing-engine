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
# MODE 1 — QUICK STRUCTURAL SIMULATION
# =====================================================

if mode == "Quick Structural Simulation":

    # limpa resultado se trocar de modo
    if "result" not in st.session_state:
        st.session_state["result"] = None
        st.session_state["prices"] = None

    st.sidebar.header("Product Parameters")

    st.sidebar.subheader("Product A")
    mean_price_A = st.sidebar.number_input("Mean Price", value=150.0)
    elasticity_A = st.sidebar.number_input("Elasticity", value=-1.5)
    cost_A = st.sidebar.number_input("Cost", value=60.0)

    st.sidebar.subheader("Product B")
    mean_price_B = st.sidebar.number_input("Mean Price", value=180.0)
    elasticity_B = st.sidebar.number_input("Elasticity", value=-2.0)
    cost_B = st.sidebar.number_input("Cost B", value=50.0)

    if st.button("Run Simulation"):

        mu_A, sigma_A = calibrate_wtp_distribution(mean_price_A, elasticity_A)
        mu_B, sigma_B = calibrate_wtp_distribution(mean_price_B, elasticity_B)

        wtp_A = generate_calibrated_population(mu_A, sigma_A)
        wtp_B = generate_calibrated_population(mu_B, sigma_B)

        prices = np.linspace(50, 400, 200)

        result = enterprise_market_strategy(
            wtp_A,
            wtp_B,
            cost_A,
            cost_B,
            prices
        )

        st.session_state["result"] = result
        st.session_state["prices"] = prices

    # ===== Renderização segura =====

    if st.session_state.get("result") is not None:

        result = st.session_state["result"]
        prices = st.session_state["prices"]

        # segurança extra
        required_keys = [
            "optimal_A_price",
            "optimal_B_price",
            "bundle_price",
            "separate_profit",
            "bundle_profit",
            "profit_separate_curve",
            "profit_bundle_curve"
        ]

        if not all(k in result for k in required_keys):
            st.error("Strategy engine is not returning all required fields.")
        else:

            col1, col2 = st.columns([1.2, 2])

            with col1:
                st.subheader("Optimal Strategy")

                m1, m2 = st.columns(2)
                m3, m4 = st.columns(2)
                m5, m6 = st.columns(2)

                m1.metric("Optimal Price A", f"${result['optimal_A_price']:,.2f}")
                m2.metric("Optimal Price B", f"${result['optimal_B_price']:,.2f}")

                m3.metric("Bundle Price", f"${result['bundle_price']:,.2f}")
                m4.metric("Separate Profit", f"${result['separate_profit']:,.2f}")

                m5.metric("Bundle Profit", f"${result['bundle_profit']:,.2f}")

                gain = result["bundle_profit"] - result["separate_profit"]
                m6.metric("Incremental Gain", f"${gain:,.2f}")

                # Regime amigável
                regime_map = {
                    "bundle_dominates": "Bundle Dominates",
                    "separate_dominates": "Separate Pricing Dominates",
                    "indifferent": "Indifferent Strategy"
                }

                friendly_regime = regime_map.get(result["regime"], result["regime"])

                st.markdown(f"**Regime:** {friendly_regime}")

            with col2:

                fig, ax = plt.subplots(figsize=(10, 6))

                # Dark background
                fig.patch.set_facecolor("#0E1117")
                ax.set_facecolor("#0E1117")

                # Cores modernas dark-friendly
                color_separate = "#4DA3FF"   # azul vibrante
                color_bundle = "#2ECC71"     # verde moderno

                ax.plot(
                    prices,
                    result["profit_separate_curve"],
                    label="Separate Profit",
                    color=color_separate,
                    linewidth=2.5
                )

                ax.plot(
                    prices,
                    result["profit_bundle_curve"],
                    label="Bundle Profit",
                    color=color_bundle,
                    linewidth=2.5
                )

                # Linha vertical no preço ótimo
                optimal_bundle_price = result["bundle_price"]
                optimal_bundle_profit = result["bundle_profit"]

                ax.axvline(
                    optimal_bundle_price,
                    linestyle="--",
                    alpha=0.5,
                    color="#AAAAAA"
                )

                # Ponto ótimo destacado
                ax.scatter(
                    optimal_bundle_price,
                    optimal_bundle_profit,
                    color="#FFFFFF",
                    s=120,
                    zorder=5
                )

                # Label somente do ponto ótimo
                ax.annotate(
                    f"{optimal_bundle_price:.2f}",
                    (optimal_bundle_price, optimal_bundle_profit),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    color="white",
                    fontsize=10,
                    fontweight="bold"
                )

                ax.set_title("Profit Comparison", color="white", fontsize=14)
                ax.set_xlabel("Price", color="white")
                ax.set_ylabel("Profit", color="white")

                ax.tick_params(colors='white')
                ax.legend(facecolor="#0E1117", edgecolor="white", labelcolor="white")

                ax.spines["bottom"].set_color("white")
                ax.spines["top"].set_color("white")
                ax.spines["left"].set_color("white")
                ax.spines["right"].set_color("white")

                st.pyplot(fig)


# =====================================================
# MODE 2 — DATA UPLOAD
# =====================================================

if mode == "Upload Dataset (Econometric Estimation)":

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        required_cols = [
            "product_id",
            "price",
            "quantity",
            "cac",
            "promotion_flag",
            "month"
        ]

        if not all(col in df.columns for col in required_cols):
            st.error("Dataset missing required columns.")
        else:

            product_ids = df["product_id"].unique()

            st.markdown("## Product A Configuration")

            product_A = st.selectbox(
                "Select Product A",
                product_ids
            )

            cost_A = st.number_input(
                "Unit Cost (Cost to Produce 1 Unit)",
                value=60.0,
                key="cost_A"
            )

            st.markdown("---")

            st.markdown("## Product B Configuration")

            product_B = st.selectbox(
                "Select Product B",
                product_ids
            )

            cost_B = st.number_input(
                "Unit Cost (Cost to Produce 1 Unit)",
                value=50.0,
                key="cost_B"
            )

            if st.button("Run Econometric Structural Analysis"):

                # ===== Fit Models =====
                model_A = fit_structural_model(df, "product_id", product_A)
                model_B = fit_structural_model(df, "product_id", product_B)

                diag_A = elasticity_stat_diagnostics(model_A)
                diag_B = elasticity_stat_diagnostics(model_B)

                mean_A = df[df["product_id"] == product_A]["price"].mean()
                mean_B = df[df["product_id"] == product_B]["price"].mean()

                mu_A, sigma_A = calibrate_wtp_distribution(mean_A, diag_A["elasticity"])
                mu_B, sigma_B = calibrate_wtp_distribution(mean_B, diag_B["elasticity"])

                wtp_A = generate_calibrated_population(mu_A, sigma_A)
                wtp_B = generate_calibrated_population(mu_B, sigma_B)

                prices = np.linspace(50, 400, 200)

                result = enterprise_market_strategy(
                    wtp_A,
                    wtp_B,
                    cost_A,
                    cost_B,
                    prices
                )

                # =====================================================
                # ELASTICITY DIAGNOSTICS
                # =====================================================

                st.markdown("## Elasticity Diagnostics")

                dcol1, dcol2 = st.columns(2)

                confidence_A = (1 - diag_A["p_value"]) * 100
                confidence_B = (1 - diag_B["p_value"]) * 100

                with dcol1:
                    st.markdown("### Product A")

                    st.metric(
                        "Price Elasticity",
                        f"{diag_A['elasticity']:.2f}"
                    )

                    st.metric(
                        "Statistical Confidence",
                        f"{confidence_A:.2f}%"
                    )

                    st.metric(
                        "Demand Sensitivity",
                        diag_A["economic_regime"].capitalize()
                    )

                    # Insight interpretativo automático
                    if abs(diag_A["elasticity"]) > 1:
                        st.info("Demand is elastic — price changes strongly impact sales.")
                    else:
                        st.info("Demand is inelastic — price changes have limited impact.")


                with dcol2:
                    st.markdown("### Product B")

                    st.metric(
                        "Price Elasticity",
                        f"{diag_B['elasticity']:.2f}"
                    )

                    st.metric(
                        "Statistical Confidence",
                        f"{confidence_B:.2f}%"
                    )

                    st.metric(
                        "Demand Sensitivity",
                        diag_B["economic_regime"].capitalize()
                    )

                    if abs(diag_B["elasticity"]) > 1:
                        st.info("Demand is elastic — price changes strongly impact sales.")
                    else:
                        st.info("Demand is inelastic — price changes have limited impact.")

                # =====================================================
                # STRATEGY RECOMMENDATION
                # =====================================================

                col1, col2 = st.columns([1, 1.6])

                # LEFT COLUMN
                with col1:

                    st.markdown(
                        """
                        <div style="
                            max-width: 600px;
                            margin-left: auto;
                            margin-right: auto;
                        ">
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("<h2>Strategy Recommendation</h2>", unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    m3, m4 = st.columns(2)
                    m5, m6 = st.columns(2)

                    m1.metric("Optimal Price A", f"${result['optimal_A_price']:,.2f}")
                    m2.metric("Optimal Price B", f"${result['optimal_B_price']:,.2f}")

                    m3.metric("Bundle Price", f"${result['bundle_price']:,.2f}")
                    m4.metric("Separate Profit", f"${result['separate_profit']:,.2f}")

                    m5.metric("Bundle Profit", f"${result['bundle_profit']:,.2f}")
                    m6.metric("Incremental Gain", f"${result['incremental_gain']:,.2f}")

                    regime_map = {
                        "bundle_dominates": "Bundle Dominates",
                        "separate_dominates": "Separate Pricing Dominates",
                        "indifferent": "Indifferent Strategy"
                    }

                    friendly_regime = regime_map.get(result["regime"], result["regime"])

                    if result["regime"] == "bundle_dominates":
                        st.success(f"Regime: {friendly_regime}")
                    elif result["regime"] == "separate_dominates":
                        st.info(f"Regime: {friendly_regime}")
                    else:
                        st.warning(f"Regime: {friendly_regime}")

                    st.markdown("</div>", unsafe_allow_html=True)

                # RIGHT COLUMN
                with col2:

                    # Container fixo para evitar expansão desproporcional
                    st.markdown(
                        """
                        <div style="
                            max-width: 750px;
                            margin-left: auto;
                            margin-right: auto;
                        ">
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "<h2 style='text-align: center;'>Profit Comparison</h2>",
                        unsafe_allow_html=True
                    )

                    fig, ax = plt.subplots(figsize=(8, 4.5))

                    fig.patch.set_facecolor("#0E1117")
                    ax.set_facecolor("#0E1117")

                    color_separate = "#4DA3FF"
                    color_bundle = "#2ECC71"

                    separate_curve = result["profit_separate_curve"]
                    bundle_curve = result["profit_bundle_curve"]

                    # Curvas
                    ax.plot(
                        prices,
                        separate_curve,
                        label="Separate Profit",
                        color=color_separate,
                        linewidth=2.5
                    )

                    ax.plot(
                        prices,
                        bundle_curve,
                        label="Bundle Profit",
                        color=color_bundle,
                        linewidth=2.5
                    )

                    # Pico Separate
                    sep_idx = np.argmax(separate_curve)
                    sep_price = prices[sep_idx]
                    sep_profit = separate_curve[sep_idx]

                    ax.scatter(sep_price, sep_profit, color=color_separate, s=90, zorder=5)

                    ax.annotate(
                        f"${sep_price:,.2f}",
                        (sep_price, sep_profit),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        color=color_separate,
                        fontsize=9,
                        fontweight="bold"
                    )

                    # Pico Bundle
                    bun_idx = np.argmax(bundle_curve)
                    bun_price = prices[bun_idx]
                    bun_profit = bundle_curve[bun_idx]

                    ax.scatter(bun_price, bun_profit, color=color_bundle, s=90, zorder=5)

                    ax.annotate(
                        f"${bun_price:,.2f}",
                        (bun_price, bun_profit),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        color=color_bundle,
                        fontsize=9,
                        fontweight="bold"
                    )

                    # Linha vertical
                    ax.axvline(
                        bun_price,
                        linestyle="--",
                        alpha=0.4,
                        color="#AAAAAA"
                    )

                    # # Estética
                    # ax.set_title(
                    #     "Profit Comparison",
                    #     color="white",
                    #     fontsize=14,
                    #     fontweight="bold",
                    #     pad=10
                    # )

                    ax.set_xlabel("Price", color="white")
                    ax.set_ylabel("Profit", color="white")
                    ax.tick_params(colors="white")

                    ax.yaxis.set_major_formatter(
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

                    # Fecha o div
                    st.markdown("</div>", unsafe_allow_html=True)