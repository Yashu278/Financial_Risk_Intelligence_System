# app.py — Phase 7: Streamlit UI

from __future__ import annotations

import numpy as np
import streamlit as st

from src.analyze_user import analyze_user
from src.fintalkbot import (
    PROVIDER_MODELS,
    get_default_model,
    get_financial_advice,
    resolve_api_key,
    test_api_key,
)


st.set_page_config(
    page_title="AI Financial Risk & Intelligence System",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI-Driven Personal Financial Risk & Intelligence System")
st.caption("Behavioral analysis system. Not a substitute for professional financial advice.")

if st.sidebar.button("🔄 Reset Session"):
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How to use:**\n1. Fill in Tab 1\n2. Click Analyze\n3. Review risk, trends, and simulation\n4. Use FinTalkBot in Tab 2 for final action guidance"
)

tab1, tab2 = st.tabs(["📊 Financial Analysis", "💬 FinTalkBot (Final Decision Layer)"])


def build_profile(result, income_list, expense_list, avg_income, avg_expense):
    neg_months = sum(1 for income, expense in zip(income_list, expense_list) if expense > income)
    return {
        "risk_label": result.get("risk_label", "Unknown"),
        "expense_trend": result.get("expense_trend", "Unknown"),
        "savings_trend": result.get("savings_trend", "Unknown"),
        "avg_income": avg_income,
        "expense_ratio_mean": avg_expense / avg_income if avg_income > 0 else 0,
        "neg_savings_freq": neg_months / len(income_list) if income_list else 0,
        "mc_summary": result.get("mc_summary", "Not available"),
    }


def latest_user_message(chat_history):
    for message in reversed(chat_history):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


with tab1:
    st.header("Enter Your Financial Information")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Income & Expenses")
        avg_income = st.number_input("Average Monthly Income (₹)", min_value=5000, value=50000, step=1000)
        avg_expense = st.number_input("Average Monthly Expenses (₹)", min_value=1000, value=40000, step=1000)

    with col_b:
        st.subheader("Investment Parameters")
        investment_amount = st.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=1000)
        annual_return = st.slider("Expected Annual Return (%)", 4.0, 20.0, 8.0) / 100
        volatility = st.slider("Market Volatility (%)", 5.0, 30.0, 15.0) / 100
        years = st.slider("Investment Period (Years)", 1, 20, 5)

    st.divider()

    if st.button("🔍 Analyze My Financial Profile", use_container_width=True):
        if avg_expense >= avg_income * 1.5:
            st.warning("⚠️ Expenses are significantly higher than income. Analysis will still run.")

        income_list = [max(1000, avg_income + np.random.normal(0, avg_income * 0.05)) for _ in range(24)]
        expense_list = [max(500, avg_expense + np.random.normal(0, avg_expense * 0.05)) for _ in range(24)]

        with st.spinner("Running analysis — please wait (forecasting may take a few seconds)..."):
            try:
                result = analyze_user(
                    income_list,
                    expense_list,
                    investment_amount,
                    annual_return,
                    volatility,
                    years,
                )
                st.session_state["result"] = result
                st.session_state["profile"] = build_profile(
                    result,
                    income_list,
                    expense_list,
                    avg_income,
                    avg_expense,
                )
                st.session_state["chat_history"] = []
                st.success("✅ Analysis complete! Scroll down to view results.")
            except ValueError as exc:
                st.error(f"Input Error: {exc}")
            except Exception:
                st.error("Analysis failed. Please try again.")

    if "result" in st.session_state:
        result = st.session_state["result"]

        st.divider()
        st.subheader("Your Financial Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            risk = result.get("risk_label", "Unknown")
            color = "red" if risk == "High" else ("orange" if risk == "Medium" else "green")
            st.markdown("**Risk Category**")
            st.markdown(f"<h2 style='color:{color}'>{risk} Risk</h2>", unsafe_allow_html=True)

        with col2:
            exp_trend = result.get("expense_trend", "Forecast Error")
            icon = "🔴" if exp_trend == "Deteriorating" else ("🟢" if exp_trend == "Improving" else "🟡")
            st.markdown("**Expense Trend (Next 6 Months)**")
            st.markdown(f"<h3>{icon} {exp_trend}</h3>", unsafe_allow_html=True)

        with col3:
            sav_trend = result.get("savings_trend", "Forecast Error")
            icon = "🔴" if sav_trend == "Deteriorating" else ("🟢" if sav_trend == "Improving" else "🟡")
            st.markdown("**Savings Trend (Next 6 Months)**")
            st.markdown(f"<h3>{icon} {sav_trend}</h3>", unsafe_allow_html=True)

        st.divider()

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("📈 Expense Forecast")
            if result.get("expense_fig") is not None:
                st.pyplot(result["expense_fig"])
            else:
                st.info("Expense forecast unavailable for this input.")

        with col_right:
            st.subheader("💰 Savings Forecast")
            if result.get("savings_fig") is not None:
                st.pyplot(result["savings_fig"])
            else:
                st.info("Savings forecast unavailable for this input.")

        st.divider()
        st.subheader("🎲 Investment Simulation (Monte Carlo — 10,000 paths)")
        if result.get("mc_fig") is not None:
            st.pyplot(result["mc_fig"])
        else:
            st.info("Monte Carlo simulation unavailable.")

        if result.get("mc_summary"):
            st.info(result["mc_summary"])
        else:
            st.info("Monte Carlo summary unavailable.")

        st.divider()
        st.caption("➡️ Switch to the **FinTalkBot** tab for final decision guidance based on your full profile.")


with tab2:
    st.header("💬 FinTalkBot — Final Decision Layer")
    st.caption("Uses your computed risk label, forecast trends, and Monte Carlo outcomes to guide next actions.")

    if "profile" not in st.session_state:
        st.warning("⚠️ Complete the Financial Analysis in Tab 1 first. FinTalkBot needs your profile.")
        st.stop()

    profile = st.session_state["profile"]
    risk = profile.get("risk_label", "Unknown")
    color = "red" if risk == "High" else ("orange" if risk == "Medium" else "green")

    st.markdown(
        f"**Your Profile:** "
        f"<span style='color:{color}; font-weight:bold'>{risk} Risk</span> | "
        f"Expense: **{profile.get('expense_trend', 'Unknown')}** | "
        f"Savings: **{profile.get('savings_trend', 'Unknown')}**",
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("🔑 AI Provider Settings")
    st.caption(
        "Enter your own key (optional). If empty, the app tries the system demo key from server environment."
    )

    provider_col, model_col = st.columns([1, 1])
    with provider_col:
        provider = st.selectbox("AI Provider", ["None (Rule-based)", "Gemini", "OpenAI", "Anthropic", "Groq"])
    with model_col:
        if provider in PROVIDER_MODELS:
            model = st.selectbox("Model", PROVIDER_MODELS[provider], index=0)
        else:
            model = ""
            st.selectbox("Model", ["Rule-based only"], index=0, disabled=True)

    manual_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your own key (optional)",
    )

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    api_key = manual_key.strip()
    resolved_key, key_source = resolve_api_key(api_key, provider)

    if provider != "None (Rule-based)" and resolved_key:
        if key_source == "user":
            st.info(f"Using {provider} with your entered key.")
        else:
            st.info(f"Using {provider} with system demo key (limited usage).")
    elif provider != "None (Rule-based)" and not resolved_key:
        st.warning(f"No key available for {provider}. The chatbot will use fallback mode.")

    if provider in PROVIDER_MODELS and st.button("Test Connection"):
        ok, msg, source = test_api_key(provider, api_key, model or get_default_model(provider))
        if ok:
            st.success(msg)
        else:
            st.error(msg)
        st.caption(f"Connection key source: {source}")

    st.session_state["provider"] = provider
    st.session_state["api_key"] = api_key
    st.session_state["model"] = model
    st.session_state["key_source"] = key_source

    st.divider()
    st.subheader("Ask a Financial Question")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask a financial question...")

    if user_input:
        user_input = user_input.strip()
        if not user_input:
            st.stop()

        last_user = latest_user_message(st.session_state["chat_history"])
        if user_input == last_user:
            st.stop()

        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("FinTalkBot is thinking..."):
            response, mode, key_source = get_financial_advice(
                user_input,
                profile,
                provider,
                api_key,
                model=model or get_default_model(provider),
                chat_history=st.session_state["chat_history"][:-1],
            )

        if mode == "AI":
            st.caption(f"Mode: AI ({provider} | {model} | key source: {key_source})")
        else:
            st.caption("Mode: Rule-based fallback")

        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)