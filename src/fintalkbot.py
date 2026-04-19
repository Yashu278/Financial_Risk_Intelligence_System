from __future__ import annotations

import logging
import os
from typing import Callable


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are FinTalkBot, a personal financial advisor.

STRICT RULES:
- You MUST use the user's financial data
- Do NOT give generic advice
- If High Risk, prioritize survival (cut expenses, emergency fund)
- If Low Risk, focus on investments and growth
- Keep response under 150 words
- Stay strictly in personal finance

Respond like you are analyzing THIS user's situation, not giving general advice.
""".strip()


PROVIDER_MODELS = {
    "Gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    "OpenAI": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"],
    "Anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
    "Groq": ["llama3-8b-8192", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
}


SYSTEM_KEY_ENV_MAP = {
    "Gemini": "SYSTEM_GEMINI_API_KEY",
    "OpenAI": "SYSTEM_OPENAI_API_KEY",
    "Anthropic": "SYSTEM_ANTHROPIC_API_KEY",
    "Groq": "SYSTEM_GROQ_API_KEY",
}


LEGACY_KEY_ENV_MAP = {
    "Gemini": "GEMINI_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Groq": "GROQ_API_KEY",
}


def get_default_model(provider: str) -> str:
    options = PROVIDER_MODELS.get(provider, [])
    return options[0] if options else ""


def resolve_api_key(user_key: str, provider: str) -> tuple[str, str]:
    clean_user_key = (user_key or "").strip()
    if clean_user_key:
        return clean_user_key, "user"

    system_key = os.environ.get(SYSTEM_KEY_ENV_MAP.get(provider, ""), "").strip()
    if system_key:
        return system_key, "system"

    legacy_key = os.environ.get(LEGACY_KEY_ENV_MAP.get(provider, ""), "").strip()
    if legacy_key:
        return legacy_key, "system"

    return "", "none"


def build_context(profile: dict) -> str:
    return (
        "User Financial Profile:\n"
        f"- Risk Level: {profile.get('risk_label', 'Unknown')}\n"
        f"- Expense Trend: {profile.get('expense_trend', 'Unknown')}\n"
        f"- Savings Trend: {profile.get('savings_trend', 'Unknown')}\n"
        f"- Avg Income: INR {float(profile.get('avg_income', 0)):,.0f}\n"
        f"- Expense Ratio: {float(profile.get('expense_ratio_mean', 0)):.2f}\n"
        f"- Negative Savings Frequency: {float(profile.get('neg_savings_freq', 0)):.2%}\n\n"
        "Investment Simulation:\n"
        f"{profile.get('mc_summary', 'Not available')}"
    )


def rule_based_advice(question: str, profile: dict) -> str:
    risk = profile.get("risk_label", "Unknown")
    expense_trend = profile.get("expense_trend", "Unknown")
    savings_trend = profile.get("savings_trend", "Unknown")
    avg_income = float(profile.get("avg_income", 0))

    advice_lines = []

    if risk == "High":
        advice_lines.append(
            f"You are in a high-risk position with around INR {avg_income:,.0f} monthly income. "
            "Prioritize survival: cut non-essential spending and build an emergency fund first."
        )
    elif risk == "Medium":
        advice_lines.append(
            f"You are in a medium-risk position with around INR {avg_income:,.0f} monthly income. "
            "Focus on consistent savings and tighter expense planning."
        )
    else:
        advice_lines.append(
            f"You are in a low-risk position with around INR {avg_income:,.0f} monthly income. "
            "You can focus on disciplined investing and long-term growth."
        )

    if expense_trend == "Deteriorating":
        advice_lines.append("Your expense trend is deteriorating, so reduce discretionary spending immediately.")
    elif expense_trend == "Improving":
        advice_lines.append("Your expense trend is improving. Keep this control in place.")

    if savings_trend == "Deteriorating":
        advice_lines.append("Savings are deteriorating, so pause aggressive investing until savings stabilize.")
    elif savings_trend == "Improving":
        advice_lines.append("Savings are improving, which supports gradual investment increases.")

    return (
        f"You asked: {question}\n\n"
        f"Based on your profile ({risk} risk, {expense_trend} expenses, {savings_trend} savings):\n"
        f"{' '.join(advice_lines)}"
    )


def _openai_style_messages(question: str, context: str, chat_history: list[dict] | None = None) -> list[dict]:
    recent = (chat_history or [])[-4:]
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{context}"}]
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def call_gemini(
    question: str,
    context: str,
    api_key: str,
    model: str,
    chat_history: list[dict] | None = None,
) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)
    recent = (chat_history or [])[-4:]
    history_block = []
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and content:
            history_block.append(f"User: {content}")
        elif role == "assistant" and content:
            history_block.append(f"Assistant: {content}")

    prompt = f"{SYSTEM_PROMPT}\n\n{context}"
    if history_block:
        prompt += "\n\nRecent conversation:\n" + "\n".join(history_block)
    prompt += f"\n\nCurrent user question: {question}"

    response = client.generate_content(prompt)
    return (response.text or "").strip()


def call_openai(
    question: str,
    context: str,
    api_key: str,
    model: str,
    chat_history: list[dict] | None = None,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=_openai_style_messages(question, context, chat_history),
        max_tokens=280,
    )
    return (response.choices[0].message.content or "").strip()


def call_anthropic(
    question: str,
    context: str,
    api_key: str,
    model: str,
    chat_history: list[dict] | None = None,
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    recent = (chat_history or [])[-4:]
    messages = []
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    response = client.messages.create(
        model=model,
        max_tokens=280,
        system=f"{SYSTEM_PROMPT}\n\n{context}",
        messages=messages,
    )
    blocks = [block.text for block in response.content if hasattr(block, "text")]
    return "\n".join(blocks).strip()


def call_groq(
    question: str,
    context: str,
    api_key: str,
    model: str,
    chat_history: list[dict] | None = None,
) -> str:
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=_openai_style_messages(question, context, chat_history),
        max_tokens=280,
    )
    return (response.choices[0].message.content or "").strip()


def _classify_error(error: Exception) -> str:
    text = str(error).lower()
    if any(token in text for token in ["invalid", "unauthorized", "authentication", "api key"]):
        return "invalid_key"
    if any(token in text for token in ["quota", "rate limit", "too many requests", "429"]):
        return "quota"
    return "provider_error"


def _error_prefix(error_type: str, provider: str) -> str:
    if error_type == "invalid_key":
        return f"API authentication failed for {provider}. Using fallback advice."
    if error_type == "quota":
        return f"API quota or rate limit reached for {provider}. Using fallback advice."
    return f"API request failed for {provider}. Using fallback advice."


PROVIDER_MAP: dict[str, Callable[..., str]] = {
    "Gemini": call_gemini,
    "OpenAI": call_openai,
    "Anthropic": call_anthropic,
    "Groq": call_groq,
}


def get_financial_advice(
    user_question: str,
    profile: dict,
    provider: str = "None (Rule-based)",
    api_key: str = "",
    model: str | None = None,
    chat_history: list[dict] | None = None,
) -> tuple[str, str, str]:
    context = build_context(profile)
    selected_provider = (provider or "None (Rule-based)").strip()
    clean_key, key_source = resolve_api_key(api_key, selected_provider)
    chosen_model = model or get_default_model(selected_provider)

    if selected_provider in PROVIDER_MAP and clean_key:
        logger.info("[Chatbot] Provider=%s Mode=AI KeySource=%s", selected_provider, key_source)
        try:
            text = PROVIDER_MAP[selected_provider](
                user_question,
                context,
                clean_key,
                chosen_model,
                chat_history,
            )
            if text:
                return text, "AI", key_source
            raise ValueError("Empty response from provider")
        except Exception as exc:
            error_type = _classify_error(exc)
            logger.warning(
                "[Chatbot] Provider=%s Mode=Fallback KeySource=%s ErrorType=%s",
                selected_provider,
                key_source,
                error_type,
            )
            prefix = _error_prefix(error_type, selected_provider)
            return f"({prefix})\n\n{rule_based_advice(user_question, profile)}", "Fallback", key_source

    logger.info("[Chatbot] Provider=%s Mode=Fallback KeySource=%s", selected_provider, key_source)
    return rule_based_advice(user_question, profile), "Fallback", key_source


def test_api_key(provider: str, api_key: str = "", model: str | None = None) -> tuple[bool, str, str]:
    selected_provider = (provider or "").strip()
    clean_key, key_source = resolve_api_key(api_key, selected_provider)

    if selected_provider not in PROVIDER_MAP:
        return False, "Select a supported provider first.", "none"
    if not clean_key:
        return False, "No API key found. Enter one or configure a system key.", "none"

    probe_profile = {
        "risk_label": "Low",
        "expense_trend": "Stable",
        "savings_trend": "Improving",
        "avg_income": 50000,
        "expense_ratio_mean": 0.55,
        "neg_savings_freq": 0.05,
        "mc_summary": "Best case INR 160,000; median INR 125,000; worst case INR 95,000.",
    }
    context = build_context(probe_profile)
    chosen_model = model or get_default_model(selected_provider)

    try:
        PROVIDER_MAP[selected_provider](
            "Reply only with: Connection OK",
            context,
            clean_key,
            chosen_model,
            [],
        )
        logger.info("[Chatbot] Provider=%s ConnectionTest=success KeySource=%s", selected_provider, key_source)
        return True, f"{selected_provider} connection successful ({key_source} key).", key_source
    except Exception as exc:
        error_type = _classify_error(exc)
        logger.warning(
            "[Chatbot] Provider=%s ConnectionTest=failed KeySource=%s ErrorType=%s",
            selected_provider,
            key_source,
            error_type,
        )
        if error_type == "invalid_key":
            return False, f"{selected_provider} key is invalid or unauthorized.", key_source
        if error_type == "quota":
            return False, f"{selected_provider} quota or rate limit issue detected.", key_source
        return False, f"{selected_provider} connection failed. Check network or provider status.", key_source
