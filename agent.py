"""
Kitchen Recipe AI
=================
A Streamlit chat agent that gives recipes and cooking help, formats
ingredients into clean tables, adapts output for "Normal" or
"Diabetic / Health-conscious" users, remembers chat history across
sessions (chat_memory.json), and falls back across three LLM
providers in this order: Groq -> Cohere -> Gemini.

Run with:
    streamlit run agent.py

Environment variables required (set as real env vars, or in a
.streamlit/secrets.toml file read via st.secrets):
    GROQ_API_KEY
    COHERE_API_KEY
    GEMINI_API_KEY
"""

import os
import json
import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

APP_TITLE = "Kitchen Recipe AI"
APP_ICON = "🍳"
MEMORY_FILE = Path(__file__).parent / "chat_memory.json"

GROQ_MODEL = "llama-3.3-70b-versatile"
COHERE_MODEL = "command-r-plus"
GEMINI_MODEL = "gemini-1.5-flash"


def get_key(name: str) -> str | None:
    """Read an API key from Streamlit secrets first, then env vars."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


GROQ_API_KEY = get_key("GROQ_API_KEY")
COHERE_API_KEY = get_key("COHERE_API_KEY")
GEMINI_API_KEY = get_key("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt(user_type: str) -> str:
    base = """You are "Kitchen Recipe AI", a friendly, precise cooking assistant.

CONVERSATION STYLE:
- If the user just greets you ("hi", "hello", "hey", "salam", etc.) or makes
  small talk, respond warmly and briefly in plain conversational text —
  introduce yourself in one short sentence and ask what they'd like to cook.
  Do NOT use the structured recipe format below for greetings or small talk.
- If the user asks a general question (e.g. "what can you do?", "how do you
  work?"), answer briefly and naturally in plain text, no table needed.
- Only switch into the full structured recipe format when the user actually
  asks for a recipe, a dish, or how to cook/make something specific.

RECIPE FORMAT (use ONLY when a recipe is actually requested):

1. A short (1-2 sentence) intro to the dish.
2. An **Ingredients** section formatted as a Markdown table with columns:
   | Ingredient | Quantity | Notes |
3. A **Steps** section as a numbered list, clear and concise.
4. A short **Tips** section (optional swaps, storage, etc).

Keep language simple and avoid unnecessary filler."""

    if user_type == "Diabetic / Health-conscious":
        extra = """

The user is DIABETIC / HEALTH-CONSCIOUS. Adapt every recipe accordingly:
- Add a 4th table column called "Health Note" flagging high-GI or high-sugar
  ingredients and suggesting a lower-sugar / lower-carb / lower-fat swap.
- Prefer whole grains, healthy fats, lean proteins, and non-starchy vegetables
  when suggesting substitutions, but do not force a substitution the user
  didn't ask for on the main ingredient list — just flag it.
- After the recipe, add a **Recommended Products & Swaps** section: name
  specific lower-sugar / lower-GI product alternatives (e.g. stevia instead
  of sugar, whole-wheat flour instead of refined flour, olive oil instead of
  butter) for the ingredients used.
- Add a **Healthier Cooking Method** section: suggest a better technique for
  this dish if relevant (e.g. baking/grilling/steaming instead of deep
  frying, air-frying, less oil, etc).
- Add a short **Nutrition Notes** section covering approximate carbs/sugar
  per serving (clearly labeled as an estimate, not medical advice) and a
  portion-size suggestion.
- Never give specific insulin dosing or medical advice; suggest the user
  confirm with their doctor or dietitian for individual dietary needs."""
        return base + extra

    extra = """

The user has NO special dietary restrictions. Keep the recipe practical for
everyday home cooking, but still call out easy healthier swaps as an
optional side note if relevant."""
    return base + extra


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def load_memory() -> list:
    if MEMORY_FILE.exists():
        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            return data.get("conversations", [])
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_memory(conversations: list) -> None:
    try:
        MEMORY_FILE.write_text(
            json.dumps({"conversations": conversations}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        st.warning(f"Could not save chat memory to disk: {e}")


def append_and_save(role: str, content: str, user_type: str) -> None:
    """Save a single message immediately (before AND after the AI call)."""
    entry = {
        "role": role,
        "content": content,
        "user_type": user_type,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.memory.append(entry)
    save_memory(st.session_state.memory)


def clear_memory() -> None:
    st.session_state.memory = []
    st.session_state.messages = []
    save_memory([])


# ---------------------------------------------------------------------------
# LLM backends (Groq -> Cohere -> Gemini fallback)
# ---------------------------------------------------------------------------

def call_groq(system_prompt: str, history: list, user_prompt: str) -> str:
    from groq import Groq

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": user_prompt})

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=1500,
    )
    return resp.choices[0].message.content


def call_cohere(system_prompt: str, history: list, user_prompt: str) -> str:
    import cohere

    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY not set")

    client = cohere.ClientV2(api_key=COHERE_API_KEY)
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_prompt})

    resp = client.chat(model=COHERE_MODEL, messages=messages)
    return resp.message.content[0].text


def call_gemini(system_prompt: str, history: list, user_prompt: str) -> str:
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system_prompt)

    # Gemini wants "user"/"model" roles
    gemini_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in history
    ]
    chat = model.start_chat(history=gemini_history)
    resp = chat.send_message(user_prompt)
    return resp.text


def get_ai_response(system_prompt: str, history: list, user_prompt: str) -> tuple[str, str]:
    """
    Backend priority: Groq -> Cohere -> Gemini.
    Explicit if / elif / else chain: each branch is only attempted if the
    previous one is unavailable or fails.
    """
    errors = []

    # 1) Try Groq first
    if GROQ_API_KEY:
        try:
            reply = call_groq(system_prompt, history, user_prompt)
            if reply and reply.strip():
                return reply, "Groq"
            errors.append("Groq: empty response")
        except Exception as e:
            errors.append(f"Groq: {e}")
    else:
        errors.append("Groq: GROQ_API_KEY not set")

    # 2) Groq unavailable/failed -> try Cohere
    if COHERE_API_KEY:
        try:
            reply = call_cohere(system_prompt, history, user_prompt)
            if reply and reply.strip():
                return reply, "Cohere"
            errors.append("Cohere: empty response")
        except Exception as e:
            errors.append(f"Cohere: {e}")
    else:
        errors.append("Cohere: COHERE_API_KEY not set")

    # 3) Groq and Cohere both unavailable/failed -> try Gemini
    if GEMINI_API_KEY:
        try:
            reply = call_gemini(system_prompt, history, user_prompt)
            if reply and reply.strip():
                return reply, "Gemini"
            errors.append("Gemini: empty response")
        except Exception as e:
            errors.append(f"Gemini: {e}")
    else:
        errors.append("Gemini: GEMINI_API_KEY not set")

    # 4) All three failed
    error_summary = "\n".join(errors)
    return (
        "⚠️ Sorry, none of the AI backends (Groq, Cohere, Gemini) could "
        "respond right now. Please check your API keys / network and try "
        f"again.\n\nDetails:\n```\n{error_summary}\n```",
        "None",
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="centered")

if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

if "messages" not in st.session_state:
    # Rebuild the visible chat window from saved memory (role/content only)
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.memory
    ]

if "user_type" not in st.session_state:
    st.session_state.user_type = "Normal"

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "🍳"

# ---- Global styling ---------------------------------------------------------
st.markdown(
    """
    <style>
    .sidebar-card {
        text-align: center;
        padding: 18px 8px 14px 8px;
        border-radius: 16px;
        background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);
        margin-bottom: 14px;
    }
    .sidebar-icon { font-size: 44px; line-height: 1; }
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: white;
        margin-top: 6px;
    }
    .sidebar-subtitle {
        font-size: 12.5px;
        color: rgba(255,255,255,0.9);
        margin-top: 4px;
    }
    .hero-banner {
        padding: 22px 20px;
        border-radius: 18px;
        background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 30px;
        font-weight: 800;
        color: white;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 14px;
        color: rgba(255,255,255,0.95);
        margin-top: 4px;
    }
    .history-item {
        padding: 8px 10px;
        border-radius: 10px;
        background: rgba(255,255,255,0.05);
        margin-bottom: 6px;
        font-size: 13px;
    }
    .history-time {
        font-size: 11px;
        opacity: 0.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-icon">{APP_ICON}</div>
            <div class="sidebar-title">{APP_TITLE}</div>
            <div class="sidebar-subtitle">Your AI sous-chef — recipes, ingredients, and cooking tips.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.user_type = st.radio(
        "🍽️ I'm cooking for:",
        options=["Normal", "Diabetic / Health-conscious"],
        index=["Normal", "Diabetic / Health-conscious"].index(st.session_state.user_type),
        help="Changes how ingredients and tips are tailored.",
    )

    st.divider()

    st.subheader("💬 Chat history")
    msg_count = len(st.session_state.messages)
    st.caption(f"{msg_count} message{'s' if msg_count != 1 else ''} in this session")

    user_msgs = [m for m in st.session_state.memory if m["role"] == "user"]
    if user_msgs:
        with st.expander(f"View recent questions ({len(user_msgs)})", expanded=False):
            for m in reversed(user_msgs[-15:]):
                time_label = m.get("timestamp", "")[11:16]
                preview = m["content"][:60] + ("…" if len(m["content"]) > 60 else "")
                st.markdown(
                    f"""<div class="history-item">
                    <span class="history-time">{time_label}</span><br>{preview}
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No questions asked yet — say hi below!")

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.confirm_clear = True

    if st.session_state.get("confirm_clear"):
        st.warning("This will permanently delete all saved chat history.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, delete", use_container_width=True, type="primary"):
                clear_memory()
                st.session_state.confirm_clear = False
                st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    st.divider()
    st.caption("Backend order: Groq → Cohere → Gemini (auto fallback)")

# ---- Main chat window -------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner">
        <p class="hero-title">{APP_ICON} {APP_TITLE}</p>
        <p class="hero-subtitle">Ask for any recipe — ingredients always come back in a clean table, tailored to you.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

user_prompt = st.chat_input("What would you like to cook today?")

if user_prompt:
    user_type = st.session_state.user_type

    # 1) Save + show the user's message BEFORE calling the AI
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    append_and_save("user", user_prompt, user_type)
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_prompt)

    # 2) Build context and call the AI (with fallback)
    system_prompt = build_system_prompt(user_type)
    history_for_model = st.session_state.messages[:-1][-10:]  # last 10 turns, excluding current

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Cooking up a response..."):
            reply, backend_used = get_ai_response(system_prompt, history_for_model, user_prompt)
        st.markdown(reply)
        if backend_used != "None":
            st.caption(f"Answered via {backend_used}")

    # 3) Save the assistant's reply AFTER the AI responds
    st.session_state.messages.append({"role": "assistant", "content": reply})
    append_and_save("assistant", reply, user_type)

    # 4) Rerun so the sidebar (message count + history list) reflects this exchange immediately
    st.rerun()
