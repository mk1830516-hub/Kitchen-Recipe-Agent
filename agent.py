"""
Kitchen Recipe Agent
=====================
"Your Smart Personal Kitchen Assistant"

A multi-page Streamlit app: Home dashboard, AI Chef Chat, My Ingredients,
Saved Recipes, and Theme Customize — all backed by a
Groq -> Cohere -> Gemini fallback chain, with chat history, ingredients,
and saved recipes all persisted to chat_memory.json.

Run with:
    streamlit run agent.py

Keys (set via st.secrets or env vars):
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

APP_TITLE = "Kitchen Recipe Agent"
APP_TAGLINE = "Your Smart Personal Kitchen Assistant"
APP_ICON = "👩‍🍳"
MEMORY_FILE = Path(__file__).parent / "chat_memory.json"

GROQ_MODEL = "openai/gpt-oss-120b"
COHERE_MODEL = "command-a-03-2025"
GEMINI_MODEL = "gemini-2.5-flash"

ACCENT_OPTIONS = {
    "Green": "#2f6f4f",
    "Cyan": "#0e7c86",
    "Blue": "#2563eb",
    "Purple": "#7c3aed",
    "Pink": "#db2777",
    "Orange": "#ea580c",
    "Yellow": "#ca8a04",
}

NAV_ITEMS = [
    ("Home", "🏠"),
    ("AI Chef Chat", "💬"),
    ("My Ingredients", "🥕"),
    ("Saved Recipes", "❤️"),
    ("Theme Customize", "🎨"),
]

UNIT_OPTIONS = ["kg", "g", "litres", "ml", "pieces", "cups", "tbsp", "tsp", "pinch"]


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
# System prompt — dynamic language matching, strict profile/allergy rules,
# kitchen-only domain, budget-friendly Markdown-table formatting
# ---------------------------------------------------------------------------

USER_TYPES = [
    "Select Profile / Default",
    "Adult",
    "Child",
    "Infant / Baby (Under 1 year)",
    "Health-conscious / Fitness",
    "Patient (Medical / Dietary Condition)",
    "Pet (Any kind)",
]


def build_system_prompt(user_type: str, user_age, user_allergies: str) -> str:
    is_default = (user_type == "Select Profile / Default")

    if is_default:
        profile_clause = (
            "No specific profile or age has been selected yet. Provide "
            "general, balanced recipes suitable for a typical adult."
        )
    elif user_type == "Child":
        profile_clause = f"""Profile: Child, Age: {user_age} years old.
- Prefer milder spice levels, softer textures, and kid-friendly presentation.
- Flag common choking hazards for young children (whole grapes, nuts,
  popcorn, hard raw vegetables) and suggest safer prep (chopped, mashed)
  when the age suggests this matters.
- Never suggest alcohol, caffeine-heavy, or age-inappropriate ingredients."""
    elif user_type == "Infant / Baby (Under 1 year)":
        profile_clause = f"""Profile: Infant / Baby, Age: {user_age} months old.
- If age is under 6 months: the baby should be exclusively breastfed or
  formula-fed. Do NOT suggest any solid food, recipe, water, juice, or
  other liquid/food item. Clearly explain this, and recommend the parent
  consult their pediatrician for any feeding questions.
- If age is 6-12 months: only general, widely-accepted starter-food
  guidance is appropriate (e.g. single-ingredient soft purees, mashed
  soft fruit/vegetable, iron-fortified cereal) introduced one at a time.
  Avoid honey, whole nuts, choking hazards, added salt/sugar, and cow's
  milk as a main drink. Always add a note to introduce new foods one at a
  time and watch for allergic reactions, and to consult a pediatrician
  for individualized guidance — this is general information, not medical
  advice."""
    elif user_type == "Health-conscious / Fitness":
        profile_clause = f"""Profile: Health-conscious / Fitness, Age: {user_age} years old.
- Favor higher-protein, lower-processed-sugar, nutrient-dense options.
- Where relevant, note an approximate calorie/protein estimate per serving
  (clearly labeled as an estimate, not medical advice).
- Suggest a lighter cooking method (bake/grill/steam/air-fry) over deep-frying
  when it fits the dish."""
    elif user_type == "Patient (Medical / Dietary Condition)":
        profile_clause = f"""Profile: Patient with a medical/dietary condition, Age: {user_age} years old.
- The user or someone they're cooking for has a medical condition
  affecting diet (e.g. diabetes, hypertension, kidney condition, post-
  surgery recovery). Any condition-specific detail they give (in the
  allergy/restriction field or in chat) should shape ingredient choices
  — e.g. lower sodium, lower sugar/GI, softer textures, easier digestion.
- Give general food information only — NEVER diagnose, NEVER claim a food
  cures or treats the condition, and NEVER give medication or dosing
  advice.
- Always add a brief note recommending they confirm any significant
  dietary change with their doctor or a registered dietitian, especially
  before making it a regular part of their diet."""
    elif user_type == "Pet (Any kind)":
        pet_label = user_age if isinstance(user_age, str) and user_age.strip() else "pet"
        profile_clause = f"""Profile: Household Pet — {pet_label}. The user wants
food/treat guidance for their {pet_label.lower()}, not themselves.
- Only suggest foods that are generally recognized as safe for a {pet_label.lower()}.
  Be honest if you're not confident a food is safe for this specific
  species — say so rather than guessing, and suggest they check with a vet.
- ALWAYS explicitly warn against and never include known toxic foods for
  common pets: chocolate, grapes/raisins, onion, garlic, chives, xylitol
  (artificial sweetener), macadamia nuts, alcohol, caffeine, raw bread
  dough, and cooked bones (choking/splintering risk) — and flag any
  species-specific toxic items you're aware of (e.g. avocado can be risky
  for some birds).
- Keep portions simple/plain (no heavy seasoning, salt, or sauces).
- Add a note recommending the user confirm with a veterinarian before
  introducing a new food, especially for young animals or pets with
  health conditions."""
    else:  # Adult
        profile_clause = f"Profile: Adult, Age: {user_age} years old. Standard, practical everyday recipes."

    if user_allergies and user_allergies.strip().lower() not in ("none", ""):
        allergy_clause = (
            f"- STRICT ALLERGY / RESTRICTION EXCLUSION: The user must strictly avoid: "
            f"{user_allergies}. This may be written in any language or script (English, "
            f"Roman Urdu, Urdu script, etc.) — understand it regardless of language. "
            f"NEVER suggest any dish, recipe, or ingredient containing these — not even "
            f"as a minor component. If a classic version of a requested dish normally "
            f"contains a restricted item, adapt the recipe to remove or substitute it "
            f"and say so clearly."
        )
    else:
        allergy_clause = "- No allergies or restricted ingredients have been specified."

    return f"""You are "{APP_TITLE}", a smart, professional personal AI chef and kitchen assistant.
Tagline: "{APP_TAGLINE}"

DYNAMIC LANGUAGE MATCHING (very important):
Always reply in the EXACT same language and script the user just used in
their message — if they write in Roman Urdu, reply in natural Roman Urdu;
if in Urdu script, reply in Urdu script; if in English, reply in English;
if in any other language, reply in that language. Never switch language on
your own. This applies to every part of your reply, including any refusal
or clarifying question.

STRICT DOMAIN — kitchen, food, and grocery only:
You ONLY help with: recipes, cooking, food, ingredients, dishes, meal
planning, kitchen techniques, food substitutions, cooking equipment
related to food prep, grocery/shopping-list help, and pet food safety
when a pet profile is selected. If the user asks about anything unrelated
(coding, homework, politics, news, general trivia, personal tasks
unrelated to food), politely decline and redirect back to cooking —
briefly, in the user's own language/script — then ask what they'd like to
cook or shop for.

PROFILE: {profile_clause}

{allergy_clause}

CONVERSATION STYLE:
- If the user just greets you or makes small talk, respond warmly and
  briefly, then ask what they'd like to cook — no table needed for that.
- MATCH THE USER'S TONE AND FORMALITY, not just their language — this is
  important for feeling like a real, natural agent rather than a scripted
  bot. If the user casually says "hi", "hello", "hey", or "salam", reply
  with an equally casual, short greeting back in the same language/script
  (e.g. just "Hi!" or "Hey!" or a casual "Salam!") — do NOT default to a
  formal greeting like "Assalam-o-Alaikum" unless the user themselves used
  that specific formal greeting first. Keep the very first reply short and
  natural, like a real person would casually respond.
- If the user gives ingredients, analyze them and suggest 1-3 suitable
  dishes: name the best match first with a brief reason, then other
  options with approximate time, difficulty, and what's missing if any.
- If the user just names a dish without detail, ask only the one
  necessary clarifying question, then proceed once answered.
- If missing an ingredient, suggest 2-3 practical substitutes and briefly
  note how each changes taste/texture, then recommend one.
- If the user asks for grocery/shopping help, suggest items with sensible
  quantities and units, and offer to note them as a shopping list.
- Remember the current conversation's context so the user doesn't have to
  repeat themselves.
- Never invent a URL, video link, or source. Only share a link if you are
  certain it is real and the user asked for one; otherwise say honestly
  that you don't have a verified link to share right now.

FORMATTING & BUDGET:
- Present ingredients and quantities using a clean Markdown table:
  | Ingredient | Quantity | Notes |, followed by numbered Steps and a
  short Tips section, for any full recipe.
- Use a short Markdown table when comparing multiple dishes, options, or
  building a meal plan. Don't use a table for a simple one-line answer.
- When the user gives a budget, tailor ingredient choices and portions to
  fit it, and mention the approximate cost angle briefly.
- Keep every suggestion practical and budget-friendly by default — prefer
  common, affordable ingredients and mention a cheaper swap when a
  natural one exists.
- When helpful, offer a couple of alternative choices, not just one
  answer, plus a short Tips line."""


# ---------------------------------------------------------------------------
# Data persistence
# ---------------------------------------------------------------------------

DEFAULT_DATA = {
    "conversations": [],
    "ingredients": [],
    "saved_recipes": [],
}


def load_data() -> dict:
    if MEMORY_FILE.exists():
        try:
            raw = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            return {**DEFAULT_DATA, **raw}
        except (json.JSONDecodeError, OSError):
            return dict(DEFAULT_DATA)
    return dict(DEFAULT_DATA)


def save_data() -> None:
    try:
        MEMORY_FILE.write_text(
            json.dumps(st.session_state.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        st.warning(f"Could not save data to disk: {e}")


def append_and_save(role: str, content: str, user_type: str) -> None:
    entry = {
        "role": role,
        "content": content,
        "user_type": user_type,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.data["conversations"].append(entry)
    save_data()


def clear_chat_history() -> None:
    st.session_state.data["conversations"] = []
    st.session_state.messages = []
    save_data()


# ---------------------------------------------------------------------------
# LLM backends (Groq -> Cohere -> Gemini fallback)
# ---------------------------------------------------------------------------

def call_groq(system_prompt: str, history: list, user_prompt: str) -> str:
    from groq import Groq

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    client = Groq(api_key=GROQ_API_KEY)
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": user_prompt}
    ]
    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.4, max_tokens=1500
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
    gemini_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in history
    ]
    chat = model.start_chat(history=gemini_history)
    resp = chat.send_message(user_prompt)
    return resp.text


def get_ai_response(system_prompt: str, history: list, user_prompt: str) -> tuple[str, str]:
    """Backend priority: Groq -> Cohere -> Gemini, explicit if/elif/else."""
    errors = []

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

    error_summary = "\n".join(errors)
    return (
        "⚠️ Sorry, none of the AI backends (Groq, Cohere, Gemini) could "
        "respond right now. Please check your API keys / network and try "
        f"again.\n\nDetails:\n```\n{error_summary}\n```",
        "None",
    )


def ask_agent(user_prompt: str) -> None:
    user_type = st.session_state.user_type
    user_age = st.session_state.user_age
    user_allergies = st.session_state.user_allergies

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    append_and_save("user", user_prompt, user_type)

    system_prompt = build_system_prompt(user_type, user_age, user_allergies)
    history_for_model = st.session_state.messages[:-1][-10:]
    reply, backend_used = get_ai_response(system_prompt, history_for_model, user_prompt)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    append_and_save("assistant", reply, user_type)
    st.session_state.last_backend = backend_used
    st.session_state.page = "AI Chef Chat"


# ---------------------------------------------------------------------------
# Streamlit setup + session state
# ---------------------------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

if "data" not in st.session_state:
    st.session_state.data = load_data()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.data["conversations"]
    ]

if "user_type" not in st.session_state:
    st.session_state.user_type = "Select Profile / Default"

if "user_age" not in st.session_state:
    st.session_state.user_age = 25

if "user_allergies" not in st.session_state:
    st.session_state.user_allergies = "None"

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "accent" not in st.session_state:
    st.session_state.accent = "Green"

if "editing_ing_idx" not in st.session_state:
    st.session_state.editing_ing_idx = None

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "👩‍🍳"
ACCENT = ACCENT_OPTIONS[st.session_state.accent]

# ---- Fixed Clean Styling with Visible Chat Text ------
st.markdown(
    f"""
    <style>
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    :root {{
        --accent: {ACCENT};
        color-scheme: light;
    }}

    .stApp {{
        background: #ffffff;
        color: #111827 !important;
    }}

    section[data-testid="stSidebar"] {{
        background: #f8f9fa;
        border-right: 1px solid #e5e7eb;
    }}

    .stButton > button {{
        border-radius: 10px !important;
        border: 1px solid #d1d5db !important;
        background: #ffffff !important;
        color: #111827 !important;
    }}
    
    .stButton > button[kind="primary"] {{
        background: {ACCENT} !important;
        border-color: {ACCENT} !important;
        color: white !important;
        font-weight: 600;
    }}

    [data-testid="stChatMessage"] {{
        background: #f3f4f6 !important;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        color: #111827 !important;
    }}

    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span, [data-testid="stChatMessage"] div {{
        color: #111827 !important;
    }}

    h1, h2, h3, h4, h5, h6 {{ color: #111827 !important; }}

    .brand-card {{
        text-align: center; padding: 18px 10px; border-radius: 16px;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        margin-bottom: 12px;
    }}
    .brand-title {{ font-size: 18px; font-weight: 800; margin-top: 4px; color: #111827; }}
    .brand-subtitle {{ font-size: 11.5px; opacity: 0.75; margin-top: 2px; color: #4b5563; }}

    .hero-banner {{
        padding: 24px 28px; border-radius: 18px;
        background: #f3f4f6;
        margin-bottom: 18px; color: #111827 !important;
        border: 1px solid #e5e7eb;
    }}
    .hero-greeting {{ font-size: 15px; color: #4b5563; margin: 0; font-weight: 500; }}
    .hero-title {{ font-size: 27px; font-weight: 800; color: #111827 !important; margin: 4px 0 0 0; }}
    .hero-subtitle {{ font-size: 14px; color: #4b5563; margin-top: 6px; font-weight: 400; }}

    .status-pill {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: #e5e7eb;
        border: 1px solid #d1d5db;
        font-size: 12px; font-weight: 700; color: #111827;
    }}
    .history-card {{
        padding: 10px 12px; border-radius: 10px;
        background: white; border: 1px solid #e5e7eb;
        margin-bottom: 8px; font-size: 12.5px; color: #111827;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — Branding, Navigation, Profile, Age, Allergies, quick theme & history
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand-card">
            <div style="font-size: 36px;">{APP_ICON}</div>
            <div class="brand-title">{APP_TITLE}</div>
            <div class="brand-subtitle">{APP_TAGLINE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for label, icon in NAV_ITEMS:
        is_active = st.session_state.page == label
        if st.button(
            f"{icon}  {label}",
            key=f"nav_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = label
            st.rerun()

    st.divider()

    st.caption("🎨 Quick theme")
    ACCENT_EMOJI = {
        "Green": "🟢", "Cyan": "🔵", "Blue": "🔷", "Purple": "🟣",
        "Pink": "🌸", "Orange": "🟠", "Yellow": "🟡",
    }
    swatch_cols = st.columns(len(ACCENT_OPTIONS))
    for col, name in zip(swatch_cols, ACCENT_OPTIONS.keys()):
        with col:
            label = ACCENT_EMOJI.get(name, "⚪")
            if st.session_state.accent == name:
                label += "✓"
            if st.button(label, key=f"quick_accent_{name}", help=name, use_container_width=True):
                st.session_state.accent = name
                st.rerun()

    st.divider()

    st.session_state.user_type = st.selectbox(
        "🍽️ Kitchen Profile:",
        options=USER_TYPES,
        index=USER_TYPES.index(st.session_state.user_type),
        help="Select a profile or leave as default.",
    )

    current_type = st.session_state.user_type

    def _validate_allergy_text(value: str, field_key: str):
        cleaned = value.strip()
        if cleaned and cleaned.lower() != "none" and cleaned.replace(" ", "").isdigit():
            st.error("Allergy/restriction must be a text name (e.g. peanuts, dairy) — not just a number.")
            st.session_state[field_key] = "None"
            return "None"
        return value

    if current_type == "Infant / Baby (Under 1 year)":
        st.session_state.user_age = st.number_input(
            "🍼 Baby's Age (Months):",
            min_value=0,
            max_value=12,
            value=min(st.session_state.user_age, 12) if isinstance(st.session_state.user_age, int) else 4,
            step=1,
        )
        raw_allergy = st.text_input(
            "⚠️ Known Allergies (if any):",
            value=st.session_state.user_allergies,
            placeholder="e.g. dairy, egg",
        )
        st.session_state.user_allergies = _validate_allergy_text(raw_allergy, "user_allergies")

    elif current_type == "Pet (Any kind)":
        st.session_state.user_age = st.text_input(
            "🐾 Pet type (e.g. Dog, Cat, Bird, Rabbit):",
            value=st.session_state.get("pet_type", "Dog"),
            placeholder="e.g. Dog, Cat, Bird, Rabbit",
        )
        st.session_state.pet_type = st.session_state.user_age
        raw_allergy = st.text_input(
            "⚠️ Known Food Sensitivities:",
            value=st.session_state.user_allergies,
            placeholder="e.g. chicken, grain-sensitive",
        )
        st.session_state.user_allergies = _validate_allergy_text(raw_allergy, "user_allergies")

    elif current_type != "Select Profile / Default":
        st.session_state.user_age = st.number_input(
            "🎂 Profile Age (Years):",
            min_value=1,
            max_value=120,
            value=st.session_state.user_age if isinstance(st.session_state.user_age, int) and st.session_state.user_age >= 1 else 25,
            step=1,
        )
        raw_allergy = st.text_input(
            "⚠️ Allergies / Restrictions:",
            value=st.session_state.user_allergies,
            placeholder="e.g. potato, dairy, peanuts",
        )
        st.session_state.user_allergies = _validate_allergy_text(raw_allergy, "user_allergies")

    st.divider()
    st.subheader("💬 Chat History")

    chats = st.session_state.data["conversations"]
    if not chats:
        st.caption("No past conversations yet.")
    else:
        if st.button("🗑️ Clear All History", use_container_width=True):
            clear_chat_history()
            st.rerun()

        pairs = []
        i = 0
        while i < len(chats):
            if chats[i]["role"] == "user":
                u_turn = chats[i]
                a_turn = chats[i + 1] if (i + 1 < len(chats) and chats[i + 1]["role"] == "assistant") else None
                pairs.append((u_turn, a_turn))
                i += 2 if a_turn else 1
            else:
                i += 1

        for idx, (u_turn, a_turn) in enumerate(reversed(pairs[-8:])):
            snippet = u_turn["content"][:32] + ("..." if len(u_turn["content"]) > 32 else "")
            col_txt, col_del = st.columns([4, 1])
            with col_txt:
                st.markdown(
                    f"<div class='history-card'><b>{u_turn.get('user_type', 'Default')}</b><br>{snippet}</div>",
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("❌", key=f"del_pair_{idx}", help="Delete conversation turn"):
                    if u_turn in st.session_state.data["conversations"]:
                        st.session_state.data["conversations"].remove(u_turn)
                    if a_turn and a_turn in st.session_state.data["conversations"]:
                        st.session_state.data["conversations"].remove(a_turn)
                    save_data()
                    st.session_state.messages = [
                        {"role": m["role"], "content": m["content"]} for m in st.session_state.data["conversations"]
                    ]
                    st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Home
# ---------------------------------------------------------------------------

def render_home():
    current_profile = st.session_state.user_type
    if current_profile == "Select Profile / Default":
        profile_badge = "Mode: Default (No Profile Selected)"
    elif current_profile == "Infant / Baby (Under 1 year)":
        profile_badge = (
            f"Profile: {current_profile} ({st.session_state.user_age} months old) "
            f"| Avoiding: {st.session_state.user_allergies}"
        )
    elif current_profile == "Pet (Any kind)":
        profile_badge = (
            f"Profile: Pet — {st.session_state.user_age} "
            f"| Avoiding: {st.session_state.user_allergies}"
        )
    else:
        profile_badge = (
            f"Profile: {current_profile} (Age: {st.session_state.user_age}) "
            f"| Avoiding: {st.session_state.user_allergies}"
        )

    st.markdown(
        f"""
        <div class="hero-banner">
            <span class="status-pill">✨ {profile_badge}</span>
            <p class="hero-greeting" style="margin-top:10px;">👋 Welcome to your Smart Kitchen Dashboard</p>
            <p class="hero-title">What would you like to eat today?</p>
            <p class="hero-subtitle">Talk in any language, manage ingredients, or chat with your AI Chef.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💬 Open AI Chef Chat", use_container_width=True, type="primary"):
            st.session_state.page = "AI Chef Chat"
            st.rerun()
    with col2:
        if st.button("🥕 Manage Ingredients", use_container_width=True):
            st.session_state.page = "My Ingredients"
            st.rerun()
    with col3:
        if st.button("❤️ View Saved Recipes", use_container_width=True):
            st.session_state.page = "Saved Recipes"
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE: AI Chef Chat
# ---------------------------------------------------------------------------

def render_chat():
    st.title("💬 AI Chef Chat")
    st.caption("Ask for recipes, ingredient substitutes, or meal plans in any language.")

    for m in st.session_state.messages:
        avatar = USER_AVATAR if m["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    if user_query := st.chat_input("Ask your AI chef anything about food or cooking..."):
        ask_agent(user_query)
        st.rerun()


# ---------------------------------------------------------------------------
# PAGE: My Ingredients
# ---------------------------------------------------------------------------

def render_ingredients():
    st.title("🥕 My Ingredients & Pantry")
    st.caption("Add items with an optional quantity and unit, edit them anytime, or delete them.")

    with st.form("add_ingredient_form", clear_on_submit=True):
        col_name, col_qty, col_unit = st.columns([2, 1, 1])
        with col_name:
            new_name = st.text_input("Ingredient Name", placeholder="e.g. Tomatoes")
        with col_qty:
            new_qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=0.5)
        with col_unit:
            new_unit = st.selectbox("Unit", options=UNIT_OPTIONS)

        submitted = st.form_submit_button("➕ Add Ingredient", type="primary")
        if submitted:
            if new_name.strip():
                st.session_state.data["ingredients"].append({
                    "name": new_name.strip(),
                    "qty": new_qty,
                    "unit": new_unit,
                })
                save_data()
                st.success(f"Added {new_name.strip()}!")
                st.rerun()
            else:
                st.error("Please enter a valid ingredient name.")

    st.divider()
    ingredients = st.session_state.data["ingredients"]

    if not ingredients:
        st.info("Your pantry is currently empty. Add some ingredients above!")
    else:
        for idx, ing in enumerate(ingredients):
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(f"**{ing['name']}** — {ing['qty']} {ing['unit']}")
            with c2:
                if st.button("✏️ Edit", key=f"edit_ing_{idx}"):
                    st.session_state.editing_ing_idx = idx
                    st.rerun()
            with c3:
                if st.button("🗑️ Delete", key=f"del_ing_{idx}"):
                    st.session_state.data["ingredients"].pop(idx)
                    save_data()
                    st.rerun()

    if st.session_state.editing_ing_idx is not None:
        idx = st.session_state.editing_ing_idx
        if idx < len(ingredients):
            st.divider()
            st.subheader(f"Editing: {ingredients[idx]['name']}")
            with st.form("edit_form"):
                e_name = st.text_input("Name", value=ingredients[idx]["name"])
                e_qty = st.number_input("Qty", min_value=0.0, value=float(ingredients[idx]["qty"]), step=0.5)
                e_unit = st.selectbox("Unit", options=UNIT_OPTIONS, index=UNIT_OPTIONS.index(ingredients[idx]["unit"]) if ingredients[idx]["unit"] in UNIT_OPTIONS else 0)
                
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.form_submit_button("💾 Save Changes", type="primary"):
                        st.session_state.data["ingredients"][idx] = {"name": e_name, "qty": e_qty, "unit": e_unit}
                        save_data()
                        st.session_state.editing_ing_idx = None
                        st.rerun()
                with col_cancel:
                    if st.form_submit_button("❌ Cancel"):
                        st.session_state.editing_ing_idx = None
                        st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Saved Recipes
# ---------------------------------------------------------------------------

def render_saved():
    st.title("❤️ Saved Recipes")
    st.caption("Your favorite recipes bookmarked for quick access.")

    saved = st.session_state.data["saved_recipes"]
    if not saved:
        st.info("No saved recipes yet. Ask the AI Chef Chat for a recipe and save it!")
    else:
        for idx, rec in enumerate(saved):
            with st.expander(f"📖 {rec.get('title', 'Recipe')}"):
                st.markdown(rec.get("content", ""))
                if st.button("🗑️ Remove", key=f"del_rec_{idx}"):
                    st.session_state.data["saved_recipes"].pop(idx)
                    save_data()
                    st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Theme Customize
# ---------------------------------------------------------------------------

def render_theme():
    st.title("🎨 Theme Customization")
    st.caption("Choose your preferred primary accent color for the application.")

    selected = st.radio("Select Accent Color:", options=list(ACCENT_OPTIONS.keys()), index=list(ACCENT_OPTIONS.keys()).index(st.session_state.accent))
    if selected != st.session_state.accent:
        st.session_state.accent = selected
        st.rerun()


# ---------------------------------------------------------------------------
# Main Router
# ---------------------------------------------------------------------------

if st.session_state.page == "Home":
    render_home()
elif st.session_state.page == "AI Chef Chat":
    render_chat()
elif st.session_state.page == "My Ingredients":
    render_ingredients()
elif st.session_state.page == "Saved Recipes":
    render_saved()
elif st.session_state.page == "Theme Customize":
    render_theme()
