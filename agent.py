"""
Kitchen Recipe Agent
=====================
"Your Smart Personal Kitchen Assistant"

A multi-page Streamlit app: Home dashboard, AI Chef Chat, My Ingredients,
Shopping List, Saved Recipes, and Theme Customize — all backed by a
Groq -> Cohere -> Gemini fallback chain, with chat history, ingredients,
shopping list, and saved recipes all persisted to chat_memory.json.

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
    ("Shopping List", "🛒"),
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
    "shopping_list": [],
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

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "👩‍🍳"
ACCENT = ACCENT_OPTIONS[st.session_state.accent]

# ---- Light, soft styling ----------------------------------------------------
st.markdown(
    f"""
    <style>
    :root {{
        --accent: {ACCENT};
        color-scheme: light;
    }}
    html, body, .stApp {{ color-scheme: light; }}

    /* Force light background on Streamlit's own header/toolbar chrome
       (only the part inside the app's own render area — the very top-right
       corner buttons like Share/GitHub are Streamlit Community Cloud's own
       platform chrome and are outside what app code can restyle). */
    header[data-testid="stHeader"] {{
        background: white !important;
    }}
    [data-testid="stToolbar"] {{ background: white !important; }}
    header[data-testid="stHeader"] {{ color: #2a2a2a !important; }}

    .stApp {{
        background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 4%, white) 0%, white 280px);
    }}

    section[data-testid="stSidebar"] {{
        background: color-mix(in srgb, var(--accent) 3%, white);
        border-right: 1px solid color-mix(in srgb, var(--accent) 12%, white);
    }}

    .stButton > button {{
        border-radius: 10px !important;
        border: 1px solid color-mix(in srgb, var(--accent) 30%, white) !important;
    }}
    .stButton > button[kind="primary"] {{
        background: color-mix(in srgb, var(--accent) 85%, white) !important;
        border-color: color-mix(in srgb, var(--accent) 85%, white) !important;
        color: white !important;
        font-weight: 600;
    }}
    .stButton > button[kind="secondary"] {{
        background: white !important;
        color: #2a2a2a !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }}

    input[type="checkbox"], input[type="radio"] {{ accent-color: var(--accent); }}
    a {{ color: var(--accent) !important; }}

    [data-testid="stChatMessage"] {{
        background: color-mix(in srgb, var(--accent) 4%, white);
        border-radius: 14px;
        border: 1px solid color-mix(in srgb, var(--accent) 10%, white);
    }}

    h1, h2, h3 {{ color: color-mix(in srgb, var(--accent) 45%, #2a2a2a) !important; }}

    .brand-card {{
        text-align: center; padding: 18px 10px; border-radius: 16px;
        background: color-mix(in srgb, var(--accent) 10%, white);
        border: 1px solid color-mix(in srgb, var(--accent) 25%, white);
        margin-bottom: 12px;
    }}
    .brand-title {{ font-size: 18px; font-weight: 800; margin-top: 4px; color: #2a2a2a; }}
    .brand-subtitle {{ font-size: 11.5px; opacity: 0.75; margin-top: 2px; color: #2a2a2a; }}

    .hero-banner {{
        padding: 24px 28px; border-radius: 18px;
        background: linear-gradient(135deg, color-mix(in srgb, var(--accent) 16%, white) 0%, color-mix(in srgb, var(--accent) 28%, white) 100%);
        margin-bottom: 18px; color: #2a2a2a !important;
        border: 1px solid color-mix(in srgb, var(--accent) 25%, white);
    }}
    .hero-greeting {{ font-size: 15px; color: #4a4a4a; margin: 0; font-weight: 500; }}
    .hero-title {{ font-size: 27px; font-weight: 800; color: #222 !important; margin: 4px 0 0 0; }}
    .hero-subtitle {{ font-size: 14px; color: #4a4a4a; margin-top: 6px; font-weight: 400; }}

    .status-pill {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: color-mix(in srgb, var(--accent) 18%, white);
        border: 1px solid color-mix(in srgb, var(--accent) 35%, white);
        font-size: 12px; font-weight: 700; color: #2a2a2a;
    }}
    .chip {{
        display: inline-block; padding: 6px 12px; border-radius: 999px;
        background: color-mix(in srgb, var(--accent) 10%, white);
        border: 1px solid color-mix(in srgb, var(--accent) 25%, white);
        font-size: 12.5px; margin: 3px 4px; font-weight: 500;
    }}
    .history-card {{
        padding: 10px 12px; border-radius: 10px;
        background: white; border: 1px solid color-mix(in srgb, var(--accent) 18%, white);
        margin-bottom: 8px; font-size: 12.5px;
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

    # Quick accent-color switcher — always visible, no need to open Theme Customize
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
        """Reject purely-numeric allergy/restriction entries — must be a
        text name (in any language/script), not a number."""
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
            help="Age in months — guidance changes significantly under vs over 6 months.",
        )
        raw_allergy = st.text_input(
            "⚠️ Known Allergies (if any):",
            value=st.session_state.user_allergies,
            placeholder="e.g. dairy, egg",
            help="Leave blank or type 'none' if not applicable. Text only — any language works.",
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
            help="Foods this pet should avoid, if known. Text only.",
        )
        st.session_state.user_allergies = _validate_allergy_text(raw_allergy, "user_allergies")

    elif current_type != "Select Profile / Default":
        st.session_state.user_age = st.number_input(
            "🎂 Profile Age (Years):",
            min_value=1,
            max_value=120,
            value=st.session_state.user_age if isinstance(st.session_state.user_age, int) and st.session_state.user_age >= 1 else 25,
            step=1,
            help="Integer age for custom tailoring.",
        )
        raw_allergy = st.text_input(
            "⚠️ Allergies / Restrictions:",
            value=st.session_state.user_allergies,
            placeholder="e.g. potato, dairy, peanuts",
            help="Dishes containing these will be strictly excluded — any language works, but must be text, not a number.",
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

    st.divider()
    st.caption("Backend order: Groq → Cohere → Gemini (auto fallback)")


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
            <p class="hero-subtitle">Talk in any language (English, Roman Urdu, Urdu script, etc.). Get clean table formats and allergen-safe dishes!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_col, btn_col = st.columns([4, 1])
    with input_col:
        home_input = st.text_input(
            "quick_prompt",
            placeholder="e.g., Hello! Mujh ko achi si dinner recipe batao (or in English)",
            label_visibility="collapsed",
        )
    with btn_col:
        find_clicked = st.button("Ask Assistant ✨", use_container_width=True, type="primary")

    if find_clicked and home_input.strip():
        ask_agent(home_input.strip())
        st.rerun()

    st.write("")
    st.subheader("💡 Quick Meal Suggestions")

    ideas = [
        ("🌅", "Quick breakfast ideas / Nashta"),
        ("☀️", "Healthy lunch options / Dopahar ka khana"),
        ("🌙", "Budget dinner recipes / Raat ka khana"),
        ("⚡", "30-minute fast dishes"),
    ]

    cols = st.columns(2)
    for idx, (emoji, label) in enumerate(ideas):
        with cols[idx % 2]:
            if st.button(f"{emoji}  {label}", use_container_width=True, key=f"idea_{idx}"):
                ask_agent(f"Please give me a table-formatted recipe for: {label} matching my language and dietary preferences.")
                st.rerun()

    if st.session_state.data["ingredients"]:
        st.write("")
        st.subheader("🥕 From your pantry ingredients")
        ing_list = st.session_state.data["ingredients"]
        chips_html = "".join(f'<span class="chip">{i["name"]}</span>' for i in ing_list)
        st.markdown(chips_html, unsafe_allow_html=True)
        if st.button("Generate recipes using my ingredients", type="primary"):
            summary_str = ", ".join(i["name"] for i in ing_list)
            ask_agent(f"I have these pantry items: {summary_str}. Give me table-formatted recipes avoiding my restrictions.")
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE: AI Chef Chat
# ---------------------------------------------------------------------------

def render_chat():
    st.markdown(
        f"""
        <div class="hero-banner" style="padding: 18px 24px;">
            <p class="hero-title" style="font-size:22px;">👩‍🍳 AI Chef Chat</p>
            <p class="hero-subtitle">Chat in any language you prefer. Responses match your exact language, table layout, and allergy rules.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, msg in enumerate(st.session_state.messages):
        avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if st.button("💾 Save this recipe", key=f"save_{i}"):
                    title = msg["content"].split("\n")[0][:60]
                    st.session_state.data["saved_recipes"].append(
                        {
                            "title": title or "Saved recipe",
                            "content": msg["content"],
                            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    save_data()
                    st.toast("Recipe saved successfully!")

    user_prompt = st.chat_input("Type in English, Roman Urdu, or Urdu script...")
    if user_prompt:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_prompt)
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("👩‍🍳 Kitchen Recipe Agent is thinking..."):
                ask_agent(user_prompt)
        st.rerun()


# ---------------------------------------------------------------------------
# PAGE: My Ingredients (string name only, no amounts)
# ---------------------------------------------------------------------------

def render_ingredients():
    st.subheader("🥕 My Ingredients & Pantry")
    st.caption("Add pantry items by name only (strings).")

    with st.form("add_ingredient", clear_on_submit=True):
        c1, c2 = st.columns([4, 1])
        name = c1.text_input("Ingredient Name (Letters only)", placeholder="e.g. Chicken, Tomato, Rice")
        submitted = c2.form_submit_button("➕ Add", use_container_width=True)

        if submitted:
            cleaned_name = name.strip()
            if not cleaned_name:
                st.error("Please enter a valid ingredient name.")
            elif any(char.isdigit() for char in cleaned_name):
                st.error("Ingredient name must be text (strings only) and cannot contain numbers or digits.")
            else:
                st.session_state.data["ingredients"].append({"name": cleaned_name})
                save_data()
                st.rerun()

    ingredients = st.session_state.data["ingredients"]
    if not ingredients:
        st.info("Your pantry is empty. Add items above!")
    else:
        st.write("")
        for idx, ing in enumerate(ingredients):
            col_info, col_del = st.columns([5, 1])
            with col_info:
                st.markdown(f"**{ing['name']}**")
            with col_del:
                if st.button("🗑️", key=f"del_ing_{idx}"):
                    ingredients.pop(idx)
                    save_data()
                    st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Shopping List (string name + integer/float amount + unit)
# ---------------------------------------------------------------------------

LIQUID_KEYWORDS = [
    "milk", "water", "oil", "juice", "yogurt", "yoghurt", "cream", "vinegar",
    "ghee", "syrup", "sauce", "broth", "stock", "wine", "honey", "doodh",
    "paani", "tel", "lassi", "shorba",
]


def guess_unit(item_name: str) -> str:
    """Suggest a sensible default unit based on the item name — liquids get
    litres, everything else defaults to kg."""
    name_lower = item_name.lower()
    if any(keyword in name_lower for keyword in LIQUID_KEYWORDS):
        return "litres"
    return "kg"


def _mark_unit_manual():
    st.session_state._shop_unit_manual = True


def render_shopping_list():
    st.subheader("🛒 Shopping List")
    st.caption("Manage grocery shopping items with item names, integer/float quantities, and units. Unit auto-suggests based on the item name — override anytime.")

    # Reset the add-item fields BEFORE creating the widgets this run, if the
    # previous run requested it (avoids Streamlit's "can't modify state after
    # widget instantiated" error from resetting them after the fact).
    if st.session_state.get("_reset_shop_form"):
        st.session_state.shop_item_input = ""
        st.session_state.shop_unit = "kg"
        st.session_state.shop_amount_input = 5.0
        st.session_state._shop_unit_manual = False
        st.session_state._reset_shop_form = False

    if "shop_unit" not in st.session_state:
        st.session_state.shop_unit = "kg"
    if "_shop_unit_manual" not in st.session_state:
        st.session_state._shop_unit_manual = False

    c1, c2, c3, c4 = st.columns([3, 1.5, 1.5, 1])
    item = c1.text_input(
        "Shopping Item (Letters only)",
        placeholder="e.g. Rice, Milk",
        key="shop_item_input",
    )

    # Recompute the suggestion in THIS same run (not a delayed callback) as
    # soon as the item text is available — unless the user has deliberately
    # picked a unit themselves, in which case we respect their choice.
    if not st.session_state._shop_unit_manual:
        st.session_state.shop_unit = guess_unit(item)

    amount = c2.number_input("Quantity (Int / Float)", min_value=0.1, step=1.0, format="%.2f", value=5.0, key="shop_amount_input")
    unit = c3.selectbox("Unit", UNIT_OPTIONS, key="shop_unit", on_change=_mark_unit_manual)
    add_clicked = c4.button("➕ Add Item", use_container_width=True)

    if add_clicked:
        cleaned_item = item.strip()
        if not cleaned_item:
            st.error("Please enter a valid shopping item name.")
        elif any(char.isdigit() for char in cleaned_item):
            st.error("Shopping item name must be text (strings only) and cannot contain numbers or digits.")
        else:
            st.session_state.data["shopping_list"].append(
                {"item": cleaned_item, "amount": amount, "unit": unit, "bought": False}
            )
            save_data()
            st.session_state._reset_shop_form = True
            st.rerun()

    shopping = st.session_state.data["shopping_list"]
    if not shopping:
        st.info("Your shopping list is empty.")
    else:
        st.write("")
        for idx, shop in enumerate(shopping):
            col_check, col_text, col_del = st.columns([1, 5, 1])
            with col_check:
                bought = st.checkbox("", value=shop["bought"], key=f"shop_check_{idx}")
                if bought != shop["bought"]:
                    shop["bought"] = bought
                    save_data()
            with col_text:
                style = "text-decoration: line-through; opacity: 0.6;" if shop["bought"] else ""
                st.markdown(
                    f"<span style='{style}'><b>{shop['item']}</b> — {shop['amount']:g} {shop['unit']}</span>",
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("🗑️", key=f"del_shop_{idx}"):
                    shopping.pop(idx)
                    save_data()
                    st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Saved Recipes
# ---------------------------------------------------------------------------

def render_saved_recipes():
    st.subheader("❤️ Saved Recipes")
    saved = st.session_state.data["saved_recipes"]
    if not saved:
        st.info("No saved recipes yet. Save recipes from the chat page!")
    else:
        for idx, rec in enumerate(reversed(saved)):
            real_idx = len(saved) - 1 - idx
            with st.expander(f"📖 {rec['title']} ({rec.get('timestamp', '')[:10]})"):
                st.markdown(rec["content"])
                if st.button("🗑️ Delete Recipe", key=f"del_rec_{real_idx}"):
                    saved.pop(real_idx)
                    save_data()
                    st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Theme Customize
# ---------------------------------------------------------------------------

def render_theme():
    st.subheader("🎨 Theme Customize")
    st.caption("Pick an accent color to instantly re-theme the entire application.")

    cols = st.columns(len(ACCENT_OPTIONS))
    for col, (name, hex_val) in zip(cols, ACCENT_OPTIONS.items()):
        with col:
            is_selected = st.session_state.accent == name
            label = f"✅ {name}" if is_selected else name
            if st.button(label, key=f"accent_{name}", use_container_width=True):
                st.session_state.accent = name
                st.rerun()
            st.markdown(
                f"""<div style="height:26px; border-radius:8px; background:{hex_val}; margin-top:4px;"></div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGES = {
    "Home": render_home,
    "AI Chef Chat": render_chat,
    "My Ingredients": render_ingredients,
    "Shopping List": render_shopping_list,
    "Saved Recipes": render_saved_recipes,
    "Theme Customize": render_theme,
}

PAGES[st.session_state.page]()
