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

NOTE ON SCOPE: Photo upload, voice input, and real-time YouTube video
lookup are shown as UI affordances but are not wired to real services in
this build (no image-recognition / speech-to-text / video-search API is
configured) — rather than fake them, they show a short "not available in
this demo" message. No food photography, video links, or ratings are
fabricated; recipe content comes only from the live AI response.
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

GROQ_MODEL = "llama-3.3-70b-versatile"
COHERE_MODEL = "command-r-plus"
GEMINI_MODEL = "gemini-1.5-flash"

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

UNIT_OPTIONS = ["g", "kg", "ml", "l", "pieces", "cups", "tbsp", "tsp", "pinch"]


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
    base = f"""You are "{APP_TITLE}", a friendly, smart, professional AI chef.
Tagline: "{APP_TAGLINE}"

STRICT DOMAIN:
You ONLY help with: recipes, cooking, food, ingredients, dishes, meal
planning, kitchen techniques, food substitutions, and cooking equipment
related to food prep. If the user asks about anything unrelated (coding,
homework, politics, news, general trivia, personal tasks unrelated to
food), reply with EXACTLY this and nothing else:
"Sorry! 👩‍🍳 I'm {APP_TITLE}, your Smart Kitchen & Food Assistant. I can only help with recipes, cooking, ingredients, dishes, meal planning, substitutions, and kitchen-related questions. What would you like to cook today?"

CONVERSATION STYLE:
- If the user just greets you or makes small talk, respond warmly and
  briefly in plain text, then ask what they'd like to cook. No table.
- If the user gives ingredients (e.g. "I have chicken, potato, onion and
  yogurt"), analyze them and suggest 1-3 suitable dishes: name the best
  match first and explain briefly why, then list other options with
  approximate time, difficulty, and what's missing if anything.
- If the user just names a dish (e.g. "biryani") without detail, ask only
  the one necessary clarifying question (e.g. which protein/variant),
  then proceed once answered.
- If the user is missing an ingredient, suggest 2-3 practical substitutes
  and briefly note how each changes taste/texture, then recommend one.
- Remember the current conversation's context (the dish/ingredients being
  discussed) so the user doesn't have to repeat themselves.
- For a person-specific ask (children, older adults, family, dietary
  restriction, allergy, or a medical/patient context), adapt guidance
  appropriately. For children ask age only if it matters. For medical or
  patient contexts, give general food information only — never diagnose
  or claim a food cures/treats a condition — and recommend the user
  confirm with a doctor or registered dietitian for anything medically
  important (allergies, medications, prescribed diets).
- Never invent a URL, video link, or source. Only mention a link if you
  are certain it is real and the user asked for one; otherwise say you
  don't have a verified link to share.

SMART RESPONSE FORMAT — choose automatically, don't force the same shape
every time:
- Simple question -> short plain-text answer.
- Full recipe -> Ingredients (as a Markdown table: | Ingredient | Quantity | Notes |), then numbered Steps, then a short Tips section.
- Multiple recipe options -> a short list or small Markdown table comparing them (dish / time / difficulty / best for).
- Meal plan -> a structured Markdown table.
- Missing ingredient -> a short substitutes list.
- Ambiguous/missing info -> one short clarifying question, nothing else.
Don't use a table when a sentence or two would answer it just as well."""

    if user_type == "Diabetic / Health-conscious":
        extra = """

The user is DIABETIC / HEALTH-CONSCIOUS. Adapt every recipe accordingly:
- Add a 4th table column "Health Note" flagging high-GI/high-sugar items
  with a lower-sugar / lower-carb / lower-fat swap suggestion.
- After the recipe, add "Recommended Products & Swaps" (specific
  lower-sugar/lower-GI alternatives) and "Healthier Cooking Method"
  (e.g. bake/grill/steam/air-fry instead of deep frying).
- Add short "Nutrition Notes": approximate carbs/sugar per serving
  (clearly an estimate, not medical advice) and a portion suggestion.
- Never give insulin dosing or medical advice; recommend confirming with
  a doctor/dietitian for individual needs."""
        return base + extra

    extra = """

The user has no special dietary restrictions. Keep recipes practical for
everyday home cooking, with an occasional easy healthier swap noted."""
    return base + extra


# ---------------------------------------------------------------------------
# Data persistence (chat history + ingredients + shopping list + saved recipes)
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

    gemini_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in history
    ]
    chat = model.start_chat(history=gemini_history)
    resp = chat.send_message(user_prompt)
    return resp.text


def analyze_ingredient_photo(image_bytes: bytes, mime_type: str) -> str:
    """
    Use Gemini's vision capability to identify ingredients in an uploaded
    photo. Returns a short comma-separated ingredient list, or raises if
    Gemini isn't configured / the call fails — callers should fall back to
    asking the user to type instead rather than fabricating a result.
    """
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set — photo analysis needs Gemini.")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(
        [
            "List only the food ingredients visible in this photo, as a short "
            "comma-separated list (e.g. 'chicken, potato, onion, yogurt'). "
            "If you can't clearly identify any food ingredients, reply exactly: "
            "NONE_DETECTED",
            {"mime_type": mime_type, "data": image_bytes},
        ]
    )
    return resp.text.strip()


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
    """Shared helper: send a prompt to the AI, save it, append to chat, rerun."""
    user_type = st.session_state.user_type
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    append_and_save("user", user_prompt, user_type)

    system_prompt = build_system_prompt(user_type)
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
    st.session_state.user_type = "Normal"

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "accent" not in st.session_state:
    st.session_state.accent = "Green"

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "👩‍🍳"
ACCENT = ACCENT_OPTIONS[st.session_state.accent]

# ---- Global styling ---------------------------------------------------------
st.markdown(
    f"""
    <style>
    :root {{ --accent: {ACCENT}; }}

    /* Whole-app background tint */
    .stApp {{
        background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 5%, white) 0%, white 320px);
    }}

    /* Sidebar tint */
    section[data-testid="stSidebar"] {{
        background: color-mix(in srgb, var(--accent) 4%, white);
        border-right: 1px solid color-mix(in srgb, var(--accent) 15%, white);
    }}

    /* Buttons themed by accent */
    .stButton > button {{
        border-radius: 10px !important;
        border: 1px solid color-mix(in srgb, var(--accent) 35%, white) !important;
    }}
    .stButton > button[kind="primary"] {{
        background: var(--accent) !important;
        border-color: var(--accent) !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }}

    /* Chat bubbles themed by accent */
    [data-testid="stChatMessage"] {{
        background: color-mix(in srgb, var(--accent) 5%, white);
        border-radius: 14px;
        border: 1px solid color-mix(in srgb, var(--accent) 12%, white);
    }}

    /* Inputs themed by accent on focus */
    .stTextInput input:focus, .stNumberInput input:focus, textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }}

    /* Headings pick up the accent color */
    h1, h2, h3 {{ color: color-mix(in srgb, var(--accent) 70%, black); }}

    .brand-card {{
        text-align: center;
        padding: 18px 10px 16px 10px;
        border-radius: 16px;
        background: color-mix(in srgb, var(--accent) 12%, white);
        border: 1px solid color-mix(in srgb, var(--accent) 30%, white);
        margin-bottom: 12px;
    }}
    .brand-icon {{ font-size: 40px; line-height: 1; }}
    .brand-title {{ font-size: 18px; font-weight: 800; margin-top: 4px; }}
    .brand-subtitle {{ font-size: 11.5px; opacity: 0.75; margin-top: 2px; }}
    .hero-banner {{
        padding: 22px 24px;
        border-radius: 18px;
        background: linear-gradient(135deg, var(--accent) 0%, color-mix(in srgb, var(--accent) 55%, black) 100%);
        margin-bottom: 18px;
    }}
    .hero-greeting {{ font-size: 14px; color: rgba(255,255,255,0.9); margin: 0; }}
    .hero-title {{ font-size: 26px; font-weight: 800; color: white; margin: 4px 0 0 0; }}
    .hero-subtitle {{ font-size: 13.5px; color: rgba(255,255,255,0.9); margin-top: 4px; }}
    .status-pill {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: color-mix(in srgb, var(--accent) 15%, white);
        border: 1px solid color-mix(in srgb, var(--accent) 35%, white);
        font-size: 12px; font-weight: 600;
    }}
    .info-card {{
        border-radius: 14px; padding: 14px 16px;
        background: color-mix(in srgb, var(--accent) 6%, white);
        border: 1px solid color-mix(in srgb, var(--accent) 18%, white);
        margin-bottom: 10px;
    }}
    .chip {{
        display: inline-block; padding: 6px 12px; border-radius: 999px;
        background: color-mix(in srgb, var(--accent) 10%, white);
        border: 1px solid color-mix(in srgb, var(--accent) 25%, white);
        font-size: 12.5px; margin: 3px 4px 3px 0;
    }}
    .history-item {{
        padding: 8px 10px; border-radius: 10px;
        background: color-mix(in srgb, var(--accent) 6%, white);
        margin-bottom: 6px; font-size: 13px;
    }}
    .history-time {{ font-size: 11px; opacity: 0.6; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — branding, nav, profile
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand-card">
            <div class="brand-icon">{APP_ICON}</div>
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

    st.session_state.user_type = st.radio(
        "🍽️ Cooking for:",
        options=["Normal", "Diabetic / Health-conscious"],
        index=["Normal", "Diabetic / Health-conscious"].index(st.session_state.user_type),
        help="Changes how ingredients and tips are tailored.",
    )

    st.divider()
    st.caption("Backend order: Groq → Cohere → Gemini (auto fallback)")


# ---------------------------------------------------------------------------
# PAGE: Home
# ---------------------------------------------------------------------------

def render_home():
    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            """<p class="hero-greeting">👋 Hi! I'm your Kitchen Recipe Agent</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p class="hero-title">What would you like to cook today?</p>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<p class="hero-subtitle">Tell me what you have or what you're craving!</p>""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """<div class="info-card" style="text-align:center;">
            <span class="status-pill">✨ AI is Ready</span><br>
            <span style="font-size:12px; opacity:0.7;">Let's cook smart!</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    if "show_photo_uploader" not in st.session_state:
        st.session_state.show_photo_uploader = False

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**⌨️ Type**")
        st.caption("Describe your ingredients or dish below")
    with c2:
        if st.button("📷 Upload Photo", use_container_width=True):
            st.session_state.show_photo_uploader = not st.session_state.show_photo_uploader
    with c3:
        if st.button("🎙️ Voice Input", use_container_width=True):
            st.info("Voice input isn't wired up in this demo yet — please type instead.")
    with c4:
        if st.button("🔗 Paste Link", use_container_width=True):
            st.info("Link import isn't wired up in this demo yet — please describe the recipe instead.")

    if st.session_state.show_photo_uploader:
        st.markdown("**📷 Upload a photo of your ingredients**")
        if not GEMINI_API_KEY:
            st.warning("Photo analysis needs the Gemini key, which isn't configured — please type your ingredients instead.")
        else:
            photo = st.file_uploader("Upload ingredients or recipe photo", type=["jpg", "jpeg", "png", "webp"])
            if photo is not None:
                st.image(photo, width=240)
                if st.button("🔍 Analyze photo", type="primary"):
                    with st.spinner("👩‍🍳 Looking at your photo..."):
                        try:
                            result = analyze_ingredient_photo(photo.getvalue(), photo.type)
                        except Exception as e:
                            result = None
                            st.error(f"Couldn't analyze the photo right now: {e}")
                    if result and result.strip().upper() != "NONE_DETECTED":
                        st.success(f"I can see: {result}")
                        if st.button(f"Find recipes using: {result}", type="primary"):
                            st.session_state.show_photo_uploader = False
                            ask_agent(f"I have these ingredients from a photo: {result}. What can I cook?")
                            st.rerun()
                    elif result:
                        st.warning("I couldn't clearly identify any food ingredients in that photo — please try a clearer photo or type them instead.")

    st.write("")
    input_col, btn_col = st.columns([4, 1])
    with input_col:
        home_input = st.text_input(
            "quick_prompt",
            placeholder="Example: I have chicken, potato, onion and yogurt",
            label_visibility="collapsed",
        )
    with btn_col:
        find_clicked = st.button("Find Recipes ✨", use_container_width=True, type="primary")

    if find_clicked and home_input.strip():
        ask_agent(home_input.strip())
        st.rerun()

    st.write("")
    st.subheader("💡 Popular ideas")
    st.caption("Tap one to ask the AI Chef directly.")
    ideas = [
        ("🍗", "Quick chicken dinner"),
        ("🥗", "Healthy salad ideas"),
        ("🍝", "30-minute pasta"),
        ("🍲", "Comfort food for a rainy day"),
    ]
    cols = st.columns(4)
    for col, (emoji, label) in zip(cols, ideas):
        with col:
            if st.button(f"{emoji}  {label}", use_container_width=True, key=f"idea_{label}"):
                ask_agent(f"Suggest a recipe for: {label}")
                st.rerun()

    if st.session_state.data["ingredients"]:
        st.write("")
        st.subheader("🥕 From your ingredients list")
        ing_list = st.session_state.data["ingredients"]
        names_display = ", ".join(f"{i['amount']:g}{i['unit']} {i['name']}" for i in ing_list)
        st.markdown(f'<span class="chip">{names_display}</span>', unsafe_allow_html=True)
        if st.button("Find recipes using these ingredients", type="primary"):
            ask_agent(f"I have {names_display}. What can I cook?")
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE: AI Chef Chat
# ---------------------------------------------------------------------------

def render_chat():
    st.markdown(
        f"""
        <div class="hero-banner">
            <p class="hero-title" style="font-size:22px;">👩‍🍳 AI Chef Chat</p>
            <p class="hero-subtitle">Ask me anything about recipes, cooking, ingredients...</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        with st.spinner("Cooking up a response..."):
            ask_agent(prompt)
        st.rerun()

    for i, msg in enumerate(st.session_state.messages):
        avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "|" in msg["content"]:
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
                    st.toast("Recipe saved! Check Saved Recipes.")

    user_prompt = st.chat_input("Ask anything about cooking...")
    if user_prompt:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_prompt)
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("👩‍🍳 Kitchen Recipe Agent is thinking..."):
                ask_agent(user_prompt)
        st.rerun()

    with st.expander(f"💬 Session history ({len(st.session_state.messages)} messages)"):
        if st.button("🗑️ Clear chat history"):
            st.session_state.confirm_clear = True
        if st.session_state.get("confirm_clear"):
            st.warning("This will permanently delete all saved chat history.")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Yes, delete", type="primary", use_container_width=True):
                    clear_chat_history()
                    st.session_state.confirm_clear = False
                    st.rerun()
            with cc2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()
        user_msgs = [m for m in st.session_state.data["conversations"] if m["role"] == "user"]
        for m in reversed(user_msgs[-15:]):
            time_label = m.get("timestamp", "")[11:16]
            preview = m["content"][:60] + ("…" if len(m["content"]) > 60 else "")
            st.markdown(
                f"""<div class="history-item"><span class="history-time">{time_label}</span><br>{preview}</div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PAGE: My Ingredients
# ---------------------------------------------------------------------------

def render_ingredients():
    st.subheader("🥕 My Ingredients")
    with st.form("add_ingredient", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([3, 1.5, 1.5, 1])
        name = c1.text_input("Ingredient", placeholder="e.g. Chicken")
        amount = c2.number_input("Amount", min_value=0.0, step=0.5, format="%.2f")
        unit = c3.selectbox("Unit", UNIT_OPTIONS)
        submitted = c4.form_submit_button("➕ Add", use_container_width=True)
        if submitted and name.strip() and amount > 0:
            st.session_state.data["ingredients"].append(
                {"name": name.strip(), "amount": amount, "unit": unit}
            )
            save_data()
            st.rerun()
        elif submitted and amount <= 0:
            st.warning("Enter an amount greater than 0.")

    ingredients = st.session_state.data["ingredients"]
    if not ingredients:
        st.caption("No ingredients added yet.")
        return

    cols = st.columns(4)
    for i, ing in enumerate(ingredients):
        qty_label = f"{ing['amount']:g} {ing['unit']}"
        with cols[i % 4]:
            st.markdown(
                f"""<div class="info-card" style="text-align:center;">
                <b>{ing['name']}</b><br><span style="opacity:0.7; font-size:12.5px;">{qty_label}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button("Remove", key=f"rm_ing_{i}", use_container_width=True):
                st.session_state.data["ingredients"].pop(i)
                save_data()
                st.rerun()

    st.write("")
    if st.button("🍳 Find recipes using these ingredients", type="primary"):
        names = ", ".join(f"{i['amount']:g}{i['unit']} {i['name']}" for i in ingredients)
        ask_agent(f"I have {names}. What can I cook?")
        st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Shopping List
# ---------------------------------------------------------------------------

def render_shopping_list():
    items = st.session_state.data["shopping_list"]
    st.subheader(f"🛒 Shopping List ({len(items)})")

    with st.form("add_item", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([3, 1.5, 1.5, 1])
        name = c1.text_input("Item", placeholder="e.g. Green chili")
        amount = c2.number_input("Amount", min_value=0.0, step=0.5, format="%.2f")
        unit = c3.selectbox("Unit", UNIT_OPTIONS)
        submitted = c4.form_submit_button("➕ Add", use_container_width=True)
        if submitted and name.strip() and amount > 0:
            items.append({"name": name.strip(), "amount": amount, "unit": unit, "checked": False})
            save_data()
            st.rerun()
        elif submitted and amount <= 0:
            st.warning("Enter an amount greater than 0.")

    if not items:
        st.caption("Your shopping list is empty.")
        return

    for i, item in enumerate(items):
        c1, c2, c3 = st.columns([0.5, 4, 1])
        checked = c1.checkbox("", value=item["checked"], key=f"chk_{i}")
        if checked != item["checked"]:
            items[i]["checked"] = checked
            save_data()
        qty_label = f"{item['amount']:g} {item['unit']}"
        label = f"~~{item['name']}~~" if checked else f"**{item['name']}**"
        c2.markdown(f"{label} — {qty_label}")
        if c3.button("🗑️", key=f"rm_item_{i}"):
            items.pop(i)
            save_data()
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Saved Recipes
# ---------------------------------------------------------------------------

def render_saved_recipes():
    recipes = st.session_state.data["saved_recipes"]
    st.subheader(f"❤️ Saved Recipes ({len(recipes)})")

    if not recipes:
        st.caption("No recipes saved yet — save one from the AI Chef Chat.")
        return

    for i, r in enumerate(reversed(recipes)):
        real_idx = len(recipes) - 1 - i
        with st.expander(f"{r['title']}  ·  saved {r['timestamp'][:10]}"):
            st.markdown(r["content"])
            if st.button("🗑️ Delete", key=f"del_recipe_{real_idx}"):
                recipes.pop(real_idx)
                save_data()
                st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Theme Customize
# ---------------------------------------------------------------------------

def render_theme():
    st.subheader("🎨 Theme Customize")
    st.caption("Pick an accent color — updates the whole interface immediately.")

    cols = st.columns(len(ACCENT_OPTIONS))
    for col, (name, hex_val) in zip(cols, ACCENT_OPTIONS.items()):
        with col:
            is_selected = st.session_state.accent == name
            label = f"✅ {name}" if is_selected else name
            if st.button(label, key=f"accent_{name}", use_container_width=True):
                st.session_state.accent = name
                st.rerun()
            st.markdown(
                f"""<div style="height:24px;border-radius:8px;background:{hex_val};margin-top:4px;"></div>""",
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
