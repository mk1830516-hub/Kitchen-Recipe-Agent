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

GROQ_MODEL = "llama-3.3-70b-versatile"
COHERE_MODEL = "command-r7b-12-2024"
GEMINI_MODEL = "gemini-3.6-flash"

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
# System prompt (Smart, Multilingual, Strict Domain & Budget/Shopping aware)
# ---------------------------------------------------------------------------

USER_TYPES = [
    "Normal / Adult",
    "Older Adult",
    "Health-conscious / Fitness",
]


def build_system_prompt(user_type: str) -> str:
    base = f"""You are "{APP_TITLE}", a smart, professional personal AI chef and kitchen assistant.
Tagline: "{APP_TAGLINE}"

BEHAVIOR & LANGUAGE GUIDELINES:
- Adapt naturally to whatever language or script the user writes in (English, Urdu, Roman Urdu, Sindhi, etc.). If the user asks in Urdu or Roman Urdu, reply appropriately in that same language or script.
- Keep your answers strictly focused on cooking, recipes, ingredients, kitchen requirements, shopping lists, quantities, and food budgets.
- If the user asks about anything unrelated to food, cooking, or kitchen management (such as coding, general news, politics, homework, etc.), politely apologize and decline, reminding them that you are only a kitchen assistant.
- Provide clear quantities, measurements, step-by-step cooking instructions, ingredient sources or substitutions when helpful, and budget-conscious recommendations aligned with kitchen inventory and shopping lists."""

    profile_notes = {
        "Normal / Adult": "User is a regular adult cook. Keep recipes practical and flavorful.",
        "Older Adult": "User is cooking for an older adult. Keep meals easily digestible, balanced, and lower in heavy sodium/spices.",
        "Health-conscious / Fitness": "User is health-conscious. Focus on clean ingredients, balanced macros, and nutritious meal planning.",
    }

    extra = profile_notes.get(user_type, profile_notes["Normal / Adult"])
    return base + "\n\n" + extra


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
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_prompt}]
    resp = client.chat.completions.create(model=GROQ_MODEL, messages=messages, temperature=0.5, max_tokens=1500)
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
    gemini_history = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in history]
    chat = model.start_chat(history=gemini_history)
    resp = chat.send_message(user_prompt)
    return resp.text


def get_ai_response(system_prompt: str, history: list, user_prompt: str) -> tuple[str, str]:
    errors = []
    if GROQ_API_KEY:
        try:
            reply = call_groq(system_prompt, history, user_prompt)
            if reply and reply.strip():
                return reply, "Groq"
        except Exception as e:
            errors.append(f"Groq: {e}")
    if COHERE_API_KEY:
        try:
            reply = call_cohere(system_prompt, history, user_prompt)
            if reply and reply.strip():
                return reply, "Cohere"
        except Exception as e:
            errors.append(f"Cohere: {e}")
    if GEMINI_API_KEY:
        try:
            reply = call_gemini(system_prompt, history, user_prompt)
            if reply and reply.strip():
                return reply, "Gemini"
        except Exception as e:
            errors.append(f"Gemini: {e}")

    return "⚠️ Sorry, none of the AI backends could respond right now. Please check your API keys.", "None"


def ask_agent(user_prompt: str) -> None:
    user_type = st.session_state.user_type
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    append_and_save("user", user_prompt, user_type)

    system_prompt = build_system_prompt(user_type)
    history_for_model = st.session_state.messages[:-1][-10:]
    reply, backend_used = get_ai_response(system_prompt, history_for_model, user_prompt)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    append_and_save("assistant", reply, user_type)
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
    st.session_state.user_type = "Normal / Adult"

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "accent" not in st.session_state:
    st.session_state.accent = "Green"

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "👩‍🍳"
ACCENT = ACCENT_OPTIONS[st.session_state.accent]

# ---- High-contrast styling --------------------------------------------------
st.markdown(
    f"""
    <style>
    :root {{ --accent: {ACCENT}; }}

    .stApp {{
        background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 5%, white) 0%, white 320px);
    }}

    section[data-testid="stSidebar"] {{
        background: color-mix(in srgb, var(--accent) 4%, white);
        border-right: 1px solid color-mix(in srgb, var(--accent) 15%, white);
    }}

    .stButton > button {{
        border-radius: 10px !important;
        border: 1px solid color-mix(in srgb, var(--accent) 35%, white) !important;
    }}
    .stButton > button[kind="primary"] {{
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: white !important;
        font-weight: 600;
    }}

    [data-testid="stChatMessage"] {{
        background: color-mix(in srgb, var(--accent) 5%, white);
        border-radius: 14px;
        border: 1px solid color-mix(in srgb, var(--accent) 12%, white);
    }}

    h1, h2, h3 {{ color: color-mix(in srgb, var(--accent) 80%, black) !important; }}

    .brand-card {{
        text-align: center; padding: 18px 10px; border-radius: 16px;
        background: color-mix(in srgb, var(--accent) 12%, white);
        border: 1px solid color-mix(in srgb, var(--accent) 30%, white);
        margin-bottom: 12px;
    }}
    .brand-title {{ font-size: 18px; font-weight: 800; margin-top: 4px; }}
    .brand-subtitle {{ font-size: 11.5px; opacity: 0.85; margin-top: 2px; }}

    .hero-banner {{
        padding: 24px 28px; border-radius: 18px;
        background: linear-gradient(135deg, var(--accent) 0%, color-mix(in srgb, var(--accent) 60%, black) 100%);
        margin-bottom: 18px; color: white !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    .hero-greeting {{ font-size: 15px; color: rgba(255,255,255,0.95); margin: 0; font-weight: 500; }}
    .hero-title {{ font-size: 27px; font-weight: 800; color: white !important; margin: 4px 0 0 0; text-shadow: 0 1px 2px rgba(0,0,0,0.15); }}
    .hero-subtitle {{ font-size: 14px; color: rgba(255,255,255,0.95); margin-top: 6px; font-weight: 400; }}

    .status-pill {{
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4);
        font-size: 12px; font-weight: 700; color: white;
    }}
    .chip {{
        display: inline-block; padding: 6px 12px; border-radius: 999px;
        background: color-mix(in srgb, var(--accent) 12%, white);
        border: 1px solid color-mix(in srgb, var(--accent) 30%, white);
        font-size: 12.5px; margin: 3px 4px; font-weight: 500;
    }}
    .history-card {{
        padding: 10px 12px; border-radius: 10px;
        background: white; border: 1px solid color-mix(in srgb, var(--accent) 20%, white);
        margin-bottom: 8px; font-size: 12.5px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — Branding, Navigation, Profile & Linked Chat History
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

    st.session_state.user_type = st.selectbox(
        "🍽️ Kitchen Profile:",
        options=USER_TYPES,
        index=USER_TYPES.index(st.session_state.user_type),
        help="Tailors recipes for general cooking, older adults, or health-conscious meals.",
    )

    st.divider()
    st.subheader("💬 Chat History")
    
    chats = st.session_state.data["conversations"]
    if not chats:
        st.caption("No past conversations yet.")
    else:
        if st.button("🗑️ Clear All History", use_container_width=True):
            clear_chat_history()
            st.rerun()
            
        # Display stored history in sidebar. Deleting a turn deletes both user prompt and assistant reply.
        # Group conversations into pairs (user prompt + assistant response)
        pairs = []
        i = 0
        while i < len(chats):
            if chats[i]["role"] == "user":
                u_turn = chats[i]
                a_turn = chats[i+1] if (i+1 < len(chats) and chats[i+1]["role"] == "assistant") else None
                pairs.append((u_turn, a_turn))
                i += 2 if a_turn else 1
            else:
                i += 1

        for idx, (u_turn, a_turn) in enumerate(reversed(pairs[-8:])):
            snippet = u_turn["content"][:32] + ("..." if len(u_turn["content"]) > 32 else "")
            col_txt, col_del = st.columns([4, 1])
            with col_txt:
                st.markdown(f"<div class='history-card'><b>{u_turn.get('user_type','Normal')}</b><br>{snippet}</div>", unsafe_allow_html=True)
            with col_del:
                if st.button("❌", key=f"del_pair_{idx}", help="Delete this conversation turn"):
                    # Remove from persistent conversations list and sync session messages
                    if u_turn in st.session_state.data["conversations"]:
                        st.session_state.data["conversations"].remove(u_turn)
                    if a_turn and a_turn in st.session_state.data["conversations"]:
                        st.session_state.data["conversations"].remove(a_turn)
                    save_data()
                    st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.data["conversations"]]
                    st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Home
# ---------------------------------------------------------------------------

def render_home():
    current_profile = st.session_state.user_type
    profile_badge = f"Profile: {current_profile}"

    st.markdown(
        f"""
        <div class="hero-banner">
            <span class="status-pill">✨ {profile_badge}</span>
            <p class="hero-greeting" style="margin-top:10px;">👋 Welcome to your Smart Kitchen Dashboard</p>
            <p class="hero-title">What would you like to cook today?</p>
            <p class="hero-subtitle">Ask naturally in any language. Get precise quantities, ingredients, sources, and budget-friendly guidance!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_col, btn_col = st.columns([4, 1])
    with input_col:
        home_input = st.text_input(
            "quick_prompt",
            placeholder="e.g., 6 kg chicken, 5 litres milk — what can I make? (or ask in Urdu/Roman Urdu)",
            label_visibility="collapsed",
        )
    with btn_col:
        find_clicked = st.button("Find Recipes ✨", use_container_width=True, type="primary")

    if find_clicked and home_input.strip():
        ask_agent(home_input.strip())
        st.rerun()

    st.write("")
    st.subheader("💡 Quick Suggestions")
    
    ideas = [
        ("🍗", "Quick chicken curry dinner"),
        ("🍛", "Traditional Chicken Biryani (6 kg batch)"),
        ("🍝", "30-minute creamy pasta"),
        ("🥘", "Healthy family vegetable rice"),
    ]

    cols = st.columns(2)
    for idx, (emoji, label) in enumerate(ideas):
        with cols[idx % 2]:
            if st.button(f"{emoji}  {label}", use_container_width=True, key=f"idea_{idx}"):
                ask_agent(f"Please give me a recipe for: {label}")
                st.rerun()

    if st.session_state.data["ingredients"]:
        st.write("")
        st.subheader("🥕 From your pantry ingredients")
        ing_list = st.session_state.data["ingredients"]
        chips_html = "".join(f'<span class="chip">{i["amount"]} {i["unit"]} {i["name"]}</span>' for i in ing_list)
        st.markdown(chips_html, unsafe_allow_html=True)
        if st.button("Generate recipes using my ingredients", type="primary"):
            summary_str = ", ".join(f"{i['amount']} {i['unit']} {i['name']}" for i in ing_list)
            ask_agent(f"I have these pantry items: {summary_str}. What can I cook according to my kitchen requirements and budget?")
            st.rerun()


# ---------------------------------------------------------------------------
# PAGE: AI Chef Chat
# ---------------------------------------------------------------------------

def render_chat():
    st.markdown(
        f"""
        <div class="hero-banner" style="padding: 18px 24px;">
            <p class="hero-title" style="font-size:22px;">👩‍🍳 AI Chef Chat</p>
            <p class="hero-subtitle">Ask questions, request recipes, check quantities, or manage kitchen requirements.</p>
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

    user_prompt = st.chat_input("Ask anything about cooking, quantities, or budget...")
    if user_prompt:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_prompt)
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("👩‍🍳 Kitchen Recipe Agent is thinking..."):
                ask_agent(user_prompt)
        st.rerun()


# ---------------------------------------------------------------------------
# PAGE: My Ingredients (Strict string name, integer/float amounts & units)
# ---------------------------------------------------------------------------

def render_ingredients():
    st.subheader("🥕 My Ingredients & Pantry")
    st.caption("Add items with exact string names and integer/float quantities (e.g., 6 kg, 5 litres, 500 g).")

    with st.form("add_ingredient", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([3, 1.5, 1.5, 1])
        name = c1.text_input("Ingredient Name (Letters only)", placeholder="e.g. Chicken")
        amount = c2.number_input("Amount (Int / Float)", min_value=0.1, step=1.0, format="%.2f", value=1.0)
        unit = c3.selectbox("Unit", UNIT_OPTIONS)
        submitted = c4.form_submit_button("➕ Add", use_container_width=True)

        if submitted:
            cleaned_name = name.strip()
            # Validate name contains no numbers/digits
            if not cleaned_name:
                st.error("Please enter a valid ingredient name.")
            elif any(char.isdigit() for char in cleaned_name):
                st.error("Ingredient name must be text (strings only) and cannot contain numbers or digits like float/int.")
            else:
                st.session_state.data["ingredients"].append(
                    {"name": cleaned_name, "amount": amount, "unit": unit}
                )
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
                st.markdown(f"**{ing['name']}** — `{ing['amount']} {ing['unit']}`")
            with col_del:
                if st.button("🗑️", key=f"del_ing_{idx}"):
                    ingredients.pop(idx)
                    save_data()
                    st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Shopping List (Strict string item name, integer/float quantity & units)
# ---------------------------------------------------------------------------

def render_shopping_list():
    st.subheader("🛒 Shopping List")
    st.caption("Manage items you need to buy with string names, integer/float quantities, and proper units.")

    with st.form("add_shopping", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([3, 1.5, 1.5, 1])
        item = c1.text_input("Shopping Item (Letters only)", placeholder="e.g. Olive Oil")
        amount = c2.number_input("Quantity (Int / Float)", min_value=0.1, step=1.0, format="%.2f", value=1.0)
        unit = c3.selectbox("Unit", UNIT_OPTIONS, key="shop_unit")
        submitted = c4.form_submit_button("➕ Add Item", use_container_width=True)

        if submitted:
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
                st.markdown(f"<span style='{style}'><b>{shop['item']}</b> — {shop['amount']} {shop['unit']}</span>", unsafe_allow_html=True)
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
            with st.expander(f"📖 {rec['title']} ({rec.get('timestamp','')[:10]})"):
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
