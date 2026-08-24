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
    "Green": {"main": "#2f6f4f", "light": "#e8f2ee"},
    "Cyan": {"main": "#0e7c86", "light": "#e4f3f4"},
    "Blue": {"main": "#2563eb", "light": "#eff6ff"},
    "Purple": {"main": "#7c3aed", "light": "#f5f3ff"},
    "Pink": {"main": "#db2777", "light": "#fdf2f8"},
    "Orange": {"main": "#ea580c", "light": "#fff7ed"},
    "Yellow": {"main": "#ca8a04", "light": "#fefce8"},
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
        f"again.\n\nDetails:\n```\n{error_summary}\n
