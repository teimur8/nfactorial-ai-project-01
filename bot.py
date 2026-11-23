"""Telegram Bot interface for Kinomaniac recommender system (Legendary Tier).

Features:
- BOT-01: Clarification dialogues for ambiguous requests without film recommendations.
- BOT-02: 'Ещё' ('more') command recommendations without repeating already shown films.
- BOT-03: Persistent exclusion of watched/unwanted movies ('уже смотрел X') within the chat session.
- BOT-04: Works with local LLM (qwen3.5:4b via Ollama) or OpenAI API.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv
from openai import OpenAI

from kino import pick_big

load_dotenv()
logger = logging.getLogger(__name__)


class ChatSession:
    """Manages conversational state and exclusions for a single chat user."""

    def __init__(self) -> None:
        self.seen_ids: set[str] = set()
        self.last_query: str | None = None

    def add_seen(self, ids: list[Any]) -> None:
        """Mark film IDs as seen or excluded."""
        for fid in ids:
            self.seen_ids.add(str(fid))

    def is_seen(self, film_id: Any) -> bool:
        """Check whether a film ID has already been recommended or excluded."""
        return str(film_id) in self.seen_ids

    def reset(self) -> None:
        """Reset the conversation state."""
        self.seen_ids.clear()
        self.last_query = None


def is_more_command(text: str) -> bool:
    """Detect if user is asking for more recommendations."""
    cleaned = text.strip().lower()
    return cleaned in {
        "еще",
        "ещё",
        "еще фильмы",
        "ещё фильмы",
        "еще раз",
        "ещё раз",
        "дальше",
        "more",
        "/more",
        "давай еще",
        "давай ещё",
    }


def parse_exclusion(text: str, films: list[dict]) -> tuple[list[dict] | None, str | None]:
    """Extract user exclusion intent like 'уже смотрел <название>' (BOT-03)."""
    patterns = [
        r"(?:уже\s+)?(?:смотрел|видал|видел)\s+(.+)",
        r"(?:исключи|убери|не\s+предлагай)\s+(.+)",
    ]
    raw_title = None
    for pat in patterns:
        m = re.match(pat, text.strip(), re.IGNORECASE)
        if m:
            raw_title = m.group(1).strip(" \"'«».,!?")
            break

    if not raw_title or len(raw_title) < 2:
        return None, None

    normalized_query = raw_title.lower()
    exact_matches = [f for f in films if f.get("title", "").strip().lower() == normalized_query]
    if exact_matches:
        matched = exact_matches
    else:
        matched = []
        for f in films:
            film_title = f.get("title", "").lower()
            if len(normalized_query) >= 4 and normalized_query in film_title:
                matched.append(f)

    if not matched:
        return None, None

    titles = ", ".join(f"«{f.get('title')}»" for f in matched)
    msg = f"✅ Запомнил: {titles} исключен(ы) из рекомендаций до конца диалога."
    return matched, msg


def format_recommendations(films: list[dict], film_ids: list[Any]) -> str:
    """Format recommended movies directly from verified catalog metadata."""
    by_id = {str(f.get("id")): f for f in films}
    lines = ["🎬 **Рекомендации:**\n"]
    for fid in film_ids:
        film = by_id.get(str(fid))
        if not film:
            continue
        title = film.get("title", "Без названия")
        year = film.get("year", "")
        mins = film.get("minutes", "")
        genres = ", ".join(film.get("genres", []))
        countries = ", ".join(film.get("countries", []))
        desc = film.get("description", "")

        lines.append(f"• **{title}** ({year}, {mins} мин)")
        if genres:
            lines.append(f"  🎭 Жанры: {genres}")
        if countries:
            lines.append(f"  🌍 Страна: {countries}")
        if desc:
            lines.append(f"  📖 {desc}")
        lines.append("")

    lines.append("💡 _Напишите **«Ещё»**, чтобы получить другие варианты, или укажите новый запрос._")
    return "\n".join(lines).strip()


def handle_user_message(
    session: ChatSession,
    text: str,
    films: list[dict],
    client: Any,
    model: str,
) -> str:
    """Process incoming user query and return bot response."""
    stripped = text.strip()
    if not stripped:
        return "Пожалуйста, напишите запрос для подбора фильма (например: 'комедия на вечер', 'фильм как Интерстеллар')."

    # Command /reset or /start
    if stripped.lower() in {"/start", "/help"}:
        return (
            "👋 Привет! Я **Киноманьяк** — рекомендательный бот с гарантией фактов.\n\n"
            "🔍 **Как мной пользоваться:**\n"
            "• Напишите пожелание: _«фантастика 90-х», «корейский детектив», «грустная драма»_\n"
            "• Напишите **«Ещё»**, чтобы посмотреть другие варианты без повторов\n"
            "• Напишите **«уже смотрел Брат»**, чтобы исключить фильм из диалога\n"
            "• Команда /reset сбрасывает историю просмотров"
        )

    if stripped.lower() in {"/reset", "сброс"}:
        session.reset()
        return "🔄 История текущего диалога сброшена. Что хотите посмотреть?"

    # Check for exclusion phrases (BOT-03)
    matched_films, exclusion_msg = parse_exclusion(stripped, films)
    if matched_films and exclusion_msg:
        session.add_seen([f.get("id") for f in matched_films])
        return exclusion_msg

    # Determine query string (BOT-02)
    if is_more_command(stripped):
        if not session.last_query:
            return "Вы еще не запрашивали фильмы. Напишите, какой фильм вы ищете!"
        query = session.last_query
    else:
        query = stripped
        session.last_query = query

    # Filter out already seen / excluded film IDs from catalog
    available_films = [f for f in films if not session.is_seen(f.get("id"))]
    if not available_films:
        return "В каталоге больше не осталось непросмотренных фильмов под этот запрос."

    # Call recommendation engine
    outcome, ids = pick_big(client, model, query, available_films)

    if outcome == "ok":
        session.add_seen(ids)
        return format_recommendations(films, ids)

    if outcome == "clarify":
        # Clarifying prompt without films (BOT-01)
        return (
            "❓ Запрос слишком широкий или неясный. "
            "Уточните, пожалуйста: какой жанр, эпоху, настроение или страну вы предпочитаете?"
        )

    if outcome == "no_match":
        return "😔 В каталоге не нашлось подходящих фильмов под указанные критерии. Попробуйте изменить запрос!"

    if outcome == "invalid":
        return "⚠️ Модель предложила некорректный ответ. Защитные механизмы заблокировали выдачу."

    return f"⚠️ Временная ошибка сервиса рекомендаций ({outcome}). Попробуйте позже."


# Global storage for chat sessions in memory
_sessions: dict[int, ChatSession] = {}


def get_session(chat_id: int) -> ChatSession:
    """Retrieve or initialize chat session for user."""
    if chat_id not in _sessions:
        _sessions[chat_id] = ChatSession()
    return _sessions[chat_id]


async def run_bot() -> None:
    """Start Telegram bot polling."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "your_telegram_bot_token_here":
        print("❌ Ошибка: переменная TELEGRAM_BOT_TOKEN не задана в .env")
        print("Получите токен у @BotFather и пропишите его в .env: TELEGRAM_BOT_TOKEN=<ваш_токен>")
        return

    # Load catalog (prefer big catalog if available)
    catalog_path = Path("films_big.json")
    if not catalog_path.exists():
        catalog_path = Path("films.json")

    with catalog_path.open(encoding="utf-8") as f:
        films = json.load(f)

    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

    client = OpenAI(api_key=api_key, base_url=base_url)

    bot = Bot(token=token)
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def cmd_start(message: types.Message) -> None:
        session = get_session(message.chat.id)
        reply = handle_user_message(session, "/start", films, client, model_name)
        await message.answer(reply, parse_mode="Markdown")

    @dp.message(Command("reset"))
    async def cmd_reset(message: types.Message) -> None:
        session = get_session(message.chat.id)
        reply = handle_user_message(session, "/reset", films, client, model_name)
        await message.answer(reply, parse_mode="Markdown")

    @dp.message()
    async def msg_handler(message: types.Message) -> None:
        if not message.text:
            return
        session = get_session(message.chat.id)
        reply = handle_user_message(session, message.text, films, client, model_name)
        # Fallback to plain text if markdown formatting encounters unescaped symbols
        try:
            await message.answer(reply, parse_mode="Markdown")
        except Exception:
            await message.answer(reply)

    print(f"🚀 Киноманьяк Бот запущен! Каталог: {len(films)} фильмов, модель: {model_name}")
    await dp.start_polling(bot)


def main() -> None:
    """Main entrypoint for telegram bot."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
