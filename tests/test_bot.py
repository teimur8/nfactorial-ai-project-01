"""Unit tests for Telegram bot dialog logic and session management (Legendary tier)."""

from unittest.mock import MagicMock
import pytest

from bot import ChatSession, get_main_keyboard, handle_user_message, is_ambiguous_query, is_more_command, parse_exclusion


@pytest.fixture
def catalog():
    """Sample catalog for testing bot session behavior."""
    return [
        {
            "id": 1,
            "title": "Брат",
            "year": 1997,
            "minutes": 100,
            "genres": ["криминал", "драма"],
            "countries": ["Россия"],
            "languages": ["русский"],
            "description": "Данила Багров отправляется в Санкт-Петербург.",
        },
        {
            "id": 2,
            "title": "Брат 2",
            "year": 2000,
            "minutes": 127,
            "genres": ["криминал", "боевик"],
            "countries": ["Россия"],
            "languages": ["русский", "английский"],
            "description": "Данила едет в Америку за справедливостью.",
        },
        {
            "id": 3,
            "title": "Интерстеллар",
            "year": 2014,
            "minutes": 169,
            "genres": ["фантастика", "драма"],
            "countries": ["США"],
            "languages": ["английский"],
            "description": "Полет сквозь червоточину в поисках нового дома.",
        },
    ]


def test_is_more_command():
    """Verify recognition of various 'more' / 'еще' commands."""
    assert is_more_command("еще")
    assert is_more_command("Ещё")
    assert is_more_command("🔄 Ещё")
    assert is_more_command("покажи еще")
    assert is_more_command("ещё фильмы")
    assert is_more_command("more")
    assert is_more_command("/more")
    assert not is_more_command("фильм про космос")


def test_is_ambiguous_query():
    """Verify detection of generic and ambiguous queries."""
    assert is_ambiguous_query("что посмотреть")
    assert is_ambiguous_query("посоветуй фильм")
    assert is_ambiguous_query("кино на вечер")
    assert is_ambiguous_query("фильм")
    assert is_ambiguous_query("кино")
    assert is_ambiguous_query("что-нибудь интересное")
    assert not is_ambiguous_query("корейский детективный триллер")
    assert not is_ambiguous_query("фильм про космос и черную дыру")


def test_main_keyboard():
    """Verify main reply keyboard configuration."""
    kb = get_main_keyboard()
    assert kb.resize_keyboard is True
    button_texts = [btn.text for row in kb.keyboard for btn in row]
    assert "🔄 Ещё" in button_texts
    assert "❓ Что посмотреть?" in button_texts
    assert "🔄 Сброс (/reset)" in button_texts


def test_parse_exclusion(catalog):
    """Verify parsing user exclusions like 'уже смотрел Брат' (BOT-03)."""
    matched, msg = parse_exclusion("уже смотрел Брат", catalog)
    assert matched is not None
    assert len(matched) >= 1
    assert any(f["title"] == "Брат" for f in matched)
    assert "исключен" in msg.lower() or "запомнил" in msg.lower()

    # Non-exclusion query
    no_match, msg2 = parse_exclusion("посоветуй фантастику", catalog)
    assert no_match is None
    assert msg2 is None


def test_bot_session_tracks_seen_ids():
    """Verify ChatSession records seen film IDs and prevents re-recommendation."""
    session = ChatSession()
    assert not session.is_seen(1)
    session.add_seen([1, "2"])
    assert session.is_seen(1)
    assert session.is_seen("1")
    assert session.is_seen(2)
    assert not session.is_seen(3)


def test_bot_clarify_flow(catalog, monkeypatch):
    """Verify BOT-01: ambiguous query returns clarifying prompt without films."""
    session = ChatSession()
    fake_client = MagicMock()

    # Mock pick to simulate clarify status
    def mock_pick(client, model, query, films):
        return "clarify", []

    monkeypatch.setattr("bot.pick_big", mock_pick)

    response = handle_user_message(session, "кино", catalog, fake_client, "gpt-4o-mini")
    assert "уточните" in response.lower() or "какой жанр" in response.lower() or "настроение" in response.lower()
    assert len(session.seen_ids) == 0


def test_bot_more_flow_no_duplicates(catalog, monkeypatch):
    """Verify BOT-02: 'еще' gives new films without repeating previously shown IDs."""
    session = ChatSession()
    fake_client = MagicMock()

    # First turn: returns film id=1
    def mock_pick_1(client, model, query, films):
        assert len(films) == 3
        return "ok", [1]

    monkeypatch.setattr("bot.pick_big", mock_pick_1)
    resp1 = handle_user_message(session, "криминал", catalog, fake_client, "gpt-4o-mini")
    assert "Брат" in resp1
    assert session.is_seen(1)

    # Second turn: user says "еще"
    def mock_pick_2(client, model, query, films):
        # Film 1 should be excluded from candidate catalog
        film_ids = [f["id"] for f in films]
        assert 1 not in film_ids
        assert 2 in film_ids
        return "ok", [2]

    monkeypatch.setattr("bot.pick_big", mock_pick_2)
    resp2 = handle_user_message(session, "еще", catalog, fake_client, "gpt-4o-mini")
    assert "Брат 2" in resp2
    assert session.is_seen(2)


def test_bot_user_exclusion_persists(catalog, monkeypatch):
    """Verify BOT-03: user exclusions remove films from subsequent queries in session."""
    session = ChatSession()
    fake_client = MagicMock()

    # User announces they already watched "Брат"
    resp_ex = handle_user_message(session, "уже смотрел Брат", catalog, fake_client, "gpt-4o-mini")
    assert "исключен" in resp_ex.lower() or "запомнил" in resp_ex.lower()
    assert session.is_seen(1)

    # Now user asks for a crime movie
    def mock_pick(client, model, query, films):
        film_ids = [f["id"] for f in films]
        assert 1 not in film_ids
        return "ok", [2]

    monkeypatch.setattr("bot.pick_big", mock_pick)
    resp_rec = handle_user_message(session, "посоветуй криминал", catalog, fake_client, "gpt-4o-mini")
    assert "Брат 2" in resp_rec


def test_bot_query_interstellar_not_space(monkeypatch):
    """Test phrase: 'Посоветуй фильм как Интерстеллар, но не про космос'."""
    session = ChatSession()
    fake_client = MagicMock()
    test_catalog = [
        {
            "id": 9,
            "title": "Интерстеллар",
            "year": 2014,
            "minutes": 169,
            "genres": ["фантастика", "драма"],
            "countries": ["США"],
            "languages": ["английский"],
            "description": "Полет сквозь червоточину в поисках нового дома в космосе.",
        },
        {
            "id": 10,
            "title": "Начало",
            "year": 2010,
            "minutes": 148,
            "genres": ["фантастика", "боевик", "триллер"],
            "countries": ["США", "Великобритания"],
            "languages": ["английский"],
            "description": "Кобб похищает секреты из подсознания во время сна.",
        },
    ]

    def mock_pick(client, model, query, films):
        # Must recommend Inception (id 10), forbidding Interstellar (id 9)
        return "ok", [10]

    monkeypatch.setattr("bot.pick_big", mock_pick)
    query = "Посоветуй фильм как Интерстеллар, но не про космос"
    response = handle_user_message(session, query, test_catalog, fake_client, "gpt-4o-mini")

    assert "Начало" in response
    assert "2010" in response
    assert "148 мин" in response
    assert "фантастика" in response
    assert session.is_seen(10)
    assert not session.is_seen(9)


def test_bot_query_kazakh_under_two_hours(monkeypatch):
    """Test phrase: 'Что посмотреть с мамой на казахском, до двух часов?'."""
    session = ChatSession()
    fake_client = MagicMock()
    test_catalog = [
        {
            "id": 3,
            "title": "Тюльпан",
            "year": 2008,
            "minutes": 100,
            "genres": ["драма", "комедия"],
            "countries": ["Казахстан"],
            "languages": ["казахский", "русский"],
            "description": "Моряк Аса возвращается в казахскую степь к семье.",
        },
        {
            "id": 7,
            "title": "Томирис",
            "year": 2019,
            "minutes": 156,
            "genres": ["история", "военный", "драма"],
            "countries": ["Казахстан"],
            "languages": ["казахский"],
            "description": "История царицы массагетов Томирис.",
        },
    ]

    def mock_pick(client, model, query, films):
        return "ok", [3]

    monkeypatch.setattr("bot.pick_big", mock_pick)
    query = "Что посмотреть с мамой на казахском, до двух часов?"
    response = handle_user_message(session, query, test_catalog, fake_client, "gpt-4o-mini")

    assert "Тюльпан" in response
    assert "100 мин" in response
    assert "2008" in response
    assert "Томирис" not in response
    assert session.is_seen(3)
    assert not session.is_seen(7)


def test_bot_query_almaty_not_melodrama(monkeypatch):
    """Test phrase: 'доброе кино про Алматы, только не мелодрама'."""
    session = ChatSession()
    fake_client = MagicMock()
    test_catalog = [
        {
            "id": 1,
            "title": "Игла",
            "year": 1988,
            "minutes": 81,
            "genres": ["драма", "криминал"],
            "countries": ["Казахстан"],
            "languages": ["русский"],
            "description": "Моро возвращается в Алма-Ату к любимой девушке.",
        },
        {
            "id": 99,
            "title": "Слезная мелодрама",
            "year": 2015,
            "minutes": 95,
            "genres": ["мелодрама"],
            "countries": ["Казахстан"],
            "languages": ["русский"],
            "description": "Романтическая история в Алматы.",
        },
    ]

    def mock_pick(client, model, query, films):
        return "ok", [1]

    monkeypatch.setattr("bot.pick_big", mock_pick)
    query = "доброе кино про Алматы, только не мелодрама"
    response = handle_user_message(session, query, test_catalog, fake_client, "gpt-4o-mini")

    assert "Игла" in response
    assert "81 мин" in response
    assert "1988" in response
    assert "Слезная мелодрама" not in response
    assert session.is_seen(1)
    assert not session.is_seen(99)
