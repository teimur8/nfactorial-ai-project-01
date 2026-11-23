"""Script to generate films_big.json (1050 films) with valid types for Epic tier."""

import json
from pathlib import Path

films_path = Path("films.json")
with films_path.open(encoding="utf-8") as f:
    base_films = json.load(f)

# Extended genres, countries, adjectives, nouns for realistic catalog generation
GENRE_POOL = [
    ["драма"], ["комедия"], ["триллер"], ["боевик"], ["фантастика"],
    ["детектив"], ["приключения"], ["фэнтези"], ["криминал"], ["мелодрама"],
    ["история", "драма"], ["комедия", "мелодрама"], ["боевик", "триллер"],
    ["фантастика", "приключения"], ["драма", "биография"], ["семейный", "комедия"],
    ["мультфильм", "приключения"], ["аниме", "фэнтези"], ["военный", "драма"],
    ["ужасы", "детектив"]
]

COUNTRY_POOL = [
    ["США"], ["Великобритания"], ["Франция"], ["Германия"], ["Италия"],
    ["Япония"], ["Республика Корея"], ["Казахстан"], ["Канада"], ["Испания"],
    ["Россия"], ["Австралия"], ["Китай"], ["Индия"], ["Швеция"]
]

LANG_POOL = [
    ["английский"], ["французский"], ["немецкий"], ["итальянский"], ["японский"],
    ["корейский"], ["казахский"], ["русский"], ["испанский"], ["шведский"]
]

ADJECTIVES = [
    "Таинственный", "Великий", "Последний", "Одинокий", "Золотой", "Холодный",
    "Невидимый", "Железный", "Далёкий", "Скрытый", "Северный", "Солнечный",
    "Вечный", "Ночной", "Бегущий", "Забытый", "Храбрый", "Глубокий", "Тёмный", "Чистый"
]

NOUNS = [
    "Горизонт", "Свет", "Путь", "След", "Океан", "Остров", "Ветер", "Рубеж",
    "Закат", "Рассвет", "Голос", "Берег", "Огонь", "Мираж", "Дворец", "Мост",
    "Секрет", "Корабль", "Странник", "Хранитель"
]

big_catalog = list(base_films)
current_id = 51

while len(big_catalog) < 1050:
    adj = ADJECTIVES[(current_id * 7) % len(ADJECTIVES)]
    noun = NOUNS[(current_id * 11) % len(NOUNS)]
    title = f"{adj} {noun} {current_id}"
    year = 1960 + (current_id * 3) % 65
    minutes = 75 + (current_id * 5) % 115
    genres = GENRE_POOL[current_id % len(GENRE_POOL)]
    countries = COUNTRY_POOL[(current_id // 3) % len(COUNTRY_POOL)]
    languages = LANG_POOL[(current_id // 5) % len(LANG_POOL)]
    desc = f"Захватывающая история о приключениях и поиске истины в проекте {title}."

    big_catalog.append({
        "id": current_id,
        "title": title,
        "year": year,
        "minutes": minutes,
        "genres": genres,
        "countries": countries,
        "languages": languages,
        "description": desc,
    })
    current_id += 1

out_path = Path("films_big.json")
with out_path.open("w", encoding="utf-8") as f:
    json.dump(big_catalog, f, ensure_ascii=False, indent=2)

print(f"Generated {len(big_catalog)} films in films_big.json")
