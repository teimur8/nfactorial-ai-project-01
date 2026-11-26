"""Fetch and build real-world films_big.json catalog from TMDb 5000 open dataset.

Dataset Source: The Movie Database (TMDb 5000 Movies Dataset)
License: CC0 1.0 Universal (Public Domain Dedication)
URL: https://huggingface.co/datasets/alejandrowallace/tmdb-5000

Why: Eliminates synthetic placeholder entries ('Таинственный Горизонт 51') by
populating films_big.json with 1050 verified, real-world films (E01 requirement).
"""

import csv
import io
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

TMDB_DATASET_URL = (
    "https://huggingface.co/datasets/alejandrowallace/tmdb-5000/resolve/main/tmdb_5000_movies.csv"
)

GENRE_TRANSLATION: dict[str, str] = {
    "Action": "боевик",
    "Adventure": "приключения",
    "Animation": "мультфильм",
    "Comedy": "комедия",
    "Crime": "криминал",
    "Documentary": "документальный",
    "Drama": "драма",
    "Family": "семейный",
    "Fantasy": "фэнтези",
    "History": "история",
    "Horror": "ужасы",
    "Music": "музыка",
    "Mystery": "детектив",
    "Romance": "мелодрама",
    "Science Fiction": "фантастика",
    "TV Movie": "телефильм",
    "Thriller": "триллер",
    "War": "военный",
    "Western": "вестерн",
}

COUNTRY_TRANSLATION: dict[str, str] = {
    "United States of America": "США",
    "United Kingdom": "Великобритания",
    "France": "Франция",
    "Germany": "Германия",
    "Italy": "Италия",
    "Spain": "Испания",
    "Canada": "Канада",
    "Australia": "Австралия",
    "Japan": "Япония",
    "China": "Китай",
    "Hong Kong": "Гонконг",
    "India": "Индия",
    "South Korea": "Южная Корея",
    "Russia": "Россия",
    "Kazakhstan": "Казахстан",
    "New Zealand": "Новая Зеландия",
    "Mexico": "Мексика",
    "Brazil": "Бразилия",
    "Sweden": "Швеция",
    "Denmark": "Дания",
    "Norway": "Норвегия",
    "Finland": "Финляндия",
    "Ireland": "Ирландия",
    "Poland": "Польша",
    "Czech Republic": "Чехия",
    "Netherlands": "Нидерланды",
    "Belgium": "Бельгия",
    "Switzerland": "Швейцария",
}

LANGUAGE_TRANSLATION: dict[str, str] = {
    "English": "английский",
    "Russian": "русский",
    "Kazakh": "казахский",
    "French": "французский",
    "German": "немецкий",
    "Spanish": "испанский",
    "Italian": "итальянский",
    "Japanese": "японский",
    "Korean": "корейский",
    "Chinese": "китайский",
    "Cantonese": "кантонский",
    "Mandarin": "мандарин",
    "Hindi": "хинди",
    "Portuguese": "португальский",
    "Swedish": "шведский",
    "Arabic": "арабский",
}

FIELDS: dict[str, Any] = {
    "id": (str, int),
    "title": str,
    "year": int,
    "minutes": int,
    "genres": list,
    "countries": list,
    "languages": list,
    "description": str,
}


def parse_json_field(raw: str, key: str = "name") -> list[str]:
    """Safely parse JSON-encoded lists from CSV cells."""
    if not raw or raw.strip() == "":
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item.get(key, "")) for item in data if isinstance(item, dict) and item.get(key)]
    except Exception:
        pass
    return []


def translate_single(text: str) -> str:
    """Translate single string text from English to Russian."""
    if not text.strip():
        return ""
    data = urllib.parse.urlencode({
        "client": "gtx",
        "sl": "en",
        "tl": "ru",
        "dt": "t",
        "q": text,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://translate.googleapis.com/translate_a/single",
        data=data,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = json.loads(r.read().decode("utf-8"))
            return "".join(x[0] for x in raw[0] if x[0]).strip()
    except Exception:
        return text


def translate_batch(batch: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Translate batch of (title, description) from English to Russian with retry."""
    if not batch:
        return []

    payload = "\n~~~~\n".join(f"{t}\n===\n{d}" for t, d in batch)
    data = urllib.parse.urlencode({
        "client": "gtx",
        "sl": "en",
        "tl": "ru",
        "dt": "t",
        "q": payload,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://translate.googleapis.com/translate_a/single",
        data=data,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = json.loads(r.read().decode("utf-8"))
                res = "".join(x[0] for x in raw[0] if x[0])
            items = res.split("~~~~")
            if len(items) == len(batch):
                results: list[tuple[str, str]] = []
                for (orig_t, orig_d), it in zip(batch, items):
                    parts = it.strip().split("===")
                    tr_t = parts[0].strip() if len(parts) > 0 and parts[0].strip() else orig_t
                    tr_d = parts[1].strip() if len(parts) > 1 and parts[1].strip() else orig_d
                    results.append((tr_t, tr_d))
                return results
        except Exception:
            time.sleep(1.0 + attempt * 0.5)

    fallback_results: list[tuple[str, str]] = []
    for t, d in batch:
        tr_t = translate_single(t)
        tr_d = translate_single(d)
        fallback_results.append((tr_t or t, tr_d or d))
    return fallback_results


def build_real_catalog(output_path: Path = Path("films_big.json")) -> list[dict]:
    """Download TMDb dataset and construct normalized films_big.json catalog."""
    print("1. Loading base catalog from films.json...")
    base_films = json.loads(Path("films.json").read_text(encoding="utf-8"))
    print(f"   Loaded {len(base_films)} base verified films.")

    print(f"2. Downloading real-world movie dataset from {TMDB_DATASET_URL}...")
    req = urllib.request.Request(TMDB_DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        csv_text = resp.read().decode("utf-8", errors="replace")

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    print(f"   Downloaded {len(rows)} raw TMDb records.")

    def sort_score(r: dict) -> float:
        try:
            return float(r.get("popularity", 0)) * 10 + float(r.get("vote_count", 0))
        except ValueError:
            return 0.0

    rows.sort(key=sort_score, reverse=True)

    result_films: list[dict] = []
    # Retain the exact verified films from films.json (with IDs as str)
    for f in base_films:
        item = dict(f)
        item["id"] = str(item["id"])
        item.setdefault("source", f"https://www.themoviedb.org/search?query={urllib.parse.quote(str(item['title']))}")
        result_films.append(item)

    base_titles = {f["title"].lower() for f in base_films}
    next_id = 51

    print("3. Processing TMDb records into catalog format...")
    raw_tmdb_candidates: list[dict] = []
    for row in rows:
        if len(result_films) + len(raw_tmdb_candidates) >= 1050:
            break

        title = (row.get("title") or "").strip()
        if not title or title.lower() in base_titles:
            continue

        overview = (row.get("overview") or "").strip()
        if not overview:
            continue

        rel_date = (row.get("release_date") or "").strip()
        if not rel_date or len(rel_date) < 4:
            continue
        try:
            year = int(rel_date[:4])
        except ValueError:
            continue

        raw_runtime = (row.get("runtime") or "").strip()
        if not raw_runtime:
            continue
        try:
            minutes = int(round(float(raw_runtime)))
            if minutes <= 0:
                continue
        except ValueError:
            continue

        genres_raw = parse_json_field(row.get("genres", ""))
        genres = [GENRE_TRANSLATION.get(g, g.lower()) for g in genres_raw] or ["драма"]

        countries_raw = parse_json_field(row.get("production_countries", ""))
        countries = [COUNTRY_TRANSLATION.get(c, c) for c in countries_raw] or ["США"]

        languages_raw = parse_json_field(row.get("spoken_languages", ""))
        languages = [LANGUAGE_TRANSLATION.get(l, "английский") for l in languages_raw] or ["английский"]

        tmdb_id = row.get("id", "")
        source_url = f"https://www.themoviedb.org/movie/{tmdb_id}" if tmdb_id else "https://www.themoviedb.org"

        movie_record = {
            "id": str(next_id),
            "title": title,
            "year": year,
            "minutes": minutes,
            "genres": genres,
            "countries": countries,
            "languages": languages,
            "description": overview,
            "source": source_url,
        }

        raw_tmdb_candidates.append(movie_record)
        base_titles.add(title.lower())
        next_id += 1

    print(f"3.1. Translating {len(raw_tmdb_candidates)} titles and descriptions to Russian...")
    batch_size = 5
    batches = [
        [(c["title"], c["description"]) for c in raw_tmdb_candidates[i : i + batch_size]]
        for i in range(0, len(raw_tmdb_candidates), batch_size)
    ]

    with ThreadPoolExecutor(max_workers=6) as pool:
        future_to_idx = {pool.submit(translate_batch, b): idx for idx, b in enumerate(batches)}
        results_by_idx: dict[int, list[tuple[str, str]]] = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results_by_idx[idx] = future.result()
            if len(results_by_idx) % 30 == 0 or len(results_by_idx) == len(batches):
                print(f"     Translated {len(results_by_idx)} / {len(batches)} batches...")

    translated_pairs: list[tuple[str, str]] = []
    for idx in range(len(batches)):
        translated_pairs.extend(results_by_idx[idx])

    for candidate, (tr_title, tr_desc) in zip(raw_tmdb_candidates, translated_pairs):
        candidate["title"] = tr_title
        candidate["description"] = tr_desc
        result_films.append(candidate)

    print(f"4. Validating {len(result_films)} catalog records...")
    # Validation checks
    assert len(result_films) >= 1000, f"Expected >= 1000 films, got {len(result_films)}"
    ids = [f["id"] for f in result_films]
    assert len(ids) == len(set(ids)), "Duplicate IDs found!"
    assert all(isinstance(i, str) for i in ids), "Non-string IDs detected!"

    for n, f in enumerate(result_films):
        for field, kind in FIELDS.items():
            assert field in f, f"Missing field {field} in record {n}"
            assert isinstance(f[field], kind), f"Field {field} in record {n} has invalid type {type(f[field])}"

    output_path.write_text(json.dumps(result_films, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"5. Successfully wrote {len(result_films)} real films to {output_path}!")
    return result_films


if __name__ == "__main__":
    import urllib.parse

    build_real_catalog()
