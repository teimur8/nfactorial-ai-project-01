"""Experiment measuring LLM factual hallucinations without catalog context."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tabulate import tabulate

load_dotenv()


class MovieFact(BaseModel):
    title: str
    year: int
    minutes: int


class MemoryExperimentResponse(BaseModel):
    movies: list[MovieFact]


def main() -> None:
    films_path = Path("films.json")
    with films_path.open(encoding="utf-8") as f:
        films = json.load(f)

    # 10 test movies across catalog
    test_ids = [1, 2, 4, 9, 11, 15, 23, 31, 41, 48]
    test_films = [f for f in films if f["id"] in test_ids]

    titles_str = ", ".join(f'"{f["title"]}"' for f in test_films)
    prompt = (
        f"Укажи точный год премьеры и точную длительность в минутах для следующих 10 фильмов: {titles_str}."
    )

    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

    client = OpenAI(api_key=api_key, base_url=base_url)

    print("Запрос к модели без каталога...")
    try:
        completion = client.chat.completions.parse(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=MemoryExperimentResponse,
        )
        parsed = completion.choices[0].message.parsed
    except Exception as e:
        print(f"Ошибка вызова модели: {e}")
        return

    model_answers = {m.title.lower(): m for m in parsed.movies} if parsed else {}

    discrepancies = 0
    table = []

    for film in test_films:
        title = film["title"]
        true_year = film["year"]
        true_min = film["minutes"]

        ans = model_answers.get(title.lower())
        pred_year = ans.year if ans else "N/A"
        pred_min = ans.minutes if ans else "N/A"

        diff_year = abs(true_year - pred_year) if isinstance(pred_year, int) else "N/A"
        diff_min = abs(true_min - pred_min) if isinstance(pred_min, int) else "N/A"

        has_error = (diff_year != 0) or (diff_min != 0)
        if has_error:
            discrepancies += 1

        table.append([
            title,
            f"{true_year} / {true_min} мин",
            f"{pred_year} / {pred_min} мин",
            f"Δ года: {diff_year}, Δ мин: {diff_min}",
            "❌ Ошибка" if has_error else "✅ Совпало",
        ])

    print("\n--- Результаты эксперимента с памятью модели ---")
    headers = ["Фильм", "Каталог (эталон)", "Ответ модели", "Расхождение", "Статус"]
    print(tabulate(table, headers=headers, tablefmt="github"))
    print(f"\nИтого расхождений с каталогом: {discrepancies} из 10 ({discrepancies * 10}%)")


if __name__ == "__main__":
    main()
