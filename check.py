"""Batch evaluation script for Kinomaniac recommender across queries.json."""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tabulate import tabulate

from kino import pick

load_dotenv()


def main() -> None:
    queries_path = Path("queries.json")
    films_path = Path("films.json")

    if not queries_path.exists() or not films_path.exists():
        print("Ошибка: queries.json или films.json не найдены")
        sys.exit(1)

    with queries_path.open(encoding="utf-8") as f:
        queries = json.load(f)

    with films_path.open(encoding="utf-8") as f:
        films = json.load(f)

    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Запуск тестирования на {len(queries)} запросах с моделью '{model_name}'...\n")

    results = []
    outcomes_counter = Counter()
    passed_count = 0
    total_with_expect = 0

    started_total = time.time()

    for idx, item in enumerate(queries, 1):
        query = item["query"]
        expect = item.get("expect", {})

        t0 = time.time()
        outcome, ids = pick(client, model_name, query, films)
        dt = time.time() - t0

        outcomes_counter[outcome] += 1

        is_passed = True
        status_note = "OK"

        if expect:
            total_with_expect += 1
            if expect.get("no_match"):
                if outcome != "no_match":
                    is_passed = False
                    status_note = f"FAIL (ожидался no_match, получен {outcome})"
            else:
                must = [str(x) for x in expect.get("must", [])]
                forbid = [str(x) for x in expect.get("forbid", [])]
                str_ids = [str(x) for x in ids]

                if must and not any(m in str_ids for m in must):
                    is_passed = False
                    status_note = f"FAIL (нет must: {must}, получено: {ids})"

                if forbid and any(f in str_ids for f in forbid):
                    is_passed = False
                    status_note = f"FAIL (содержит forbid: {forbid})"

            if is_passed:
                passed_count += 1
        else:
            status_note = "нет expect"

        results.append([
            idx,
            query[:40] + ("..." if len(query) > 40 else ""),
            outcome,
            ids,
            f"{dt:.2f}s",
            "PASS" if is_passed else status_note,
        ])

    headers = ["#", "Запрос", "Исход", "IDs", "Время", "Статус"]
    print(tabulate(results, headers=headers, tablefmt="github"))

    print("\n--- Сводка тестирования ---")
    print(f"Всего запросов: {len(queries)}")
    print(f"Распределение исходов: {dict(outcomes_counter)}")
    if total_with_expect:
        accuracy = (passed_count / total_with_expect) * 100
        print(f"Пройдено с ожиданиями (Expect Pass Rate): {passed_count}/{total_with_expect} ({accuracy:.1f}%)")
    print(f"Общее время: {time.time() - started_total:.1f} с")


if __name__ == "__main__":
    main()
