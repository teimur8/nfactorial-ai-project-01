"""Log analytics and token/cost aggregation tool for Kinomaniac calls.jsonl."""

import json
from collections import Counter
from pathlib import Path

from tabulate import tabulate

JOURNAL_PATH = Path("calls.jsonl")


def main() -> None:
    if not JOURNAL_PATH.exists():
        print("Файл calls.jsonl не найден.")
        return

    rows = []
    broken = 0
    with JOURNAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                broken += 1

    if not rows:
        print("Журнал calls.jsonl пуст.")
        return

    total_calls = len(rows)
    models_counter = Counter(r.get("model", "unknown") for r in rows)
    outcomes_counter = Counter(r.get("outcome", "unknown") for r in rows)
    steps_counter = Counter(r.get("step", "unknown") for r in rows)

    total_prompt_tokens = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows)
    total_completion_tokens = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
    total_tokens = total_prompt_tokens + total_completion_tokens

    total_usd = sum(r.get("usd", 0.0) for r in rows if isinstance(r.get("usd"), (int, float)))
    rejected_list = [i for r in rows for i in (r.get("rejected") or [])]

    print("=" * 60)
    print(" 📊 АНАЛИТИКА ЖУРНАЛА ВЫЗОВОВ (calls.jsonl)")
    print("=" * 60)

    summary_table = [
        ["Всего записей в журнале", total_calls],
        ["Битых строк", broken],
        ["Всего токенов (Prompt)", f"{total_prompt_tokens:,}"],
        ["Всего токенов (Completion)", f"{total_completion_tokens:,}"],
        ["Всего токенов (Суммарно)", f"{total_tokens:,}"],
        ["Суммарная стоимость (USD)", f"${total_usd:.4f}"],
        ["Отклонённых ID (rejected)", len(rejected_list)],
    ]
    print(tabulate(summary_table, tablefmt="github"))
    print()

    print("--- Распределение по моделям ---")
    model_table = [[m, count] for m, count in models_counter.most_common()]
    print(tabulate(model_table, headers=["Модель", "Количество вызовов"], tablefmt="github"))
    print()

    print("--- Распределение по исходам ---")
    outcome_table = [[out, count] for out, count in outcomes_counter.most_common()]
    print(tabulate(outcome_table, headers=["Исход", "Количество"], tablefmt="github"))
    print()

    print("--- Распределение по этапам (Steps) ---")
    steps_table = [[st, count] for st, count in steps_counter.most_common()]
    print(tabulate(steps_table, headers=["Этап", "Количество"], tablefmt="github"))
    print()

    if rejected_list:
        print(f"Примеры отклонённых ID: {rejected_list[:10]}")


if __name__ == "__main__":
    main()
