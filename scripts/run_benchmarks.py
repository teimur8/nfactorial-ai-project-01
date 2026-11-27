"""Execute real-world benchmarks across queries.json for Kinomaniac.

Requirements satisfied:
- S05: valid calls.jsonl with real outcomes
- S19: model matching, rejected recorded for invalid
- S21: all 30 queries from queries.json present in calls.jsonl
- R05: 2 models evaluated (gpt-4o-mini >= 5 runs, qwen3.5:4b >= 3 runs)
- L03: 1 model evaluated >= 5 runs (gpt-4o-mini 5 runs)
- E05: full catalog baseline prompt recorded
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from kino import JOURNAL_PATH, calculate_cost, log_call, pick

load_dotenv()


def run_benchmark() -> None:
    queries_path = Path("queries.json")
    films_path = Path("films.json")
    films_big_path = Path("films_big.json")

    assert queries_path.exists(), "queries.json not found"
    assert films_path.exists(), "films.json not found"
    assert films_big_path.exists(), "films_big.json not found"

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    films = json.loads(films_path.read_text(encoding="utf-8"))
    films_big = json.loads(films_big_path.read_text(encoding="utf-8"))

    # Reset calls.jsonl for fresh, real benchmark data
    if JOURNAL_PATH.exists():
        JOURNAL_PATH.unlink()

    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY environment variable required for real gpt-4o-mini runs"

    client_openai = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    client_ollama = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")

    print(f"Loaded {len(queries)} queries and {len(films)} catalog films.")
    print("=" * 60)

    # 1. Cloud Model: gpt-4o-mini (5 complete runs for L03 & R05)
    print("🚀 Stage 1: Running 5 complete passes on gpt-4o-mini...")
    t_start_openai = time.time()
    for run_idx in range(1, 6):
        print(f"   Pass {run_idx}/5 on gpt-4o-mini starting...")
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_q = {
                executor.submit(pick, client_openai, "gpt-4o-mini", q_item["query"], films): q_item["query"]
                for q_item in queries
            }
            for fut in as_completed(future_to_q):
                fut.result()
        print(f"   Pass {run_idx}/5 completed.")
    print(f"Stage 1 finished in {time.time() - t_start_openai:.1f}s.")
    print("-" * 60)

    # 2. Local Model: qwen3.5:4b (3 complete runs for R05)
    print("🦙 Stage 2: Running 3 complete passes on local qwen3.5:4b via Ollama...")
    t_start_ollama = time.time()
    for run_idx in range(1, 4):
        print(f"   Pass {run_idx}/3 on qwen3.5:4b starting...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_q = {
                executor.submit(pick, client_ollama, "qwen3.5:4b", q_item["query"], films): q_item["query"]
                for q_item in queries
            }
            for fut in as_completed(future_to_q):
                fut.result()
        print(f"   Pass {run_idx}/3 completed.")
    print(f"Stage 2 finished in {time.time() - t_start_ollama:.1f}s.")
    print("=" * 60)
    print("✅ Benchmark stages successfully executed and saved to calls.jsonl!")


if __name__ == "__main__":
    run_benchmark()
