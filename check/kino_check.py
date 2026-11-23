"""Автопроверка проекта «Киноманьяк». Без ключа API.

    py -3 -I kino_check.py ПАПКА_ПРОЕКТА [--live] [--seed N] [--json ФАЙЛ]

На macOS и Linux вместо py -3 пишется python3.

Что делает:
1. берёт файлы последнего коммита (git archive HEAD) и проверяет их без запуска;
2. ставит окружение проекта во временную папку (uv sync --frozen);
3. запускает pytest проекта;
4. запускает kino_probe.py: pick() получает заглушку вместо OpenAI; имя модели,
   id, запросы и чужой каталог случайны в каждом запуске;
5. с флагом --live ещё пять запросов к локальной модели через Ollama
   (по умолчанию qwen3.5:4b на 127.0.0.1:11434), около 2-3 минут.

Код проекта выполняется на этой машине. Ключ в окружении подменяется
фальшивым, адрес API по умолчанию ведёт на 127.0.0.1:9, поэтому платный
вызов модели невозможен.
"""

import argparse
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
REQUIRED = ["pyproject.toml", "uv.lock", ".python-version", ".env.example",
            "README.md", "kino.py", "check.py", "show.py", "films.json",
            "queries.json", "calls.jsonl"]
FIELDS = {"id": (str, int), "title": str, "year": int, "minutes": int,
          "genres": list, "countries": list, "languages": list,
          "description": str}
KEY = re.compile(r"\b(sk-(?:proj-|svcacct-|ant-)?[A-Za-z0-9_\-]{20,}"
                 r"|AIza[0-9A-Za-z_\-]{35}|gsk_[A-Za-z0-9]{40,})")
KAZAKH = re.compile(r"[әғқңөұүһі]", re.IGNORECASE)
NEGATION = re.compile(r"(^|\s)(не|без|кроме|емес|жоқ)(\s|$|,)", re.IGNORECASE)
STUB = re.compile(r"stub|fake|mock|test|заглушк|^model-[0-9a-f]{6}$",
                  re.IGNORECASE)

TITLES = {
    "S01": "файлы на месте",
    "S02": "нет ключей и .env в git, включая историю",
    "S03": "каталог films.json",
    "S04": "набор queries.json",
    "S05": "журнал calls.jsonl читается",
    "S06": "uv sync --frozen",
    "S07": "pytest проекта",
    "S08": "импорт kino.py без запуска, есть pick",
    "S09": "верный ответ показан, название из каталога",
    "S10": "чужой id не показан, исход invalid",
    "S11": "повтор не показан дважды",
    "S12": "не больше трёх фильмов",
    "S13": "no_match",
    "S14": "отказ модели",
    "S15": "обрезанный ответ",
    "S16": "нет связи",
    "S17": "ошибка API 5xx",
    "S18": "теги <каталог> и <запрос>, весь каталог в промпте",
    "S19": "строки журнала: модель из pick, rejected при invalid",
    "S20": "pick работает с переданным каталогом",
    "S21": "журнал совпадает с queries.json",
    "R01": "30+ запросов, у каждого expect",
    "R02": "вызов со схемой Filters",
    "R03": "код отсекает фильм, нарушающий условие",
    "R04": "каталог отфильтрован до выбора",
    "R05": "две модели: каждый запрос набора в журнале 3+ раз",
    "E01": "films_big.json: 1000+ фильмов",
    "E02": "candidates: до 30 id без модели",
    "E03": "pick_big: модели уходят только кандидаты",
    "E04": "pick_big: чужой id не показан",
    "E05": "в журнале есть прогон со всем каталогом в промпте",
    "L01": "SPEC.md, AGENTS.md, pre-commit, CI с pytest",
    "L02": "статус clarify без фильмов",
    "L03": "одна модель: каждый запрос набора в журнале 5+ раз",
    "V00": "локальная модель доступна",
    "V01": "живой запрос 1",
    "V02": "живой запрос 2",
    "V03": "живой запрос 3",
    "V04": "живой запрос 4",
    "V05": "живой запрос 5",
    "probe": "проба отработала",
}
LEVELS = [("Common", "S"), ("Rare", "R"), ("Epic", "E"), ("Legendary", "L")]


class Report:
    def __init__(self):
        self.items = {}
        self.info = []

    def add(self, check, ok, note=""):
        self.items[check] = {"ok": bool(ok), "note": note}

    def failed(self, prefix):
        return sorted(k for k, v in self.items.items()
                      if k.startswith(prefix) and not v["ok"])


def run(cmd, cwd, env, timeout=900):
    try:
        done = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
        return done.returncode, done.stdout + done.stderr
    except subprocess.TimeoutExpired:
        return 124, f"таймаут {timeout} с"
    except FileNotFoundError as error:
        return 127, str(error)


def tail(text, lines=3):
    return " | ".join(x.strip() for x in text.strip().splitlines()[-lines:])


# ---------- проверки без запуска кода ----------

def export_head(repo, target):
    code, out = run(["git", "-C", str(repo), "rev-parse", "HEAD"], repo,
                    os.environ)
    if code:
        return False, "не git-репозиторий или нет коммитов"
    done = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar",
                           "HEAD"], capture_output=True)
    if done.returncode:
        return False, "git archive не сработал"
    safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    with tarfile.open(fileobj=io.BytesIO(done.stdout)) as tar:
        tar.extractall(target, **safe)
    return True, out.strip()[:10]


def check_files(copy, rep):
    missing = [f for f in REQUIRED if not (copy / f).exists()]
    tests = list((copy / "tests").glob("test*.py")) \
        if (copy / "tests").exists() else []
    if not tests:
        missing.append("tests/test*.py")
    rep.add("S01", not missing, "нет: " + ", ".join(missing) if missing else "")


def check_secrets(repo, copy, rep):
    problems = []
    _, tracked = run(["git", "-C", str(repo), "log", "--all", "--name-only",
                      "--pretty=format:"], repo, os.environ)
    if ".env" in {Path(x).name for x in tracked.split()}:
        problems.append(".env был в коммите")
    for path in copy.rglob("*"):
        if path.is_file() and path.stat().st_size < 5_000_000:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if KEY.search(text):
                problems.append(f"похоже на ключ в {path.relative_to(copy)}")
    _, history = run(["git", "-C", str(repo), "log", "--all", "-p",
                      "--no-color"], repo, os.environ, timeout=120)
    if KEY.search(history):
        problems.append("похоже на ключ в истории коммитов")
    rep.add("S02", not problems, "; ".join(sorted(set(problems))))


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except FileNotFoundError:
        return None, f"нет {path.name}"
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return None, f"{path.name} не читается: {error}"


def catalog_problems(films, low, high):
    if not isinstance(films, list):
        return ["каталог должен быть списком"]
    problems = []
    if not low <= len(films) <= high:
        problems.append(f"фильмов {len(films)}, нужно {low}-{high}")
    for n, film in enumerate(films):
        if not isinstance(film, dict):
            problems.append(f"запись {n} не объект")
            break
        for field, kind in FIELDS.items():
            if field not in film:
                problems.append(f"у записи {n} нет поля {field}")
                break
            value = film[field]
            if not isinstance(value, kind) or isinstance(value, bool):
                problems.append(f"у записи {n} поле {field} типа "
                                f"{type(value).__name__}")
                break
        if len(problems) > 5:
            break
    ids = [str(f.get("id")) for f in films if isinstance(f, dict)]
    repeats = [i for i, c in Counter(ids).items() if c > 1]
    if repeats:
        problems.append(f"повтор id: {repeats[:3]}")
    if len({type(f.get("id")) for f in films if isinstance(f, dict)}) > 1:
        problems.append("id разных типов")
    return problems


def usable(films):
    """Каталог, на котором можно запустить код проекта, даже если S03 провален.

    Провал S03 (39 фильмов, четыре казахстанских) не должен лишать студента
    отзыва по коду: проба работает, если у каждой записи есть id и title.
    """
    return (isinstance(films, list) and len(films) >= 4
            and all(isinstance(f, dict) and "id" in f and "title" in f
                    for f in films))


def check_catalog(copy, rep):
    films, error = load_json(copy / "films.json")
    if films is None:
        rep.add("S03", False, error)
        return None
    problems = catalog_problems(films, 40, 60)
    kz = [f for f in films if isinstance(f, dict)
          and "Казахстан" in (f.get("countries") or [])]
    if len(kz) < 5:
        problems.append(f"казахстанских фильмов {len(kz)}, нужно 5")
    rep.add("S03", not problems, "; ".join(problems))
    if not usable(films):
        return None
    sample = random.sample(films, min(3, len(films)))
    rep.info.append("Для сверки с источником: " + "; ".join(
        f"{f.get('title')} ({f.get('year')}, {f.get('minutes')} мин)"
        for f in sample))
    return films


def check_queries(copy, films, rep):
    items, error = load_json(copy / "queries.json")
    if items is None:
        rep.add("S04", False, error)
        return None
    if not isinstance(items, list) or not all(
            isinstance(x, dict) and isinstance(x.get("query"), str)
            for x in items):
        rep.add("S04", False, "нужен список объектов с полем query")
        return None
    ids = {str(f["id"]) for f in films or []}
    no_match = [x for x in items if (x.get("expect") or {}).get("no_match")]
    with_must = [x for x in items if (x.get("expect") or {}).get("must")]
    kazakh = [x for x in items if KAZAKH.search(x["query"])]
    negation = [x for x in items if NEGATION.search(x["query"])]
    problems = []
    if len(items) < 15:
        problems.append(f"запросов {len(items)}, нужно 15")
    if len(no_match) < 3:
        problems.append(f"no_match {len(no_match)}, нужно 3")
    if len(with_must) < 3:
        problems.append(f"с must {len(with_must)}, нужно 3")
    if len(kazakh) < 3:
        problems.append(f"на казахском {len(kazakh)}, нужно 3")
    if len(negation) < 3:
        problems.append(f"с «не/без/кроме» {len(negation)}, нужно 3")
    unknown = sorted({str(i) for x in items for key in ("must", "forbid")
                      for i in (x.get("expect") or {}).get(key, [])} - ids)
    if ids and unknown:
        problems.append(f"в expect id не из каталога: {unknown[:3]}")
    rep.add("S04", not problems, "; ".join(problems))
    with_expect = [x for x in items if x.get("expect")]
    rep.add("R01", len(items) >= 30 and len(with_expect) == len(items),
            f"запросов {len(items)}, с expect {len(with_expect)}")
    return [x["query"] for x in items]


def read_journal(copy):
    path = copy / "calls.jsonl"
    if not path.exists():
        return None, 0
    rows, broken = [], 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows.append(row if isinstance(row, dict) else {})
        except json.JSONDecodeError:
            broken += 1
    return rows, broken


def runs_per_model(rows, queries):
    """Сколько прогонов набора у каждой модели: самый редкий запрос.

    Строки вызова фильтров ("step": "filters") не считаются, иначе у Rare
    один прогон давал бы две строки на запрос. Если поле step студент не
    пишет, у Rare число завышено вдвое, и ментор сверяет его с README.
    """
    result = {}
    for model in {r.get("model") for r in rows}:
        if not isinstance(model, str) or STUB.search(model):
            continue
        requests = [str(r.get("request", "")) for r in rows
                    if r.get("model") == model
                    and r.get("step", "choice") != "filters"]
        counts = [sum(q in req for req in requests) for q in queries]
        result[model] = min(counts) if counts else 0
    return result


def check_journal(copy, queries, rep):
    rows, broken = read_journal(copy)
    if rows is None:
        rep.add("S05", False, "нет calls.jsonl")
        return {}
    outcomes = Counter(r.get("outcome") for r in rows)
    problems = []
    if broken:
        problems.append(f"битых строк {broken}")
    if len(rows) < 15:
        problems.append(f"строк {len(rows)}, нужно не меньше 15")
    if not outcomes.get("ok") or not outcomes.get("no_match"):
        problems.append("в журнале нет ok или no_match")
    if any(not {"outcome", "model", "request"} <= set(r) for r in rows):
        problems.append("есть строки без outcome, model или request")
    rep.add("S05", not problems, "; ".join(problems))
    runs = runs_per_model(rows, queries or [])
    if queries:
        real = [r for r in rows if isinstance(r.get("model"), str)
                and not STUB.search(r["model"])]
        missing = [q for q in queries
                   if not any(q in str(r.get("request", "")) for r in real)]
        rep.add("S21", not missing and bool(real),
                f"запросов из queries.json нет в журнале настоящей модели: "
                f"{len(missing)}, например {missing[0]!r}" if missing else
                "" if real else "в журнале нет строк настоящей модели")
        rep.add("R05", sum(n >= 3 for n in runs.values()) >= 2,
                "прогонов по самому редкому запросу: "
                + (", ".join(f"{m} {n}" for m, n in runs.items()) or "нет"))
        rep.add("L03", any(n >= 5 for n in runs.values()),
                "прогонов по самому редкому запросу: "
                + (", ".join(f"{m} {n}" for m, n in runs.items()) or "нет"))
    models = Counter(r.get("model") for r in rows)
    usd = sum(r.get("usd") or 0 for r in rows
              if isinstance(r.get("usd"), (int, float)))
    rejected = sum(len(r.get("rejected") or []) for r in rows
                   if isinstance(r.get("rejected"), list))
    rep.info.append(
        f"Журнал: {len(rows)} строк; исходы {dict(outcomes)}; модели "
        f"{dict(models)}; ${usd:.4f}; отклонённых id {rejected}. "
        "Эти числа должны совпасть с README.")
    return {"rows": rows}


def check_big(copy, journal, rep):
    path = copy / "films_big.json"
    if not path.exists():
        return False
    films, error = load_json(path)
    problems = [error] if films is None else catalog_problems(films, 1000,
                                                              10 ** 7)
    rep.add("E01", not problems, "; ".join(problems[:4]))
    if films is not None:
        # Прогон «весь каталог в промпте» узнаётся по длине request или,
        # если журнал пишет каталог коротко (id вместо записей), по входным
        # токенам: один токен это 3-4 символа, порог около половины каталога.
        whole = len(json.dumps(films, ensure_ascii=False))
        rows = journal.get("rows", [])
        longest = max((len(str(r.get("request", ""))) for r in rows), default=0)
        tokens = max((int((r.get("usage") or {}).get("prompt_tokens") or 0)
                      for r in rows if isinstance(r.get("usage"), dict)),
                     default=0)
        rep.add("E05", longest >= whole * 0.5 or tokens >= whole / 8,
                f"самый длинный request {longest} символов, больше всего "
                f"входных токенов {tokens}, каталог {whole} символов")
    return True


def check_process(copy, rep):
    if not (copy / "SPEC.md").exists():
        return False
    problems = [f for f in ["SPEC.md", "AGENTS.md", ".pre-commit-config.yaml"]
                if not (copy / f).exists()]
    flows_dir = copy / ".github" / "workflows"
    flows = list(flows_dir.glob("*.y*ml")) if flows_dir.exists() else []
    if not flows:
        problems.append(".github/workflows")
    elif not any("pytest" in f.read_text(encoding="utf-8") for f in flows):
        problems.append("CI не запускает pytest")
    rep.add("L01", not problems, "нет: " + ", ".join(problems) if problems else "")
    return True


# ---------- запуск кода проекта ----------

def project_env(env_dir):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OPENAI", "ANTHROPIC", "VIRTUAL_ENV"))}
    env.update(UV_PROJECT_ENVIRONMENT=str(env_dir), PYTHONUTF8="1",
               PYTHONIOENCODING="utf-8", OPENAI_API_KEY="check-fake-key",
               OPENAI_BASE_URL="http://127.0.0.1:9/v1", NO_COLOR="1")
    return env


def probe(copy, env, config, timeout):
    probe_dir = Path(tempfile.mkdtemp(prefix="kino-probe-"))
    shutil.copy(HERE / "kino_probe.py", probe_dir / "kino_probe.py")
    code, out = run(["uv", "run", "--frozen", "--project", str(copy), "python",
                     str(probe_dir / "kino_probe.py"), str(copy),
                     json.dumps(config)], copy, env, timeout=timeout)
    shutil.rmtree(probe_dir, ignore_errors=True)
    line = next((x for x in reversed(out.splitlines())
                 if x.startswith("PROBE_RESULT ")), None)
    if line is None:
        return [{"id": "probe", "ok": False, "note": tail(out, 5)}]
    return json.loads(line[len("PROBE_RESULT "):])


def ollama_ready(url, model):
    base = url.rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as answer:
            names = [m.get("name") for m in json.load(answer).get("models", [])]
    except OSError as error:
        return False, f"Ollama не отвечает на {base}: {error}"
    if model not in names:
        return False, f"нет модели {model}, команда: ollama pull {model}"
    return True, ""


def run_project(copy, env_dir, args, rep):
    env = project_env(env_dir)
    code, out = run(["uv", "sync", "--frozen"], copy, env)
    rep.add("S06", code == 0, "" if code == 0 else tail(out))
    if code:
        return
    code, out = run(["uv", "run", "--frozen", "pytest", "-q", "-p",
                     "no:cacheprovider"], copy, env)
    passed = re.search(r"(\d+) passed", out)
    count = int(passed.group(1)) if passed else 0
    ok = code == 0 and count >= 3
    rep.add("S07", ok, f"прошло {count}" if ok else tail(out))
    for item in probe(copy, env, {"seed": args.seed}, 600):
        rep.add(item["id"], item["ok"], item["note"])
    if not args.live:
        return
    ready, note = ollama_ready(args.live_url, args.live_model)
    rep.add("V00", ready, note)
    if ready:
        config = {"live": {"url": args.live_url, "model": args.live_model,
                           "timeout": 180}}
        for item in probe(copy, env, config, 2400):
            if item["id"] != "S08":
                rep.add(item["id"], item["ok"], item["note"])


# ---------- отчёт ----------

def print_level(rep, name, prefix):
    bad = rep.failed(prefix)
    verdict = "автопроверка пройдена" if not bad else \
        "есть провалы: " + ", ".join(bad)
    print(f"\n{name}: {verdict}")
    for check in sorted(k for k in rep.items if k.startswith(prefix)):
        item = rep.items[check]
        mark = "ok  " if item["ok"] else "FAIL"
        note = f"  ({item['note']})" if item["note"] else ""
        print(f"  {mark} {check} {TITLES.get(check, '')}{note}")


def print_report(rep, levels, live):
    for name, prefix in LEVELS:
        if prefix != "S" and not levels.get(name):
            print(f"\n{name}: не найден в проекте")
            continue
        print_level(rep, name, prefix)
    if live:
        print_level(rep, "Живой прогон", "V")
    else:
        print("\nЖивой прогон: не запускался (флаг --live)")
    if "probe" in rep.items:
        print(f"\nПроба упала: {rep.items['probe']['note']}")
    print()
    for line in rep.info:
        print(line)
    print("\nРучную часть ментор проверяет отдельно: факты каталога, README, "
          "журнал и видео для Legendary.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--live", action="store_true",
                        help="пять запросов к локальной модели через Ollama")
    parser.add_argument("--live-model", default="qwen3.5:4b")
    parser.add_argument("--live-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--seed", type=int, help="повторить прежний запуск")
    parser.add_argument("--env", type=Path, help="папка окружения uv")
    parser.add_argument("--json", type=Path, help="сохранить итог в JSON")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(10 ** 6)
    repo = args.repo.resolve()
    rep = Report()
    work = Path(tempfile.mkdtemp(prefix="kino-check-"))
    copy = work / "repo"
    copy.mkdir()
    env_dir = args.env.resolve() if args.env else work / "env"
    ok, head = export_head(repo, copy)
    if not ok:
        print(head)
        return 2
    print(f"Проверка {repo.name}, коммит {head}, seed {args.seed}")
    _, count = run(["git", "-C", str(repo), "rev-list", "--count", "HEAD"],
                   repo, os.environ)
    rep.info.append(f"Коммитов: {count.strip()}")
    check_files(copy, rep)
    check_secrets(repo, copy, rep)
    films = check_catalog(copy, rep)
    queries = check_queries(copy, films, rep)
    journal = check_journal(copy, queries, rep)
    levels = {"Epic": check_big(copy, journal, rep),
              "Legendary": check_process(copy, rep)}
    if films is not None and (copy / "kino.py").exists():
        run_project(copy, env_dir, args, rep)
    else:
        rep.add("S06", False, "код не запускался: films.json не читается "
                "или в записях нет id и title, либо нет kino.py")
    levels["Rare"] = "R03" in rep.items or rep.items.get("R01", {}).get("ok")
    for code, level in [("L02", "Legendary"), ("L03", "Legendary"),
                        ("R01", "Rare"), ("R02", "Rare"), ("R05", "Rare")]:
        if not levels[level]:
            rep.items.pop(code, None)
    if not levels["Epic"]:
        rep.items.pop("E05", None)
    print_report(rep, levels, args.live)
    if args.json:
        args.json.write_text(json.dumps(
            {"seed": args.seed, "items": rep.items, "info": rep.info,
             "levels": levels}, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    failed = rep.failed("S") + (rep.failed("V") if args.live else [])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
