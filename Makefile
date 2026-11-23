.PHONY: help test test-bot check probe eval bot run clean

help:
	@echo "Доступные команды:"
	@echo "  make bot        - Запустить Telegram-бота (uv run python bot.py)"
	@echo "  make test       - Запустить все тесты проекта (uv run pytest -v)"
	@echo "  make test-bot   - Запустить тесты логики Telegram-бота"
	@echo "  make check      - Запустить менторскую автопроверку (kino_check.py)"
	@echo "  make probe      - Запустить контрактную пробу с моками (kino_probe.py)"
	@echo "  make eval       - Запустить батч-прогон по queries.json (check.py)"
	@echo "  make run q=...  - CLI запуск kino.py (например: make run q=\"фильм 90-х\")"
	@echo "  make clean      - Очистить временные кэши (__pycache__, .pytest_cache)"

bot:
	uv run python bot.py

test:
	uv run pytest -v

test-bot:
	uv run pytest tests/test_bot.py -v

check:
	python3 -I check/kino_check.py

probe:
	uv run python check/kino_probe.py

eval:
	uv run python check.py

run:
	uv run python kino.py "$(or $(q),фильм как Интерстеллар)"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
