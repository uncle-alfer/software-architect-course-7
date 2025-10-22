# ==== Config ====
PY ?= python3
VENV_DIR_TASK2 ?= .venv_task2
REQUIREMENTS_TASK2 ?= requirements_task2.txt
SEEDS ?= seeds/ru_wiki_hp_titles.txt
SEED := seeds/urls_hp_strict.txt
RAW_DIR ?= data/raw
CLEAN_DIR ?= data/clean
LOGS ?= logs
C ?= 8           # concurrency (6–10 safely)
RPS ?= 4.0       # requests per second (3–5 safely)
TIMEOUT ?= 30    # total per-request timeout, seconds

.PHONY: venv deps reset fetch normalize validate kb

venv:
	$(PY) -m venv $(VENV_DIR_TASK2)

deps: venv
	. $(VENV_DIR_TASK2)/bin/activate && pip install -U pip && pip install -r $(REQUIREMENTS_TASK2)

reset:
	rm -rf $(RAW_DIR) $(CLEAN_DIR) $(LOGS)
	mkdir -p $(RAW_DIR) $(CLEAN_DIR) $(LOGS)

fetch: deps
	. $(VENV_DIR_TASK2)/bin/activate && $(PY) scripts/fetch_wiki.py --seed $(SEED) --out $(RAW_DIR)

normalize: deps
	. $(VENV_DIR_TASK2)/bin/activate && $(PY) scripts/normalize_texts.py $(RAW_DIR) $(CLEAN_DIR)

validate: deps
	. $(VENV_DIR_TASK2)/bin/activate && $(PY) scripts/validate_corpus.py $(CLEAN_DIR)

kb: reset fetch validate normalize
