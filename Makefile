# ==== Config ====
PY ?= python3
VENV_DIR_TASK2 ?= .venv
REQUIREMENTS_TASK2 ?= requirements_task2.txt
SEEDS ?= seeds/ru_wiki_hp_titles.txt
SEED := seeds/urls_hp_strict.txt
RAW_DIR ?= data/raw
CLEAN_DIR ?= data/clean
LOGS ?= logs
C ?= 8           # concurrency (6–10 safely)
RPS ?= 4.0       # requests per second (3–5 safely)
TIMEOUT ?= 30    # total per-request timeout, seconds

# ==== task 2 ====

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

kb: reset fetch normalize validate

# ==== task 3 ====

VENV_DIR_TASK3 ?= .venv
REQUIREMENTS_TASK3 ?= requirements-task3-index.txt
# EMBED_MODEL ?= sentence-transformers/all-MiniLM-L6-v2
EMBED_MODEL ?= intfloat/multilingual-e5-base
KB_DIR      ?= knowledge_base
ARTIFACTS   ?= artifacts

export OMP_NUM_THREADS ?= 8
export MKL_NUM_THREADS ?= 8

.PHONY: kb.index venv3 deps3 kb.search kb.qc kb.index.clean

venv3:
	$(PY) -m venv $(VENV_DIR_TASK3)

deps3: venv3
	. $(VENV_DIR_TASK3)/bin/activate && pip install -U pip && pip install -r $(REQUIREMENTS_TASK3)

kb.index: deps3
	@mkdir -p $(ARTIFACTS)
	. $(VENV_DIR_TASK3)/bin/activate && $(PY) scripts/index/build_index.py --kb $(KB_DIR) --out $(ARTIFACTS) --model $(EMBED_MODEL)

kb.search: deps3
	@if [ -z "$(Q)" ]; then echo 'Usage: make kb.search Q="ваш вопрос"'; exit 1; fi
	. $(VENV_DIR_TASK3)/bin/activate && $(PY) scripts/index/search_index.py --query "$(Q)" --model $(EMBED_MODEL) --out $(ARTIFACTS)

kb.qc: deps3
	. $(VENV_DIR_TASK3)/bin/activate && $(PY) scripts/index/quality_check.py

kb.index.clean: deps3
	rm -f $(ARTIFACTS)/kb_faiss.index $(ARTIFACTS)/kb_meta.jsonl $(ARTIFACTS)/kb_summary.json
