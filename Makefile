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

# ==== task 4 ====

VENV_DIR_TASK4 ?= .venv
REQUIREMENTS_TASK4 ?= requirements-task4-rag.txt
OLLAMA_MODEL ?= qwen2.5:7b-instruct
OLLAMA_URL   ?= http://localhost:11434/api/chat
LLM_HTTP_TIMEOUT ?= 600
LLM_MAX_TOKENS   ?= 400
LLM_NUM_CTX      ?= 2048

.PHONY: venv4 deps4

venv4:
	$(PY) -m venv $(VENV_DIR_TASK4)

deps4: venv4
	. $(VENV_DIR_TASK4)/bin/activate && pip install -U pip && pip install -r $(REQUIREMENTS_TASK4)

.PHONY: ollama.install ollama.up ollama.pull ollama.test rag.repl.ollama rag.ask.ollama

# Установка Ollama (Ubuntu/Debian)
ollama.install:
	@if command -v ollama >/dev/null 2>&1; then \
		echo "Ollama is already installed: $$(ollama -v)"; \
	else \
		echo "Installing Ollama..."; \
		curl -fsSL https://ollama.com/install.sh | sh; \
		echo "Enabling Ollama systemd service..."; \
		sudo systemctl enable --now ollama || true; \
		sleep 2; \
		systemctl --no-pager --full status ollama || true; \
	fi

# Проверка, что сервис запущен и API доступен
ollama.up:
	@if ! pgrep -f "ollama" >/dev/null 2>&1; then \
		echo "Starting Ollama service..."; \
		sudo systemctl start ollama || true; \
		sleep 2; \
	fi
	@curl -s http://localhost:11434/api/tags >/dev/null && echo "Ollama API is available" || (echo "Ollama API is not available"; exit 1)

# Загрузка выбранной модели
ollama.pull: ollama.install ollama.up
	@echo "Pulling model: $(OLLAMA_MODEL)"; \
	ollama pull $(OLLAMA_MODEL)

# Короткий функциональный тест API
ollama.test: ollama.install ollama.up
	@echo "Models:"; \
	curl -s http://localhost:11434/api/tags | sed -e 's/{"models":/models: /' || true; \
	echo "Single prompt test:"; \
	curl -s -X POST $(OLLAMA_URL) \
		-H "Content-Type: application/json" \
		-d '{"model":"$(OLLAMA_MODEL)","messages":[{"role":"user","content":"Напиши короткое предложение по-русски."}],"stream":false}' \
		| sed 's/.*"content":"\([^"]*\)".*/\1/'


# Прогрев модели: короткий запрос перед запуском REPL
.PHONY: ollama.warmup
ollama.warmup: ollama.pull
	@echo "Warming up model $(OLLAMA_MODEL)..."
	@curl -s -X POST $(OLLAMA_URL) \
		-H "Content-Type: application/json" \
		-d '{"model":"$(OLLAMA_MODEL)","messages":[{"role":"user","content":"ok?"}],"stream":false}' \
		> /dev/null || true

# Запуск RAG-бота (REPL) с локальной LLM
rag.repl.ollama: deps4 ollama.pull
	. $(VENV_DIR_TASK4)/bin/activate && \
	LLM_BACKEND=ollama OLLAMA_MODEL=$(OLLAMA_MODEL) OLLAMA_URL=$(OLLAMA_URL) \
	EMBED_MODEL=$(EMBED_MODEL) ARTIFACTS=$(ARTIFACTS) \
	$(PY) -m scripts.rag.bot --mode repl

# Разовый вопрос к RAG-боту
rag.ask.ollama: deps4 ollama.pull
	@if [ -z "$(Q)" ]; then echo 'Usage: make rag.ask.ollama Q="ваш вопрос"'; exit 1; fi
	. $(VENV_DIR_TASK4)/bin/activate && \
	LLM_BACKEND=ollama OLLAMA_MODEL=$(OLLAMA_MODEL) OLLAMA_URL=$(OLLAMA_URL) \
	EMBED_MODEL=$(EMBED_MODEL) ARTIFACTS=$(ARTIFACTS) \
	$(PY) -m scripts.rag.bot --mode ask --q "$(Q)"