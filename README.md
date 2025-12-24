# AI_DMP_RAG — NIH Data Management Plan (DMP) Generator

This project generates an NIH-style **Data Management & Sharing Plan (DMP)** using a pipeline that can ingest documents, build/search an index, and produce a final draft through a **FastAPI** web UI.

---

## Repository Structure

```text
AI_DMP_RAG/
│── app.py
│── README.md
│── requirements.txt
│── setup.py
│── .env
│── .gitignore
│
├── config/                 # Configuration files (YAML, etc.)
├── data/                   # Input documents / datasets (raw or processed)
├── model/                  # Saved models / embeddings / checkpoints (if any)
├── logs/                   # Runtime logs (app + pipeline)
├── notebook_DMP_RAG/       # Experiments, notebooks, prototypes
│
├── src/                    # Main application code
│   ├── __init__.py
│   ├── core_pipeline_UI.py # Pipeline logic used by the UI/app
│   └── data_ingestion.py   # Document ingestion + indexing utilities
│
├── prompt/                 # Prompt templates and prompt tools
│   ├── __init__.py
│   └── prompt_library.py
│
├── logger/                 # Custom logger utilities
│   ├── __init__.py
│   └── custom_logger.py
│
├── exception/              # Custom exceptions
│   ├── __init__.py
│   └── custom_exception.py
│
├── utils/                  # Shared helper functions (general utilities)
│
├── build/                  # Build artifacts (packaging)
├── dist/                   # Distribution artifacts (packaging)
└── DMP_RAG.egg-info/       # Package metadata (created during install/build)
## Project Setup & Usage (All-in-One)

**Prerequisites:** Python **3.10+** (recommended), (optional) Git, and a `.env` file for secrets (API keys, endpoints, etc.).

**Setup (Local Development):** Create and activate a virtual environment.

- **Windows (PowerShell):**
```python
python -m venv venv
.\venv\Scripts\Activate.ps1
```

- **macOS/Linux:**
```python
python -m venv venv
source venv/bin/activate
```

Install dependencies:
```python
pip install -r requirements.txt
```

(Optional but recommended for development):
```python
pip install -e .
```

**Environment Variables (`.env`):** Create a `.env` file in the project root (same level as `app.py`). Example:
```python
# LLM Provider / API
OPENAI_API_KEY=your_key_here

# Optional settings
ENV=dev
LOG_LEVEL=INFO
```
Keep `.env` out of Git (it should be in `.gitignore`).

**Configuration (`config/`):** The pipeline typically reads settings from `config/` (e.g., `config/config.yaml`). Common items include: data paths, index/vectorstore settings, embedding model settings, LLM model settings, chunking parameters, and a rebuild-index flag.

**Data Ingestion & Indexing:** Key modules are `src/data_ingestion.py` (loads documents, cleans/chunks them, builds an index/vector store) and `src/core_pipeline_UI.py` (orchestrates retrieval + prompting + generation). Typical workflow: put reference documents into `data/`, run the pipeline once (or with a rebuild flag) to build the index, then run the app and generate DMPs through the UI. If rebuilding is needed, use a config flag like `force_rebuild_index=True` (or set it in YAML) or delete the old index folder (e.g., `data/index/`).

**Run the Web App (FastAPI):** From the project root (where `app.py` is):
```python
uvicorn app:app --reload
```
Open in your browser: `http://127.0.0.1:8000/` (and if enabled: `http://127.0.0.1:8000/docs`).

**Logging:** `logger/custom_logger.py` centralizes logging format/handlers; runtime logs are typically written to `logs/`. If logs aren’t showing, check `LOG_LEVEL` in `.env` and ensure `logs/` exists and is writable.

**Prompts:** Prompt templates/utilities are in `prompt/prompt_library.py`. You can update the DMP template text, add section-by-section prompts, and enforce NIH structure/format rules.

**Common Troubleshooting:** If the page doesn’t open, copy the printed Uvicorn URL (e.g., `http://127.0.0.1:8000`) into your browser. For import errors, run `uvicorn` from the project root, confirm `src/__init__.py` exists, and try:
```python
pip install -e .
```
If retrieval is empty, confirm docs exist under `data/` and rebuild the index via config/ingestion. For log permission issues, ensure `logs/` exists and run terminal as admin (Windows) if needed.

**Recommended `.gitignore`:** Ignore `venv/`, `__pycache__/`, `.env`, `build/`, `dist/`, `*.egg-info/`, and optionally `logs/` (depending on whether you want to commit logs or not).

**Example Commands (Conda):**
```python
git clone https://github.com/fairdataihub/dmpchef.git
cd dmpchef
code .

conda create -n dmpchef python=3.10 -y
conda activate dmpchef

python -m pip install --upgrade pip
pip install -r requirements.txt

python setup.py install
# or (recommended for development)
pip install -e .

uvicorn app:app --reload
```
Then open: `http://127.0.0.1:8000/` and test the app.
