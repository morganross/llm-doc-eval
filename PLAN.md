## Detailed Project Plan: Document Evaluation Tool

### 1. Project Overview

The goal is to build a Python tool that performs single-document evaluations using two LLMs (Model A and Model B), each running two independent trials per document. The tool will store raw scores, calculate key metrics (mean, standard deviation, outlier flags), and provide CLI access for running evaluations and viewing summaries. Additionally, the tool will support pairwise document evaluations to determine win rates between documents.

### 2. Technology Stack

*   **LLM orchestration, prompt templating**: LangChain
*   **In-memory table & persistence**: pandas (`to_csv`)
*   **Zero-config DB**: SQLite (for future M5, using SQLAlchemy)
*   **CLI**: Typer
*   **Test suite**: pytest
*   **Environment**: Python ≥3.10, dotenv for API keys

### 3. Key Decisions Incorporated

*   **Review Criteria (Single-Doc)**: "Clarity", "Coherence", "Relevance" (scoring scale: 1-5)
*   **Model A**: OpenAI GPT-3.5 Turbo
*   **Model B**: Google Gemini Pro
*   **Storage Format (v0.1)**: CSV (separate files for single-doc and pairwise)
*   **Outlier Flag Threshold**: |z| > 2
*   **Pairwise Pairing Logic**: All unique pairs (N choose 2 combinations) from the input folder.

### 4. Folder & Module Layout

```text
doc_eval/
│
├── loaders/            # read .txt / .md
│   └── text_loader.py
│
├── prompts/
│   └── single_doc.jinja   # criteria + document
│   └── pairwise_doc.jinja # compare two documents
│
├── models/
│   └── registry.py        # get_llm("A") / get_llm("B")
│
├── engine/
│   ├── evaluator.py       # main evaluation loop
│   └── metrics.py         # mean, std‑dev, outlier flags
│
├── cli.py                 # Typer entry point
└── config.yaml            # criteria, model names, trial count
```

### 5. Processing Pipeline (Visualized)

```mermaid
graph TD
    A[Start] --> B{Load Documents};
    B --> C{For each Document};
    C --> D{Prepare Single-Doc Prompt};
    D --> E{For Model A (2 trials)};
    E --> F{Call LLM};
    F --> G{Validate JSON Response};
    G --> H{Record Single-Doc Row};
    E --> I{For Model B (2 trials)};
    I --> F;
    H --> J{Append to Single-Doc DataFrame};
    J --> K{Persist Single-Doc to CSV};
    K --> L{End Single-Doc Loop};

    L --> M{Generate Document Pairs};
    M --> N{For each Document Pair};
    N --> O{Prepare Pairwise Prompt};
    O --> P{For Model A (1 trial)};
    P --> Q{Call LLM};
    Q --> R{Validate JSON Response};
    R --> S{Record Pairwise Row};
    P --> T{For Model B (1 trial)};
    T --> Q;
    S --> U{Append to Pairwise DataFrame};
    U --> V{Persist Pairwise to CSV};
    V --> W{End Pairwise Loop};

    W --> X{Calculate Metrics};
    X --> Y{CLI Summary};
    Y --> Z[End];

    subgraph Single-Doc Evaluation Loop
        D --> E;
        E --> F;
        F --> G;
        G --> H;
        I --> F;
    end

    subgraph Pairwise Evaluation Loop
        O --> P;
        P --> Q;
        Q --> R;
        R --> S;
        T --> Q;
    end
```

### 6. Detailed Development Tasks & Milestones

#### **M1: Repo Scaffold, Loader, Config, CLI Skeleton (ETA: 0.5 day)**

*   **Task 1.1**: Project bootstrap
    *   Initialize `doc_eval/` directory.
    *   Set up `pyproject.toml` (Poetry) or `requirements.txt` (pip-env) with initial dependencies: `langchain`, `pandas`, `typer`, `python-dotenv`, `pyyaml`, `jinja2`.
    *   Create `.env` for API keys.
*   **Task 1.2**: Implement `loaders/text_loader.py`
    *   Function to scan a target folder for `.txt` and `.md` files.
    *   Function to read file content and yield `(doc_id, text)`.
*   **Task 1.3**: Create `config.yaml`
    *   Define `criteria` (Clarity, Coherence, Relevance) with scoring scale (1-5) for single-doc evaluations.
    *   Define `models` (Model A: OpenAI GPT-3.5 Turbo, Model B: Google Gemini Pro).
    *   Define `trial_count` (2) for single-doc evaluations.
    *   Define `outlier_threshold` (2.0).
    *   **NEW**: Add configuration for pairwise evaluations (e.g., `pairwise_trial_count`, `pairwise_criteria` if applicable).
*   **Task 1.4**: Set up `cli.py` skeleton
    *   Initialize Typer app.
    *   Create placeholder commands: `run` and `summary`.

#### **M2: Prompt Template + Model Registry (Mocked) (ETA: 0.5 day)**

*   **Task 2.1**: Create `prompts/single_doc.jinja`
    *   Template for prompt handling, including `{criteria}` and `{document}` placeholders.
    *   Include "Return JSON exactly like..." instruction for structured output.
*   **Task 2.2**: Build `models/registry.py`
    *   `get_llm(model_name)` function to return LangChain LLM instances.
    *   Configure LLMs with `temperature=0.0` (or low value like 0.3).
    *   Initially, mock LLM responses for testing purposes.
    *   Integrate actual OpenAI and Google LLM clients.
*   **Task 2.3 (NEW):** Create `prompts/pairwise_doc.jinja`
    *   Template for comparing two documents (`{document_1}`, `{document_2}`) and determining a winner, including instructions for structured output (e.g., JSON with `winner_doc_id`, `reason`).

#### **M3: Evaluation Loop Writing to CSV; Unit Tests (ETA: 1 day)**

*   **Task 3.1**: Develop `engine/evaluator.py` for **single-document evaluations**
    *   Main evaluation loop:
        *   Iterate through loaded documents.
        *   For each document, iterate through models (A, B) and trials (2 per model).
        *   Render prompt using `single_doc.jinja`.
        *   Call LLM via LangChain `LLMChain`.
        *   Implement JSON validation for LLM responses.
        *   Handle retries for rate limits and JSON parse errors (basic implementation).
        *   Append results to a pandas DataFrame: `doc_id, model, trial, criterion, score, timestamp`.
    *   Implement persistence: Write DataFrame to `single_doc_results.csv` after every batch (e.g., after each document or every N documents) for safety.
*   **Task 3.2**: Implement initial unit tests (`pytest`)
    *   Test `text_loader.py` functions.
    *   Test `registry.py` LLM instantiation.
    *   Test JSON parsing and validation in `evaluator.py`.
*   **Task 3.3 (NEW):** Extend `engine/evaluator.py` to include a separate function/class for **pairwise evaluations**:
    *   Generate pairs of documents from the input folder (all unique pairs, N choose 2 combinations).
    *   For each pair, iterate through models (A, B) and trials (e.g., 1 trial per model for pairwise).
    *   Render prompt using `pairwise_doc.jinja`.
    *   Call LLM via LangChain `LLMChain`.
    *   Implement JSON validation for LLM responses (e.g., ensuring `winner_doc_id` is one of the two input `doc_id`s).
    *   Handle retries for rate limits and JSON parse errors.
    *   Append results to a separate pandas DataFrame: `doc_id_1, doc_id_2, model, winner_doc_id, reason, timestamp`.
    *   Implement persistence: Write DataFrame to `pairwise_results.csv` after every batch for safety.

#### **M4: Metrics + `summary` Command (ETA: 0.5 day)**

*   **Task 4.1**: Develop `engine/metrics.py` for **single-document metrics**
    *   Functions for `groupby` queries on the `single_doc_results` DataFrame:
        *   Calculate mean score per document per criterion.
        *   Calculate standard deviation per document per criterion.
        *   Calculate Z-score for each score.
        *   Implement outlier flagging based on `|z| > 2`.
*   **Task 4.2**: Enhance `cli.py` for **single-document evaluations**
    *   Implement `doc-eval run-single` command to trigger the single-document evaluation pipeline.
    *   Implement `doc-eval summary-single` command to:
        *   Load results from `single_doc_results.csv`.
        *   Call `metrics.py` functions.
        *   Print top documents with mean scores and outlier flags.
*   **Task 4.3 (NEW):** Extend `engine/metrics.py` to include functions for **pairwise win rate calculations**:
    *   Load results from `pairwise_results.csv`.
    *   Calculate win rates for each model (e.g., how many times Model A picked Document X over Document Y).
*   **Task 4.4 (NEW):** Enhance `cli.py` to include commands for **pairwise evaluations**:
    *   Implement `doc-eval run-pairwise` command to trigger the pairwise evaluation pipeline.
    *   Implement `doc-eval summary-pairwise` command to:
        *   Load results from `pairwise_results.csv`.
        *   Call `metrics.py` functions for win rates.
        *   Print summary of pairwise win rates.

#### **M5: SQLite Persistence, Async Calls, Retries (ETA: 1 day)**

*   **Task 5.1**: Refactor persistence to SQLite for **single-document results**
    *   Integrate SQLAlchemy for SQLite database interaction.
    *   Modify `evaluator.py` to write single-doc results to a `single_doc_results` table in SQLite.
    *   Ensure schema creation if table doesn't exist.
*   **Task 5.2 (NEW):** Refactor persistence to SQLite for **pairwise results**
    *   Modify `evaluator.py` to write pairwise results to a `pairwise_results` table in SQLite.
    *   Ensure schema creation if table doesn't exist.
*   **Task 5.3**: Implement robust logging and retry logic
    *   Use Python's `logging` module.
    *   Implement exponential backoff for API rate limits.
    *   Improve JSON parsing error handling and retries.
*   **Task 5.4**: Consider async calls (stretch goal for M5 if time permits)
    *   Explore `asyncio` and `httpx` for parallel LLM calls to speed up evaluation.

#### **M6: Docs & GitHub Actions CI (ETA: 0.5 day)**

*   **Task 6.1**: Create comprehensive `README.md`
    *   Project description.
    *   Setup instructions (environment, dependencies, API keys).
    *   Usage examples for `doc-eval run-single`, `doc-eval summary-single`, `doc-eval run-pairwise`, and `doc-eval summary-pairwise`.
    *   Description of `config.yaml` parameters.
*   **Task 6.2**: Set up GitHub Actions CI
    *   Workflow to run tests on push/pull request.
    *   Linting (e.g., `flake8` or `black`).