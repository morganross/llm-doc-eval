# Document Evaluation Tool

This tool provides a flexible framework for evaluating documents using Large Language Models (LLMs). It supports both single-document evaluations (scoring documents against predefined criteria) and pairwise evaluations (comparing two documents to determine a winner). All evaluation results are persisted to a SQLite database for easy analysis.

## Features

*   **Single-Document Evaluation**: Score documents based on criteria like Clarity, Coherence, and Relevance.
*   **Pairwise Evaluation**: Compare pairs of documents to determine a superior document.
*   **LLM Integration**: Leverages LangChain for seamless interaction with various LLMs (e.g., OpenAI GPT, Google Gemini).
*   **Data Persistence**: Stores all raw evaluation results in a SQLite database.
*   **Metrics & Summaries**: Calculates mean scores, standard deviations, outlier flags, and pairwise win rates.
*   **Command-Line Interface (CLI)**: Easy-to-use commands for running evaluations and viewing summaries.

## Technology Stack

*   **LLM orchestration, prompt templating**: LangChain
*   **In-memory table & persistence**: pandas
*   **Zero-config DB**: SQLite (via SQLAlchemy)
*   **CLI**: Typer
*   **Test suite**: pytest
*   **Environment**: Python ≥3.10, dotenv for API keys
*   **Prompt Templating**: Jinja2
*   **Configuration**: YAML

## Folder & Module Layout

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
├── config.yaml            # criteria, model names, trial count
└── README.md              # This file
```

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd doc_eval
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    Create a `.env` file in the `doc_eval/` directory (at the same level as `config.yaml`) and add your LLM API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    GOOGLE_API_KEY="your_google_api_key_here"
    ```
    *Replace `"your_openai_api_key_here"` and `"your_google_api_key_here"` with your actual API keys.*

## Configuration (`config.yaml`)

The `config.yaml` file defines the parameters for your evaluations:

```yaml
single_doc_eval:
  criteria:
    - name: Clarity
      scale: 5
    - name: Coherence
      scale: 5
    - name: Relevance
      scale: 5
  trial_count: 2 # Number of independent trials per model for single-doc eval
  outlier_threshold: 2.0 # Z-score threshold for flagging outliers

pairwise_eval:
  trial_count: 1 # Number of independent trials per model for pairwise eval (typically 1)

models:
  model_a:
    name: OpenAI GPT-3.5 Turbo # Name of the LLM for Model A
    provider: openai # Provider (e.g., openai, google)
  model_b:
    name: Google Gemini Pro # Name of the LLM for Model B
    provider: google # Provider (e.g., openai, google)
```

## Usage

Navigate to the `doc_eval/` directory in your terminal.

### 1. Run Single-Document Evaluations

To run evaluations on a folder containing `.txt` or `.md` documents:

```bash
python cli.py run-single --folder-path /path/to/your/documents
```
*Replace `/path/to/your/documents` with the actual path to your document folder.*

### 2. View Single-Document Summary

To see the aggregated results for single-document evaluations:

```bash
python cli.py summary-single
```

### 3. Run Pairwise Evaluations

To run pairwise comparisons on a folder containing `.txt` or `.md` documents:

```bash
python cli.py run-pairwise --folder-path /path/to/your/documents
```
*The tool will generate all unique pairs (N choose 2 combinations) from the documents in the specified folder.*

### 4. View Pairwise Summary

To see the aggregated results for pairwise evaluations:

```bash
python cli.py summary-pairwise
```

## Testing

To run the unit tests:

```bash
pytest doc_eval/tests/
```

## Future Enhancements

*   Asynchronous LLM calls for faster evaluations.
*   More sophisticated retry mechanisms and error handling.
*   Support for additional document types (e.g., PDF, DOCX).
*   Web-based UI for visualization and interaction.
*   Integration with CI/CD pipelines for automated testing.