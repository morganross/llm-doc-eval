# LLM Document Evaluation Tool

This tool provides a robust and flexible framework for evaluating documents using Large Language Models (LLMs). It supports two primary evaluation modes:

*   **Single-Document Evaluation**: Assess individual documents against predefined criteria.
*   **Pairwise Document Evaluation**: Compare two documents to determine which one performs better based on specific criteria.

All evaluation results are systematically persisted to a SQLite database (`results.db`) for comprehensive analysis and easy data retrieval.

## Key Features

*   **Multiple LLM Support**: Seamless integration with various LLMs, including OpenAI GPT-3.5 Turbo and Google Gemini Pro, allowing for diverse evaluation perspectives.
*   **Configurable Trial Counts**: Define the number of independent evaluation trials to ensure statistical robustness and reliability of results.
*   **Persistent Results**: All raw evaluation data and summaries are stored in a SQLite database (`results.db`), enabling historical tracking and detailed post-analysis.
*   **Command-Line Interface (CLI)**: A user-friendly command-line interface for initiating evaluations, viewing aggregated summaries, and exporting data.
*   **Custom Evaluation Criteria**: Support for defining and using custom evaluation criteria via a `criteria.yaml` file, allowing for highly tailored assessments.
*   **Google Search Grounding**: Enhance the accuracy and relevance of pairwise evaluations for Gemini models by enabling Google Search Grounding, providing real-time factual context.
*   **Detailed Single-Document Summaries**: Generate comprehensive summaries for single-document evaluations, including mean scores, standard deviations, and individual trial scores for each criterion.
*   **Comprehensive Pairwise Summaries**: Obtain in-depth summaries for pairwise evaluations, featuring overall win rates per document and clear identification of the best-performing document within a comparison set.
*   **Data Export Functionality**: Export raw and summarized evaluation data to CSV format for further analysis in external tools.
*   **Modular Project Structure**:
    *   `cli.py`: The main command-line interface entry point.
    *   `config.yaml`: Centralized configuration file for evaluation parameters.
    *   `criteria.yaml`: Defines custom evaluation criteria.
    *   `evaluator.py`: Core logic for running LLM evaluations.
    *   `metrics.py`: Handles calculation of evaluation metrics and statistical analysis.
    *   `registry.py`: Manages the registration and retrieval of LLM models.
    *   `prompts/`: Contains Jinja2 templates for LLM prompts.

## Installation

To set up the project, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/doc_eval.git
    cd doc_eval
    ```
    *(Replace `https://github.com/your-repo/doc_eval.git` with the actual repository URL)*

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys**:
    Create a `.env` file in the `doc_eval/` directory (at the same level as `config.yaml`) and add your LLM API keys. This file is used to securely load environment variables.

    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    GOOGLE_API_KEY="your_google_api_key_here"
    ```
    *Replace `"your_openai_api_key_here"` and `"your_google_api_key_here"` with your actual API keys.*

## Usage

Navigate to the `doc_eval/` directory in your terminal to use the CLI commands.

### Running Evaluations

*   **Run Single-Document Evaluations**:
    Evaluate documents in a specified folder against defined criteria.
    ```bash
    python cli.py run-single --folder-path /path/to/your/documents
    ```
    *Replace `/path/to/your/documents` with the actual path to your document folder.*

*   **Run Pairwise Evaluations**:
    Compare documents in a specified folder in a pairwise manner. The tool will generate all unique pairs (N choose 2 combinations) from the documents.
    ```bash
    python cli.py run-pairwise --folder-path /path/to/your/documents
    ```

### Viewing Summaries

*   **View Single-Document Summary**:
    Display aggregated results for single-document evaluations, including mean scores and outlier flags.
    ```bash
    python cli.py summary-single
    ```

*   **View Raw Pairwise Results**:
    Display the raw, trial-by-trial results of pairwise evaluations.
    ```bash
    python cli.py raw-pairwise
    ```

*   **View Pairwise Summary**:
    Display aggregated results for pairwise evaluations, including win rates and best-performing documents.
    ```bash
    python cli.py summary-pairwise
    ```

### Exporting Data

*   **Export Single-Document Raw Data**:
    Export all raw single-document evaluation results to a CSV file.
    ```bash
    python cli.py export-single-raw --output-path single_raw_results.csv
    ```

*   **Export Single-Document Summary Data**:
    Export the summarized single-document evaluation results to a CSV file.
    ```bash
    python cli.py export-single-summary --output-path single_summary_results.csv
    ```

*   **Export Pairwise Raw Data**:
    Export all raw pairwise evaluation results to a CSV file.
    ```bash
    python cli.py export-pairwise-raw --output-path pairwise_raw_results.csv
    ```

*   **Export Pairwise Summary Data**:
    Export the summarized pairwise evaluation results to a CSV file.
    ```bash
    python cli.py export-pairwise-summary --output-path pairwise_summary_results.csv
    ```

## Configuration (`config.yaml`)

The `config.yaml` file allows you to customize various aspects of the evaluation process:

```yaml
single_doc_eval:
  criteria_file: criteria.yaml # Path to the custom criteria file
  trial_count: 3 # Number of independent trials per model for single-doc eval
  outlier_threshold: 2.0 # Z-score threshold for flagging outliers

pairwise_eval:
  trial_count: 1 # Number of independent trials per model for pairwise eval (typically 1)
  enable_grounding: false # Set to true to enable Google Search Grounding for Gemini models

models:
  model_a:
    name: OpenAI GPT-3.5 Turbo # Name of the LLM for Model A
    provider: openai # Provider (e.g., openai, google)
  model_b:
    name: Google Gemini Pro # Name of the LLM for Model B
    provider: google # Provider (e.g., openai, google)
```

*   `trial_count`: Specifies how many times each evaluation (single or pairwise) should be run independently to ensure result consistency.
*   `enable_grounding`: A boolean flag that, when set to `true`, enables Google Search Grounding for Gemini models during pairwise evaluations. This helps ground the LLM's responses in real-world information.
*   `criteria_file`: Specifies the path to the YAML file containing custom evaluation criteria.

## Custom Criteria (`criteria.yaml`)

You can define your own evaluation criteria in `criteria.yaml`. This file allows you to specify the `name` of each criterion and its `scale` (e.g., a scale of 1 to 5).

Example `criteria.yaml`:

```yaml
- name: Clarity
  scale: 5
- name: Coherence
  scale: 5
- name: Relevance
  scale: 5
- name: Accuracy
  scale: 10
```

## Development and Contribution

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and ensure tests pass.
4.  Submit a pull request with a clear description of your changes.

## Testing

To run the unit tests:

```bash
pytest doc_eval/tests/