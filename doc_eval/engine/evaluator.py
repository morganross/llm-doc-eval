import os
import json
import yaml
import pandas as pd
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from itertools import combinations
from sqlalchemy import create_engine
import logging
import time # For exponential backoff

from doc_eval.models.registry import get_llm
import google.genai # Import the new Google GenAI SDK
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch # Import types for grounding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Evaluator:
    def __init__(self, config_path='doc_eval/config.yaml', prompts_dir='doc_eval/prompts', db_path='doc_eval/results.db'):
        self.config = self._load_config(config_path)
        self.env = Environment(loader=FileSystemLoader(prompts_dir))
        self.single_doc_template = self.env.get_template('single_doc.jinja')
        self.pairwise_doc_template = self.env.get_template('pairwise_doc.jinja')
        self.single_doc_results = pd.DataFrame()
        self.pairwise_results = pd.DataFrame()
        self.db_engine = create_engine(f'sqlite:///{db_path}')
        self.genai_client = google.genai.Client() # Initialize Google GenAI client
        logging.info(f"Evaluator initialized with config from {config_path}, prompts from {prompts_dir}, and DB at {db_path}")

    def _load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load criteria from a separate file if specified
        if 'single_doc_eval' in config and 'criteria_file' in config['single_doc_eval']:
            criteria_file_path = os.path.join(os.path.dirname(config_path), config['single_doc_eval']['criteria_file'])
            if os.path.exists(criteria_file_path):
                with open(criteria_file_path, 'r', encoding='utf-8') as f:
                    criteria_data = yaml.safe_load(f)
                # Assuming criteria are under 'single_doc_criteria' key in criteria.yaml
                config['single_doc_eval']['criteria'] = criteria_data.get('single_doc_criteria', [])
            else:
                logging.warning(f"Criteria file not found: {criteria_file_path}. Single-document criteria will be empty.")
                config['single_doc_eval']['criteria'] = []
        
        # Load criteria for pairwise evaluation if specified
        if 'pairwise_eval' in config and 'criteria_file' in config['pairwise_eval']:
            criteria_file_path = os.path.join(os.path.dirname(config_path), config['pairwise_eval']['criteria_file'])
            if os.path.exists(criteria_file_path):
                with open(criteria_file_path, 'r', encoding='utf-8') as f:
                    criteria_data = yaml.safe_load(f)
                # Assuming criteria are under 'single_doc_criteria' key in criteria.yaml (reusing for simplicity)
                config['pairwise_eval']['criteria'] = criteria_data.get('single_doc_criteria', [])
            else:
                logging.warning(f"Criteria file not found: {criteria_file_path}. Pairwise criteria will be empty.")
                config['pairwise_eval']['criteria'] = []
        
        return config

    def _get_llm_and_parser(self, model_name: str):
        """Returns the LLM instance, a JSON parser, and grounding status."""
        model_config = None
        for model_key in ['model_a', 'model_b']:
            if self.config['models'][model_key]['name'] == model_name:
                model_config = self.config['models'][model_key]
                break
        
        if model_config is None:
            raise ValueError(f"Model configuration not found for {model_name}")

        enable_grounding = model_config.get('enable_grounding', False)
        
        llm = get_llm(model_name, temperature=0.0, enable_grounding=enable_grounding)
        parser = JsonOutputParser()
        return llm, parser, model_name, enable_grounding # Return model_name and enable_grounding

    def _run_llm_call(self, llm, parser, prompt_text, model_name, enable_grounding, retries=3):
        for attempt in range(1, retries + 1):
            try:
                response = None
                if model_name.startswith("gemini-") and enable_grounding:
                    # Use direct google.genai client for grounding
                    google_search_tool = Tool(google_search=GoogleSearch())
                    
                    # LangChain's prompt_text is a string, need to wrap it for genai client
                    contents = [{"parts": [{"text": prompt_text}]}]

                    genai_response = self.genai_client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=GenerateContentConfig(
                            tools=[google_search_tool],
                            response_modalities=["TEXT"],
                        )
                    )
                    # Extract content and grounding metadata from genai_response
                    response_content = ""
                    grounding_metadata = None
                    if genai_response.candidates and genai_response.candidates[0].content.parts:
                        for part in genai_response.candidates[0].content.parts:
                            response_content += part.text
                        grounding_metadata = genai_response.candidates[0].grounding_metadata
                    
                    # Simulate LangChain's BaseMessage structure for consistency with parser
                    class MockAIMessage:
                        def __init__(self, content, additional_kwargs=None):
                            self.content = content
                            self.additional_kwargs = additional_kwargs if additional_kwargs is not None else {}
                    
                    response = MockAIMessage(content=response_content, additional_kwargs={"grounding_metadata": grounding_metadata})
                    
                else:
                    # Use LangChain's LLM for other models or no grounding
                    response = llm.invoke(prompt_text)
                
                # Extract grounding metadata if available (from either path)
                grounding_metadata = response.additional_kwargs.get("grounding_metadata")
                if grounding_metadata:
                    logging.info(f"  Grounding metadata received: {grounding_metadata}")
                    # You might want to store this metadata or process it further
                    # For now, just logging it.

                # LangChain's parser expects a string, then parses it.
                # If the LLM directly returns a dict (e.g., from a mock), handle it.
                if isinstance(response, dict):
                    parsed_response = response
                else:
                    parsed_response = parser.parse(response.content) # Assuming response is a BaseMessage or similar

                # Basic validation for JSON structure
                if isinstance(parsed_response, dict):
                    logging.info(f"LLM call successful on attempt {attempt}.")
                    return parsed_response
                else:
                    raise ValueError("Parsed LLM response is not a valid JSON object.")
            except json.JSONDecodeError as e:
                logging.warning(f"JSON decode error on attempt {attempt}: {e}. Retrying...")
            except Exception as e:
                logging.warning(f"LLM call error on attempt {attempt}: {e}. Retrying...")
            
            if attempt < retries:
                sleep_time = 2 ** attempt # Exponential backoff
                logging.info(f"Waiting {sleep_time} seconds before retry...")
                time.sleep(sleep_time)
        logging.error(f"Failed LLM call after {retries} attempts.")
        raise Exception(f"Failed LLM call after {retries} attempts.")

    def evaluate_single_document(self, doc_id, document_content):
        model_a_name = self.config['models']['model_a']['name']
        model_b_name = self.config['models']['model_b']['name']
        criteria = self.config['single_doc_eval']['criteria'] # Now loaded from criteria.yaml
        trial_count = self.config['single_doc_eval']['trial_count']

        logging.info(f"Starting single-document evaluation for doc_id: {doc_id}")
        for model_name in [model_a_name, model_b_name]:
            llm, parser, model_name_from_get_llm, enable_grounding = self._get_llm_and_parser(model_name)
            for trial in range(1, trial_count + 1):
                logging.info(f"  Evaluating with {model_name}, trial {trial}")
                input_data = {
                    "criteria": criteria,
                    "document": document_content
                }
                rendered_prompt = self.single_doc_template.render(input_data)
                
                try:
                    llm_response = self._run_llm_call(llm, parser, rendered_prompt, model_name_from_get_llm, enable_grounding)
                    for eval_item in llm_response.get('evaluations', []):
                        row = {
                            "doc_id": doc_id,
                            "model": model_name,
                            "trial": trial,
                            "criterion": eval_item.get('criterion'),
                            "score": eval_item.get('score'),
                            "reason": eval_item.get('reason'),
                            "timestamp": datetime.now()
                        }
                        self.single_doc_results = pd.concat([self.single_doc_results, pd.DataFrame([row])], ignore_index=True)
                    logging.info(f"  Successfully processed LLM response for {model_name}, trial {trial}.")
                except Exception as e:
                    logging.error(f"Error evaluating single document {doc_id} with {model_name} trial {trial}: {e}")

        self._persist_single_doc_results()
        logging.info(f"Finished single-document evaluation for doc_id: {doc_id}")

    def evaluate_pairwise_documents(self, doc_ids_contents):
        """
        Evaluates all unique pairs of documents from the provided list.
        doc_ids_contents: list of (doc_id, content) tuples
        """
        model_a_name = self.config['models']['model_a']['name']
        model_b_name = self.config['models']['model_b']['name']
        pairwise_trial_count = self.config['pairwise_eval']['trial_count'] # Typically 1 for pairwise
        pairwise_criteria = self.config['pairwise_eval']['criteria'] # Load pairwise criteria

        document_map = {doc_id: content for doc_id, content in doc_ids_contents}
        all_doc_ids = [doc_id for doc_id, _ in doc_ids_contents]

        logging.info("Starting pairwise document evaluation.")
        for doc_id_1, doc_id_2 in combinations(all_doc_ids, 2):
            logging.info(f"  Evaluating pair: {doc_id_1} vs {doc_id_2}")
            for model_name in [model_a_name, model_b_name]:
                llm, parser, model_name_from_get_llm, enable_grounding = self._get_llm_and_parser(model_name)
                for trial in range(1, pairwise_trial_count + 1):
                    logging.info(f"    Evaluating with {model_name}, trial {trial}")
                    input_data = {
                        "doc_id_1": doc_id_1,
                        "document_1": document_map[doc_id_1],
                        "doc_id_2": doc_id_2,
                        "document_2": document_map[doc_id_2],
                        "criteria": pairwise_criteria # Pass pairwise criteria to the template
                    }
                    rendered_prompt = self.pairwise_doc_template.render(input_data)
                    
                    try:
                        llm_response = self._run_llm_call(llm, parser, rendered_prompt, model_name_from_get_llm, enable_grounding)
                        winner_doc_id = llm_response.get('winner_doc_id')
                        reason = llm_response.get('reason')

                        # Basic validation for winner_doc_id
                        if winner_doc_id not in [doc_id_1, doc_id_2]:
                            raise ValueError(f"Invalid winner_doc_id: {winner_doc_id}. Must be {doc_id_1} or {doc_id_2}.")

                        row = {
                            "doc_id_1": doc_id_1,
                            "doc_id_2": doc_id_2,
                            "model": model_name,
                            "trial": trial,
                            "winner_doc_id": winner_doc_id,
                            "reason": reason,
                            "timestamp": datetime.now()
                        }
                        self.pairwise_results = pd.concat([self.pairwise_results, pd.DataFrame([row])], ignore_index=True)
                        logging.info(f"    Successfully processed LLM response for {model_name}, trial {trial}.")
                    except Exception as e:
                        logging.error(f"Error evaluating pairwise documents {doc_id_1} vs {doc_id_2} with {model_name} trial {trial}: {e}")

        self._persist_pairwise_results()
        logging.info("Finished pairwise document evaluation.")

    def _persist_single_doc_results(self, table_name="single_doc_results"):
        if not self.single_doc_results.empty:
            try:
                self.single_doc_results.to_sql(table_name, self.db_engine, if_exists='append', index=False)
                logging.info(f"Single-document results persisted to SQLite table: {table_name}")
                self.single_doc_results = pd.DataFrame() # Clear DataFrame after persisting
            except Exception as e:
                logging.error(f"Error persisting single-document results to SQLite: {e}")
        else:
            logging.info("No single-document results to persist.")

    def _persist_pairwise_results(self, table_name="pairwise_results"):
        if not self.pairwise_results.empty:
            try:
                self.pairwise_results.to_sql(table_name, self.db_engine, if_exists='append', index=False)
                logging.info(f"Pairwise results persisted to SQLite table: {table_name}")
                self.pairwise_results = pd.DataFrame() # Clear DataFrame after persisting
            except Exception as e:
                logging.error(f"Error persisting pairwise results to SQLite: {e}")
        else:
            logging.info("No pairwise results to persist.")

if __name__ == '__main__':
    # This block is for testing the Evaluator class directly.
    # It requires dummy config, prompts, and potentially mocked LLMs or actual API keys.

    # Ensure doc_eval directory exists for testing
    if not os.path.exists('doc_eval'):
        os.makedirs('doc_eval')

    # Create dummy config.yaml if it doesn't exist for testing
    config_path = 'doc_eval/config.yaml'
    if not os.path.exists(config_path):
        dummy_config_content = """
single_doc_eval:
  criteria_file: criteria.yaml
  trial_count: 2
  outlier_threshold: 2.0

pairwise_eval:
  trial_count: 1

models:
  model_a:
    name: o4-mini-2025-04-16
    provider: openai
  model_b:
    name: gemini-2.5-flash-preview-05-20
    provider: google
"""
        with open(config_path, 'w') as f:
            f.write(dummy_config_content)
    
    # Create dummy criteria.yaml if it doesn't exist for testing
    criteria_path = 'doc_eval/criteria.yaml'
    if not os.path.exists(criteria_path):
        dummy_criteria_content = """
single_doc_criteria:
  - name: Clarity
    scale: 5
  - name: Coherence
    scale: 5
  - name: Relevance
    scale: 5
"""
        with open(criteria_path, 'w') as f:
            f.write(dummy_criteria_content)

    # Create dummy prompts if they don't exist for testing
    prompts_dir = 'doc_eval/prompts'
    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir)
    if not os.path.exists(os.path.join(prompts_dir, 'single_doc.jinja')):
        with open(os.path.join(prompts_dir, 'single_doc.jinja'), 'w') as f:
            f.write("""
You are an expert document reviewer. Your task is to evaluate the provided document based on the following criteria and assign a score from 1 to {{ criteria[0].scale }} for each.

**Criteria:**
{% for criterion in criteria %}
- {{ criterion.name }} (Score: 1-{{ criterion.scale }})
{% endfor %}

**Document to Evaluate:**
```
{{ document }}
```

Return your evaluation as a JSON object with the following structure:
```json
{
  "evaluations": [
    {% for criterion in criteria %}
    {
      "criterion": "{{ criterion.name }}",
      "score": <score_value_1_to_{{ criterion.scale }}>,
      "reason": "<brief_reason_for_score>"
    }{% if not loop.last %},{% endif %}
    {% endfor %}
  ]
}
""")
    if not os.path.exists(os.path.join(prompts_dir, 'pairwise_doc.jinja')):
        with open(os.path.join(prompts_dir, 'pairwise_doc.jinja'), 'w') as f:
            f.write("""
You are an expert document comparison AI. Your task is to compare two documents and determine which one is superior based on overall quality, clarity, and relevance.

**Document 1 (ID: {{ doc_id_1 }}):**
```
{{ document_1 }}
```

**Document 2 (ID: {{ doc_id_2 }}):**
```
{{ document_2 }}
```

Return your decision as a JSON object with the following structure:
```json
{
  "winner_doc_id": "<ID_of_the_superior_document>",
  "reason": "<brief_reason_for_the_decision>"
}
""")

    # Define a temporary database path for testing
    test_db_path = 'doc_eval/test_results.db'
    if os.path.exists(test_db_path):
        os.remove(test_db_path) # Clean up previous test db

    evaluator = Evaluator(db_path=test_db_path)

    # Test single document evaluation
    logging.info("\n--- Testing Single Document Evaluation ---")
    evaluator.evaluate_single_document("test_doc_1", "This is a test document content.")
    # Results are persisted immediately, so DataFrame should be empty after persist
    logging.info(f"Single-doc DataFrame empty after persist: {evaluator.single_doc_results.empty}")

    # Test pairwise document evaluation
    logging.info("\n--- Testing Pairwise Document Evaluation ---")
    test_docs_contents = [
        ("docA", "Content of document A."),
        ("docB", "Content of document B."),
        ("docC", "Content of document C.")
    ]
    evaluator.evaluate_pairwise_documents(test_docs_contents)
    # Results are persisted immediately, so DataFrame should be empty after persist
    logging.info(f"Pairwise DataFrame empty after persist: {evaluator.pairwise_results.empty}")

    # Verify data in SQLite (optional, for manual check)
    # import sqlite3
    # conn = sqlite3.connect(test_db_path)
    # single_df_from_db = pd.read_sql_table("single_doc_results", conn)
    # pairwise_df_from_db = pd.read_sql_table("pairwise_results", conn)
    # logging.info("\nSingle-doc results from DB:")
    # logging.info(single_df_from_db)
    # logging.info("\nPairwise results from DB:")
    # logging.info(pairwise_df_from_db)
    # conn.close()

    # Clean up dummy files and folder if created by this script
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            # Check if it's the dummy one created by this script (by checking for criteria_file)
            if "criteria_file" in f.read():
                os.remove(config_path)
    if os.path.exists(criteria_path):
        os.remove(criteria_path)
    if os.path.exists(os.path.join(prompts_dir, 'single_doc.jinja')):
        os.remove(os.path.join(prompts_dir, 'single_doc.jinja'))
    if os.path.exists(os.path.join(prompts_dir, 'pairwise_doc.jinja')):
        os.remove(os.path.join(prompts_dir, 'pairwise_doc.jinja'))
    if os.path.exists(prompts_dir) and not os.listdir(prompts_dir):
        os.rmdir(prompts_dir)
    if os.path.exists(test_db_path):
        os.remove(test_db_path)