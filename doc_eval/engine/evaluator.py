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
import asyncio # Import asyncio

from doc_eval.models.registry import get_llm
import google.genai # Import the new Google GenAI SDK
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch # Import types for grounding

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)

class Evaluator:
    def __init__(self, config_path='doc_eval/config.yaml', prompts_dir='doc_eval/prompts', db_path='doc_eval/results.db', max_concurrent_llm_calls: int = 5):
        self.config = self._load_config(config_path)
        self.env = Environment(loader=FileSystemLoader(prompts_dir))
        self.single_doc_template = self.env.get_template('single_doc.jinja')
        self.pairwise_doc_template = self.env.get_template('pairwise_doc.jinja')
        self.single_doc_results = pd.DataFrame()
        self.pairwise_results = pd.DataFrame()
        self.db_engine = create_engine(f'sqlite:///{db_path}')
        self.genai_client = google.genai.Client() # Initialize Google GenAI client
        self.llm_semaphore = asyncio.Semaphore(max_concurrent_llm_calls) # Initialize semaphore
        logging.info(f"Evaluator initialized with config from {config_path}, prompts from {prompts_dir}, and DB at {db_path} with max_concurrent_llm_calls={max_concurrent_llm_calls}")

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
                # Assuming criteria are under 'pairwise_doc_criteria' key in criteria.yaml
                config['pairwise_eval']['criteria'] = criteria_data.get('pairwise_doc_criteria', [])
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

    async def _run_llm_call(self, llm, parser, prompt_text, model_name, enable_grounding, retries=3):
        logging.debug(f"Attempting to acquire semaphore for {model_name}...")
        async with self.llm_semaphore: # Acquire semaphore before making LLM call
            logging.debug(f"Sending prompt to {model_name}:\n{prompt_text[:500]}...") # Log first 500 chars of prompt
            for attempt in range(1, retries + 1):
                logging.debug(f"Semaphore acquired for {model_name}. Starting LLM call (attempt {attempt}/{retries})...")
                try:
                    response = None
                    if model_name.startswith("gemini-") and enable_grounding:
                        # Use direct google.genai client for grounding
                        google_search_tool = Tool(google_search=GoogleSearch())
                        
                        # LangChain's prompt_text is a string, need to wrap it for genai client
                        contents = [{"parts": [{"text": prompt_text}]}]

                        genai_response = await self.genai_client.models.generate_content( # Await here
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
                        response = await llm.ainvoke(prompt_text) # Await here
                    
                    logging.debug(f"Received raw response from {model_name}:\n{response.content[:500]}...") # Log first 500 chars of response

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
                        logging.debug(f"LLM call completed for {model_name}.")
                        return parsed_response
                    else:
                        raise ValueError("Parsed LLM response is not a valid JSON object.")
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON decode error on attempt {attempt}: {e}. Retrying...")
                except Exception as e:
                    logging.warning(f"LLM call error on attempt {attempt}: {e}. Retrying...")
                finally:
                    logging.debug(f"Semaphore released for {model_name}.") # Log semaphore release

                if attempt < retries:
                    sleep_time = 2 ** attempt # Exponential backoff
                    logging.info(f"Waiting {sleep_time} seconds before retry...")
                    await asyncio.sleep(sleep_time) # Use asyncio.sleep for async context
            logging.error(f"Failed LLM call after {retries} attempts.")
            raise Exception(f"Failed LLM call after {retries} attempts.")

    async def evaluate_single_document(self, doc_id, document_content):
        model_a_name = self.config['models']['model_a']['name']
        model_b_name = self.config['models']['model_b']['name']
        criteria = self.config['single_doc_eval']['criteria'] # Now loaded from criteria.yaml
        trial_count = self.config['single_doc_eval']['trial_count']

        logging.info(f"Starting single-document evaluation for doc_id: {doc_id}")
        
        tasks = []
        for model_name in [model_a_name, model_b_name]:
            llm, parser, model_name_from_get_llm, enable_grounding = self._get_llm_and_parser(model_name)
            for trial in range(1, trial_count + 1):
                logging.info(f"  Creating task for {model_name}, trial {trial}")
                input_data = {
                    "criteria": criteria,
                    "document": document_content
                }
                rendered_prompt = self.single_doc_template.render(input_data)
                
                tasks.append(self._run_llm_call(llm, parser, rendered_prompt, model_name_from_get_llm, enable_grounding))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True) # Run tasks concurrently

        for i, response in enumerate(responses):
            model_idx = i // trial_count
            trial_idx = i % trial_count
            model_name = [model_a_name, model_b_name][model_idx]
            trial = trial_idx + 1

            if isinstance(response, Exception):
                logging.error(f"Error in LLM call for {model_name}, trial {trial}: {response}")
                continue

            try:
                for eval_item in response.get('evaluations', []):
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
                logging.error(f"Error processing response for {model_name}, trial {trial}: {e}")

        self._persist_single_doc_results()
        logging.info(f"Finished single-document evaluation for doc_id: {doc_id}")

    async def evaluate_pairwise_documents(self, doc_ids_contents):
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
        
        tasks = []
        for doc_id_1, doc_id_2 in combinations(all_doc_ids, 2):
            logging.info(f"  Creating tasks for pair: {doc_id_1} vs {doc_id_2}")
            for model_name in [model_a_name, model_b_name]:
                llm, parser, model_name_from_get_llm, enable_grounding = self._get_llm_and_parser(model_name)
                for trial in range(1, pairwise_trial_count + 1):
                    input_data = {
                        "doc_id_1": doc_id_1,
                        "document_1": document_map[doc_id_1],
                        "doc_id_2": doc_id_2,
                        "document_2": document_map[doc_id_2],
                        "criteria": pairwise_criteria # Pass pairwise criteria to the template
                    }
                    rendered_prompt = self.pairwise_doc_template.render(input_data)
                    
                    tasks.append(self._run_llm_call(llm, parser, rendered_prompt, model_name_from_get_llm, enable_grounding))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True) # Run tasks concurrently

        task_idx = 0
        for doc_id_1, doc_id_2 in combinations(all_doc_ids, 2):
            for model_name in [model_a_name, model_b_name]:
                for trial in range(1, pairwise_trial_count + 1):
                    response = responses[task_idx]
                    task_idx += 1

                    if isinstance(response, Exception):
                        logging.error(f"Error in LLM call for pair {doc_id_1} vs {doc_id_2} with {model_name} trial {trial}: {response}")
                        continue

                    try:
                        winner_doc_id = response.get('winner_doc_id')
                        reason = response.get('reason')

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
                        logging.error(f"Error processing response for pair {doc_id_1} vs {doc_id_2} with {model_name} trial {trial}: {e}")

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
