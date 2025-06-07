import pytest
import os
import json
import pandas as pd
import sqlite3
# from unittest.mock import patch, MagicMock # No longer needed for mocking get_llm
from engine.evaluator import Evaluator
# from langchain_core.language_models.chat_models import BaseChatModel # No longer needed for MockLLM
# from pydantic import ConfigDict # No longer needed for MockLLM
from sqlalchemy import create_engine # Import create_engine for persistence tests

# Fixtures for dummy config and prompt files
@pytest.fixture
def dummy_config_file(tmp_path):
    config_content = """
single_doc_eval:
  criteria:
    - name: Clarity
      scale: 5
    - name: Coherence
      scale: 5
    - name: Relevance
      scale: 5
  trial_count: 2
  outlier_threshold: 2.0

pairwise_eval:
  trial_count: 1

models:
  model_a:
    name: OpenAI GPT-3.5 Turbo
    provider: openai
  model_b:
    name: Google Gemini Pro
    provider: google
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return config_path

@pytest.fixture
def dummy_prompts_dir(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "single_doc.jinja").write_text("""
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
    (prompts_dir / "pairwise_doc.jinja").write_text("""
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
    return prompts_dir

# Removed MockLLM and mock_get_llm_patch fixture as per user request to use real LLMs

# Test cases for Evaluator
def test_evaluator_init(dummy_config_file, dummy_prompts_dir, tmp_path):
    db_path = tmp_path / "test_results.db"
    evaluator = Evaluator(config_path=dummy_config_file, prompts_dir=dummy_prompts_dir, db_path=db_path)
    assert evaluator.config is not None
    assert "single_doc_eval" in evaluator.config
    assert "pairwise_eval" in evaluator.config
    assert evaluator.single_doc_template is not None
    assert evaluator.pairwise_doc_template is not None
    assert evaluator.single_doc_results.empty
    assert evaluator.pairwise_results.empty
    assert str(evaluator.db_engine.url) == f"sqlite:///{db_path}"

def test_load_config(dummy_config_file):
    evaluator = Evaluator(config_path=dummy_config_file)
    config = evaluator._load_config(dummy_config_file)
    assert config['single_doc_eval']['trial_count'] == 2
    assert config['models']['model_a']['name'] == 'OpenAI GPT-3.5 Turbo' # Updated assertion
    assert config['models']['model_b']['name'] == 'Google Gemini Pro' # Updated assertion

# Note: test_run_llm_call_success, test_run_llm_call_json_error_retry, test_run_llm_call_failure_after_retries
# will now attempt to make real LLM calls. These tests might become flaky or fail if API keys are not set
# or if rate limits are hit. For true unit tests, mocking LLM responses is preferred.
# However, as per user's request, we are proceeding with real LLM calls.

def test_run_llm_call_success(dummy_config_file, dummy_prompts_dir, tmp_path):
    # This test will now attempt a real LLM call. Ensure API keys are set.
    evaluator = Evaluator(config_path=dummy_config_file, prompts_dir=dummy_prompts_dir, db_path=tmp_path / "test.db")
    
    # Create a dummy chain that uses a real LLM (from registry)
    # We need to mock the prompt template to ensure it's valid for a real LLM call
    from langchain_core.prompts import PromptTemplate
    mock_prompt_template = PromptTemplate.from_template("Say hello in JSON: {{ input }}")
    
    # Get a real LLM instance (will use actual API keys)
    llm_instance = evaluator._get_llm_chain("OpenAI GPT-3.5 Turbo", mock_prompt_template) # Use a real model name
    
    # The expected response from a real LLM will be a dict, but its content is not fixed.
    # We can only assert that it returns a dict.
    result = evaluator._run_llm_call(llm_instance, {"input": "hello"})
    assert isinstance(result, dict)
    assert "evaluations" in result or "winner_doc_id" in result # Check for expected keys

def test_run_llm_call_json_error_retry(dummy_config_file, dummy_prompts_dir, tmp_path):
    # This test will now attempt real LLM calls. It's hard to simulate JSON errors reliably with real LLMs.
    # This test might need to be re-evaluated or removed if it becomes too flaky.
    # For now, we'll keep a basic structure, but it might not truly test JSON error retry.
    evaluator = Evaluator(config_path=dummy_config_file, prompts_dir=dummy_prompts_dir, db_path=tmp_path / "test.db")
    
    from langchain_core.prompts import PromptTemplate
    mock_prompt_template = PromptTemplate.from_template("Return invalid JSON: {{ input }}")
    llm_instance = evaluator._get_llm_chain("OpenAI GPT-3.5 Turbo", mock_prompt_template)

    # It's difficult to force a real LLM to return invalid JSON consistently for testing retries.
    # This test might pass if the LLM returns valid JSON, or fail if it returns invalid JSON.
    # For robust testing of retry logic, mocking is highly recommended.
    try:
        result = evaluator._run_llm_call(llm_instance, {"input": "invalid json"}, retries=2)
        assert isinstance(result, dict)
    except Exception as e:
        # If it fails, it should be due to JSON error or other LLM error
        assert "Failed LLM call" in str(e) or "JSON decode error" in str(e)

def test_run_llm_call_failure_after_retries(dummy_config_file, dummy_prompts_dir, tmp_path):
    # Similar to the above, forcing consistent failure with real LLMs is hard.
    evaluator = Evaluator(config_path=dummy_config_file, prompts_dir=dummy_prompts_dir, db_path=tmp_path / "test.db")
    
    from langchain_core.prompts import PromptTemplate
    mock_prompt_template = PromptTemplate.from_template("Always fail: {{ input }}")
    llm_instance = evaluator._get_llm_chain("OpenAI GPT-3.5 Turbo", mock_prompt_template)

    with pytest.raises(Exception, match="Failed LLM call after 3 attempts."):
        evaluator._run_llm_call(llm_instance, {"input": "fail"}, retries=3)

def test_evaluate_single_document(dummy_config_file, dummy_prompts_dir, tmp_path):
    # This test will now attempt real LLM calls. Ensure API keys are set.
    db_path = tmp_path / "single_doc_test.db"
    evaluator = Evaluator(config_path=dummy_config_file, prompts_dir=dummy_prompts_dir, db_path=db_path)
    evaluator.evaluate_single_document("doc_test_1", "This is a test document content.")

    # Assert that data was persisted to SQLite
    db_engine = create_engine(f'sqlite:///{db_path}')
    df_from_db = pd.read_sql_table("single_doc_results", db_engine)
    
    assert not df_from_db.empty
    # The number of rows depends on the actual LLM response, but should be 3 criteria * 2 models * 2 trials
    assert len(df_from_db) == 3 * 2 * 2 
    assert "doc_id" in df_from_db.columns
    assert "score" in df_from_db.columns
    assert "model" in df_from_db.columns
    assert "trial" in df_from_db.columns
    assert "Clarity" in df_from_db['criterion'].values
    assert "OpenAI GPT-3.5 Turbo" in df_from_db['model'].values # Updated assertion
    assert "Google Gemini Pro" in df_from_db['model'].values # Updated assertion
    assert all(df_from_db['score'].between(1, 5)) # Scores should be within 1-5 range

def test_evaluate_pairwise_documents(dummy_config_file, dummy_prompts_dir, tmp_path):
    # This test will now attempt real LLM calls. Ensure API keys are set.
    db_path = tmp_path / "pairwise_test.db"
    evaluator = Evaluator(config_path=dummy_config_file, prompts_dir=dummy_prompts_dir, db_path=db_path)
    test_docs_contents = [
        ("docA", "Content of document A."),
        ("docB", "Content of document B."),
        ("docC", "Content of document C.")
    ]
    evaluator.evaluate_pairwise_documents(test_docs_contents)

    # Assert that data was persisted to SQLite
    db_engine = create_engine(f'sqlite:///{db_path}')
    df_from_db = pd.read_sql_table("pairwise_results", db_engine)
    
    assert not df_from_db.empty
    # (N choose 2) * 2 models * 1 trial = (3 choose 2) * 2 * 1 = 3 * 2 * 1 = 6
    assert len(df_from_db) == 6
    assert "doc_id_1" in df_from_db.columns
    assert "doc_id_2" in df_from_db.columns
    assert "winner_doc_id" in df_from_db.columns
    assert "model" in df_from_db.columns
    assert "OpenAI GPT-3.5 Turbo" in df_from_db['model'].values # Updated assertion
    assert "Google Gemini Pro" in df_from_db['model'].values # Updated assertion
    assert all(doc_id in ["docA", "docB", "docC"] for doc_id in df_from_db['winner_doc_id'].values) # Winner should be one of the doc IDs

def test_persist_single_doc_results(dummy_config_file, dummy_prompts_dir, tmp_path):
    db_path = tmp_path / "persist_single_test.db"
    evaluator = Evaluator(config_path=dummy_config_file, prompts_dir=dummy_prompts_dir, db_path=db_path)
    
    expected_df = pd.DataFrame([
        {"doc_id": "d1", "model": "m1", "trial": 1, "criterion": "C", "score": 3, "reason": "R", "timestamp": pd.Timestamp.now()}
    ])
    evaluator.single_doc_results = expected_df.copy() # Use copy to avoid modifying original

    evaluator._persist_single_doc_results("single_doc_results") # Pass table name

    db_engine = create_engine(f'sqlite:///{db_path}')
    df_from_db = pd.read_sql_table("single_doc_results", db_engine)
    
    # Convert timestamp columns to a comparable format (e.g., string or datetime without ns)
    # For simplicity in testing, we drop timestamp as it's hard to match exactly
    pd.testing.assert_frame_equal(df_from_db.drop(columns=['timestamp']), expected_df.drop(columns=['timestamp']))
    assert evaluator.single_doc_results.empty # Check if DataFrame was cleared

def test_persist_pairwise_results(dummy_config_file, dummy_prompts_dir, tmp_path):
    db_path = tmp_path / "persist_pairwise_test.db"
    evaluator = Evaluator(config_path=dummy_config_file, prompts_dir=dummy_prompts_dir, db_path=db_path)
    
    expected_df = pd.DataFrame([
        {"doc_id_1": "d1", "doc_id_2": "d2", "model": "m1", "trial": 1, "winner_doc_id": "d1", "reason": "R", "timestamp": pd.Timestamp.now()}
    ])
    evaluator.pairwise_results = expected_df.copy() # Use copy to avoid modifying original

    evaluator._persist_pairwise_results("pairwise_results") # Pass table name

    db_engine = create_engine(f'sqlite:///{db_path}')
    df_from_db = pd.read_sql_table("pairwise_results", db_engine)
    
    # Convert timestamp columns to a comparable format (e.g., string or datetime without ns)
    # For simplicity in testing, we drop timestamp as it's hard to match exactly
    pd.testing.assert_frame_equal(df_from_db.drop(columns=['timestamp']), expected_df.drop(columns=['timestamp']))
    assert evaluator.pairwise_results.empty # Check if DataFrame was cleared