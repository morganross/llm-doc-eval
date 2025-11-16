# llm-doc-eval
langchain based scripts to evaluate and compare documents based on criteria to provide structed evaluation output.

## Verbose Logging

To enable verbose logging, set the log level to DEBUG before running evaluations:

### Python API:
```python
import logging
logging.getLogger('llm_doc_eval.api').setLevel(logging.DEBUG)
```

### Environment Variable:
```bash
export PYTHONVERBOSE=1
```

### Programmatic Setup:
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
```

### Log Output Includes:
- Document discovery and counts
- Judge model configurations
- Run group IDs and temp directories
- FPF batch submission progress
- Output file parsing status
- Database persistence results
- Cost aggregation summaries

### FPF Detailed Logs:
FPF writes detailed JSON logs to temp directories:
- Single-doc: `/tmp/llm_doc_eval_single_logs_{run_group_id}/`
- Pairwise: `/tmp/llm_doc_eval_pair_logs_{run_group_id}/`

These logs contain full request/response data and cost breakdowns. 
