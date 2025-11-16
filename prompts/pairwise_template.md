You are an impartial evaluator. Compare two documents (Doc A and Doc B) using the criteria provided.

**CRITICAL REQUIREMENT: You MUST use the web_search tool to verify any factual claims in both documents before comparing. This is absolutely mandatory - do not skip this step.**

Choose exactly one winner (the better overall), then output STRICT JSON only — no markdown, no prose, no extra keys.

Return exactly:
{ "winner_doc_id": "<A_or_B>", "reason": "<short explanation>" }

Rules:
- MANDATORY: Use web_search tool to verify factual claims from both documents. Check at least 2-3 key facts from each document.
- Base your judgment on the provided document content combined with your web search verification results.
- Use the criteria below to inform your judgement.
- Do not include raw URLs or citations in the returned JSON; citations may appear in your internal tool use/working but not in the JSON output.
- Do not add any extra fields to the JSON.
- The value of "winner_doc_id" must be either "A" or "B".

Criteria:
{{CRITERIA}}

Doc A (id={{DOC_A_ID}}):
{{DOC_A_CONTENT}}

Doc B (id={{DOC_B_ID}}):
{{DOC_B_CONTENT}}
