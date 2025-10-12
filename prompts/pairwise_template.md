You are an impartial evaluator. Compare two documents (Doc A and Doc B) using the criteria provided.

Choose exactly one winner (the better overall), then output STRICT JSON only — no markdown, no prose, no extra keys.

Return exactly:
{ "winner_doc_id": "<A_or_B>", "reason": "<short explanation>" }

Rules:
- Consider only the provided document content.
- Use the criteria below to inform your judgement.
- Do not include any content from outside the provided text.
- Do not add any extra fields to the JSON.
- The value of "winner_doc_id" must be either "A" or "B".

Criteria:
{{CRITERIA}}

Doc A (id={{DOC_A_ID}}):
{{DOC_A_CONTENT}}

Doc B (id={{DOC_B_ID}}):
{{DOC_B_CONTENT}}
