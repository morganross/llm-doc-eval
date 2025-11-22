import os
import sqlite3
import logging
from collections import defaultdict

def shorten_label(filename):
    """
    Shortens a filename label to '{ReportType} {ModelName} {Hash}'.
    Expected format: {DocName}.{ReportType}.{RunNum}.{ModelName}.{Hash}.{Ext}
    Example: 100_ EO 14er & Block.dr.1.gemini-2.5-flash.xmh.md -> DR gemini-2.5-flash xmh
    """
    parts = filename.split('.')
    if len(parts) < 5:
        return filename
        
    # Try to find the anchor point (ReportType followed by RunNum)
    anchor_idx = -1
    known_types = {'dr', 'fpf', 'gptr', 'ma'}
    
    for i in range(len(parts) - 1):
        # Check if this part is a known type and next part is a digit (RunNum)
        if parts[i] in known_types and parts[i+1].isdigit():
            anchor_idx = i
            break
    
    if anchor_idx != -1:
        report_type = parts[anchor_idx]
        # Model name is everything between RunNum (i+1) and Hash (second to last)
        # We assume the last two parts are Hash and Extension
        if len(parts) > anchor_idx + 3:
            model_parts = parts[anchor_idx+2 : -2]
            model_name = ".".join(model_parts)
            hash_val = parts[-2]
            return f"{report_type.upper()} {model_name} {hash_val}"
            
    return filename

def generate_html_report(db_path, output_dir):
    """
    Generates a self-contained HTML report with CSS styling and color-coded scores.
    Includes a Pairwise Matrix Heatmap.
    """
    logger = logging.getLogger("eval")
    logger.info(f"[HTML_EXPORT_START] Generating HTML report from {db_path}")

    if not os.path.exists(db_path):
        logger.error(f"[HTML_EXPORT_ERROR] Database not found: {db_path}")
        return

    html_path = os.path.join(output_dir, "evaluation_report.html")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # CSS Styles
        css = """
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f4f9; }
            h2 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 40px; }
            table { border-collapse: collapse; margin-bottom: 30px; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f8f9fa; font-weight: 600; color: #555; }
            tr:hover { background-color: #f1f1f1; }
            .score-badge { display: inline-block; padding: 4px 8px; border-radius: 4px; color: white; font-weight: bold; text-align: center; min-width: 20px; margin-right: 2px; cursor: help; }
            .score-high { background-color: #28a745; } /* Green */
            .score-mid { background-color: #ffc107; color: #333; } /* Yellow */
            .score-low { background-color: #dc3545; } /* Red */
            .footer { font-size: 0.8em; color: #777; margin-top: 20px; text-align: center; }
            
            /* Matrix Styles */
            .matrix-table { table-layout: fixed; width: auto; }
            .matrix-table th.matrix-header-top { 
                color: #dc3545; 
                font-weight: bold; 
                text-align: center; 
                border-bottom: 2px solid #dc3545;
                writing-mode: vertical-rl;
                transform: rotate(180deg);
                height: 150px;
                white-space: nowrap;
                padding: 5px;
                vertical-align: bottom;
            } 
            .matrix-table th.matrix-header-left { 
                color: #28a745; 
                font-weight: bold; 
                text-align: right; 
                border-right: 2px solid #28a745;
                white-space: nowrap;
            } 
            .matrix-cell { 
                text-align: center; 
                font-weight: bold; 
                border: 1px solid #eee;
                width: 50px;
                height: 50px;
                min-width: 50px;
                max-width: 50px;
                padding: 0;
                vertical-align: middle;
            }
            .matrix-cell-green-3 { background-color: #28a745; color: white; }
            .matrix-cell-green-2 { background-color: #5cb85c; color: white; }
            .matrix-cell-green-1 { background-color: #dff0d8; color: #3c763d; }
            .matrix-cell-red-3 { background-color: #dc3545; color: white; }
            .matrix-cell-red-2 { background-color: #d9534f; color: white; }
            .matrix-cell-red-1 { background-color: #f2dede; color: #a94442; }
            .matrix-cell-neutral { background-color: #f9f9f9; color: #ccc; }
        </style>
        """

        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            f'<head><meta charset="UTF-8"><title>Evaluation Report</title>{css}</head>',
            '<body>',
            '<h1>Evaluation Report</h1>'
        ]

        # --- Table 1: Pairwise Matrix Heatmap ---
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pairwise_results'")
        if cursor.fetchone():
            html_parts.append('<h2>Pairwise Head-to-Head Matrix</h2>')
            
            # 1. Aggregate Data
            # Map: (DocA, DocB) -> NetScore (Positive = A wins, Negative = B wins)
            scores = defaultdict(int)
            docs = set()
            
            cursor.execute("SELECT doc_id_1, doc_id_2, winner_doc_id FROM pairwise_results")
            for d1, d2, winner in cursor.fetchall():
                docs.add(d1)
                docs.add(d2)
                
                # Normalize key so d1 < d2 for storage
                if d1 < d2:
                    key = (d1, d2)
                    direction = 1 # +1 means d1 wins
                else:
                    key = (d2, d1)
                    direction = -1 # -1 means d2 (which is d1 in key) wins
                
                if winner == d1:
                    scores[key] += (1 * direction)
                elif winner == d2:
                    scores[key] -= (1 * direction)
            
            sorted_docs = sorted(list(docs))
            
            # Calculate Total Net Score for Winner Determination
            doc_totals = defaultdict(int)
            for row_doc in sorted_docs:
                for col_doc in sorted_docs:
                    if row_doc == col_doc: continue
                    
                    if row_doc < col_doc:
                        net = scores[(row_doc, col_doc)]
                    else:
                        net = -scores[(col_doc, row_doc)]
                    doc_totals[row_doc] += net
            
            # Identify Winner (Max Total Score)
            winner_doc = None
            if doc_totals:
                winner_doc = max(doc_totals, key=doc_totals.get)

            # 2. Build Matrix Table
            html_parts.append('<table class="matrix-table">')
            
            # Header Row (Red Axis)
            html_parts.append('<thead><tr><th></th>') # Empty corner
            for d in sorted_docs:
                label = shorten_label(d)
                if d == winner_doc:
                    label = f"🏆 {label}"
                html_parts.append(f'<th class="matrix-header-top">{label}</th>')
            html_parts.append('</tr></thead><tbody>')
            
            # Data Rows (Green Axis)
            for row_doc in sorted_docs:
                html_parts.append('<tr>')
                label = shorten_label(row_doc)
                if row_doc == winner_doc:
                    label = f"🏆 {label}"
                html_parts.append(f'<th class="matrix-header-left">{label}</th>')
                
                for col_doc in sorted_docs:
                    if row_doc == col_doc:
                        html_parts.append('<td class="matrix-cell matrix-cell-neutral">-</td>')
                        continue
                    
                    # Retrieve score
                    if row_doc < col_doc:
                        raw_score = scores[(row_doc, col_doc)]
                        # raw_score > 0 means row_doc (d1) won -> Green
                        # raw_score < 0 means col_doc (d2) won -> Red
                        net_score = raw_score
                    else:
                        raw_score = scores[(col_doc, row_doc)]
                        # raw_score stored relative to col_doc.
                        # if raw_score > 0, col_doc won -> Red for us
                        # if raw_score < 0, row_doc won -> Green for us
                        net_score = -raw_score
                    
                    # Determine Class
                    cls = "matrix-cell-neutral"
                    display_score = abs(net_score)
                    
                    if net_score > 0: # Row (Green) Wins
                        if net_score >= 3: cls = "matrix-cell-green-3"
                        elif net_score >= 2: cls = "matrix-cell-green-2"
                        else: cls = "matrix-cell-green-1"
                    elif net_score < 0: # Col (Red) Wins
                        if net_score <= -3: cls = "matrix-cell-red-3"
                        elif net_score <= -2: cls = "matrix-cell-red-2"
                        else: cls = "matrix-cell-red-1"
                        
                    html_parts.append(f'<td class="matrix-cell {cls}">{display_score}</td>')
                
                html_parts.append('</tr>')
            html_parts.append('</tbody></table>')

        # --- Table 2: Single Document Consensus Matrix ---
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='single_doc_results'")
        if cursor.fetchone():
            html_parts.append('<h2>Single Document Consensus Matrix</h2>')
            
            cursor.execute("SELECT doc_id, model, criterion, score, reason FROM single_doc_results")
            rows = cursor.fetchall()
            
            # Process data
            doc_map = {} # doc_id -> { label: str, criteria: { crit_name: [ {model, score, reason} ] } }
            all_criteria = set()
            
            for r in rows:
                doc_id, evaluator, criterion, score, reason = r
                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        'label': shorten_label(doc_id),
                        'criteria': {}
                    }
                
                if criterion not in doc_map[doc_id]['criteria']:
                    doc_map[doc_id]['criteria'][criterion] = []
                
                doc_map[doc_id]['criteria'][criterion].append({
                    'model': evaluator,
                    'score': score,
                    'reason': reason
                })
                all_criteria.add(criterion)
            
            sorted_criteria = sorted(list(all_criteria))
            sorted_docs = sorted(doc_map.values(), key=lambda x: x['label'])
            
            html_parts.append('<table>')
            
            # Header
            html_parts.append('<thead><tr><th>Document</th>')
            for c in sorted_criteria:
                html_parts.append(f'<th>{c}</th>')
            html_parts.append('</tr></thead><tbody>')
            
            # Rows
            for doc in sorted_docs:
                html_parts.append('<tr>')
                html_parts.append(f'<td style="font-weight:bold;">{doc["label"]}</td>')
                
                for crit in sorted_criteria:
                    html_parts.append('<td>')
                    evals = doc['criteria'].get(crit, [])
                    # Sort evals by model name for consistency
                    evals.sort(key=lambda x: x['model'])
                    
                    for e in evals:
                        score = e['score']
                        model = e['model']
                        reason = e['reason'].replace('"', '&quot;')
                        
                        # Color class
                        c_class = "score-low"
                        if score >= 4: c_class = "score-high"
                        elif score >= 3: c_class = "score-mid"
                        
                        # Render badge with tooltip
                        html_parts.append(f'<span class="score-badge {c_class}" title="{model}: {reason}">{score}</span>')
                        
                    html_parts.append('</td>')
                html_parts.append('</tr>')
            
            html_parts.append('</tbody></table>')

        # --- Table 3: Single Document Evaluations (Raw) ---
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='single_doc_results'")
        if cursor.fetchone():
            html_parts.append('<h2>Single Document Evaluations (Raw)</h2>')
            html_parts.append('<table>')
            
            cursor.execute("SELECT * FROM single_doc_results")
            rows = cursor.fetchall()
            headers = [d[0] for d in cursor.description]
            
            score_col_idx = -1
            for idx, header in enumerate(headers):
                if header.lower() == "score":
                    score_col_idx = idx
                    break

            html_parts.append('<thead><tr>')
            for h in headers:
                html_parts.append(f'<th>{h}</th>')
            html_parts.append('</tr></thead><tbody>')

            for row in rows:
                html_parts.append('<tr>')
                for idx, cell in enumerate(row):
                    val_str = str(cell)
                    if idx == score_col_idx:
                        try:
                            val = float(cell)
                            if val >= 4:
                                val_str = f'<span class="score-badge score-high">{cell}</span>'
                            elif val >= 3:
                                val_str = f'<span class="score-badge score-mid">{cell}</span>'
                            else:
                                val_str = f'<span class="score-badge score-low">{cell}</span>'
                        except:
                            pass
                    html_parts.append(f'<td>{val_str}</td>')
                html_parts.append('</tr>')
            html_parts.append('</tbody></table>')

        # --- Table 4: Raw Pairwise Data ---
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pairwise_results'")
        if cursor.fetchone():
            html_parts.append('<h2>Raw Pairwise Data</h2>')
            html_parts.append('<table>')
            cursor.execute("SELECT * FROM pairwise_results")
            rows = cursor.fetchall()
            headers = [d[0] for d in cursor.description]
            html_parts.append('<thead><tr>')
            for h in headers:
                html_parts.append(f'<th>{h}</th>')
            html_parts.append('</tr></thead><tbody>')
            for row in rows:
                html_parts.append('<tr>')
                for cell in row:
                    html_parts.append(f'<td>{cell}</td>')
                html_parts.append('</tr>')
            html_parts.append('</tbody></table>')

        html_parts.append('<div class="footer">Generated by API Cost Multiplier Reporting</div>')
        html_parts.append('</body></html>')
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        conn.close()
        logger.info(f"[HTML_EXPORT_SUCCESS] Report saved to {html_path}")
        print(f"Exported HTML report to: {html_path}")

    except Exception as e:
        logger.error(f"[HTML_EXPORT_ERROR] Failed to generate HTML report: {e}", exc_info=True)
        print(f"HTML export failed: {e}")
