"""
evaluate.py
===========
Evaluation Harness for RepairCheck Segment 2 Retrieval Layer.

Features:
1. Calculates Top-1 and Top-3 accuracy metrics against ground-truth KB entries.
2. Loads official evaluation queries directly from:
   - 'Test_Queries' sheet in repaircheck_final_dataset.xlsx (if present), OR
   - The customizable BENCHMARK_QUERIES Python list below.
3. Generates a clear, readable evaluation report displaying:
   - Query text and Brand filter
   - Expected KB row (Doc ID, Brand, Code, Fault Description)
   - Retrieved Top-3 candidates (Doc ID, Brand, Code, Similarity Score)
   - Top-1 and Top-3 Hit verification
4. Summary banner showing overall Top-1 and Top-3 Accuracy percentages.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from retriever import retrieve


# =====================================================================
# Customizable benchmark list:
# You can paste/edit the 15-20 queries directly here if preferred.
# =====================================================================
BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": 1,
        "query": "Daikin AC shows error code A1",
        "brand": "Daikin",
        "expected_code": "A1",
        "expected_doc_id": 23,
        "query_type": "exact_code"
    },
    {
        "query_id": 2,
        "query": "My Daikin AC has a problem with PCB / Inverter and is showing A1",
        "brand": "Daikin",
        "expected_code": "A1",
        "expected_doc_id": 23,
        "query_type": "code_plus_symptom"
    },
    {
        "query_id": 3,
        "query": "Daikin AC shows error code A3",
        "brand": "Daikin",
        "expected_code": "A3",
        "expected_doc_id": 24,
        "query_type": "exact_code"
    },
    {
        "query_id": 4,
        "query": "My Daikin AC has a problem with Mechanical and is showing A3",
        "brand": "Daikin",
        "expected_code": "A3",
        "expected_doc_id": 24,
        "query_type": "code_plus_symptom"
    },
    {
        "query_id": 5,
        "query": "Daikin AC shows error code A5",
        "brand": "Daikin",
        "expected_code": "A5",
        "expected_doc_id": 25,
        "query_type": "exact_code"
    },
    {
        "query_id": 6,
        "query": "My Daikin AC has a problem with Sensor and is showing A5",
        "brand": "Daikin",
        "expected_code": "A5",
        "expected_doc_id": 25,
        "query_type": "code_plus_symptom"
    },
    {
        "query_id": 7,
        "query": "Voltas AC shows error code E1",
        "brand": "Voltas",
        "expected_code": "E1",
        "expected_doc_id": 1,
        "query_type": "exact_code"
    },
    {
        "query_id": 8,
        "query": "My Voltas AC has a problem with Sensor and is showing E1",
        "brand": "Voltas",
        "expected_code": "E1",
        "expected_doc_id": 1,
        "query_type": "code_plus_symptom"
    },
    {
        "query_id": 9,
        "query": "Voltas AC shows error code E2",
        "brand": "Voltas",
        "expected_code": "E2",
        "expected_doc_id": 2,
        "query_type": "exact_code"
    },
    {
        "query_id": 10,
        "query": "My Voltas AC has a problem with Sensor and is showing E2",
        "brand": "Voltas",
        "expected_code": "E2",
        "expected_doc_id": 2,
        "query_type": "code_plus_symptom"
    },
    {
        "query_id": 11,
        "query": "Voltas AC shows error code E3",
        "brand": "Voltas",
        "expected_code": "E3",
        "expected_doc_id": 3,
        "query_type": "exact_code"
    },
    {
        "query_id": 12,
        "query": "My Voltas AC has a problem with Sensor and is showing E3",
        "brand": "Voltas",
        "expected_code": "E3",
        "expected_doc_id": 3,
        "query_type": "code_plus_symptom"
    },
    {
        "query_id": 13,
        "query": "My Daikin AC is having trouble with pcb / inverter; what could be wrong?",
        "brand": "Daikin",
        "expected_code": "A1",
        "expected_doc_id": 23,
        "query_type": "vague_symptom"
    },
    {
        "query_id": 14,
        "query": "My Daikin AC is having trouble with mechanical; what could be wrong?",
        "brand": "Daikin",
        "expected_code": "A3",
        "expected_doc_id": 24,
        "query_type": "vague_symptom"
    },
    {
        "query_id": 15,
        "query": "My Daikin AC is having trouble with sensor; what could be wrong?",
        "brand": "Daikin",
        "expected_code": "A5",
        "expected_doc_id": 25,
        "query_type": "vague_symptom"
    },
    {
        "query_id": 16,
        "query": "My Voltas AC is having trouble with sensor; what could be wrong?",
        "brand": "Voltas",
        "expected_code": "E1",
        "expected_doc_id": 1,
        "query_type": "vague_symptom"
    },
    {
        "query_id": 17,
        "query": "My Voltas AC is having trouble with indoor coil sensor; what could be wrong?",
        "brand": "Voltas",
        "expected_code": "E2",
        "expected_doc_id": 2,
        "query_type": "vague_symptom"
    },
    {
        "query_id": 18,
        "query": "My Voltas AC is having trouble with outdoor unit sensor leak; what could be wrong?",
        "brand": "Voltas",
        "expected_code": "E3",
        "expected_doc_id": 3,
        "query_type": "vague_symptom"
    }
]


def load_queries_from_excel(
    file_path: str = "repaircheck_final_dataset.xlsx",
    sheet_name: str = "Test_Queries"
) -> Optional[List[Dict[str, Any]]]:
    """
    Attempts to read evaluation queries directly from the Excel dataset.
    """
    if not os.path.exists(file_path):
        return None
    try:
        df_queries = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        queries = []
        for _, row in df_queries.iterrows():
            queries.append({
                "query_id": int(row.get("query_id", len(queries) + 1)),
                "query": str(row.get("query", "")).strip(),
                "brand": str(row.get("brand", "")).strip() if pd.notna(row.get("brand")) else None,
                "expected_code": str(row.get("correct_code", "")).strip() if pd.notna(row.get("correct_code")) else "",
                "expected_doc_id": int(row.get("correct_document_id")) if pd.notna(row.get("correct_document_id")) else None,
                "query_type": str(row.get("query_type", "general")).strip(),
            })
        return queries
    except Exception:
        return None


def is_match(expected_item: dict, candidate: dict) -> bool:
    """
    Determines if a retrieved candidate matches the expected benchmark row.
    Matches primarily by document_id, or brand + normalized code.
    """
    exp_doc_id = expected_item.get("expected_doc_id")
    cand_doc_id = candidate.get("document_id")

    if exp_doc_id is not None and cand_doc_id is not None:
        if exp_doc_id == cand_doc_id:
            return True

    # Secondary check: brand & code
    exp_brand = str(expected_item.get("brand", "")).lower()
    cand_brand = str(candidate.get("brand", "")).lower()
    exp_code = str(expected_item.get("expected_code", "")).strip().lower()
    cand_code = str(candidate.get("code", "")).strip().lower()

    if exp_code and cand_code:
        if exp_code == cand_code:
            if not exp_brand or exp_brand == cand_brand or cand_brand == "cross-brand":
                return True

    return False


def run_evaluation(queries: Optional[List[Dict[str, Any]]] = None):
    """
    Runs evaluation against queries, computes Top-1 and Top-3 accuracy,
    and prints a detailed, readable comparison report.
    """
    if queries is None:
        # Check if Excel has Test_Queries sheet, else use BENCHMARK_QUERIES
        queries = load_queries_from_excel()
        if not queries:
            queries = BENCHMARK_QUERIES

    print("=" * 80)
    print(" RepairCheck - Retrieval Layer Evaluation Benchmark")
    print(f" Total Evaluation Queries: {len(queries)}")
    print("=" * 80)

    top1_hits = 0
    top3_hits = 0
    detailed_rows = []

    for i, item in enumerate(queries, 1):
        q = item["query"]
        b = item.get("brand")
        exp_id = item.get("expected_doc_id")
        exp_code = item.get("expected_code")
        q_type = item.get("query_type", "n/a")

        # Call retrieve()
        retrieved = retrieve(query=q, k=3, brand=b)

        top1_match = False
        top3_match = False
        retrieved_summaries = []

        for rank, cand in enumerate(retrieved, 1):
            matched = is_match(item, cand)
            if rank == 1 and matched:
                top1_match = True
            if matched:
                top3_match = True

            retrieved_summaries.append(
                f"#{rank} [Doc {cand['document_id']}|{cand['brand']} {cand['code']}|score={cand['similarity_score']:.3f}]"
            )

        if top1_match:
            top1_hits += 1
        if top3_match:
            top3_hits += 1

        status_str = "Top-1 HIT" if top1_match else ("Top-3 HIT" if top3_match else "MISS")
        detailed_rows.append({
            "id": i,
            "query": q,
            "brand": b or "Any",
            "type": q_type,
            "expected": f"Doc {exp_id} ({exp_code})",
            "retrieved_top3": " | ".join(retrieved_summaries),
            "status": status_str,
            "top1": "YES" if top1_match else "NO",
            "top3": "YES" if top3_match else "NO"
        })

        # Print single query summary
        print(f"\n[Query {i}/{len(queries)}] ({q_type})")
        print(f"  Query:    \"{q}\" (Brand: {b or 'Any'})")
        print(f"  Expected: Doc ID {exp_id} | Code: {exp_code}")
        print(f"  Retrieved Top 3:")
        for r_idx, cand in enumerate(retrieved, 1):
            flag = " <-- MATCH" if is_match(item, cand) else ""
            print(f"    {r_idx}. Doc {cand['document_id']} | {cand['brand']} {cand['code']} | Sim: {cand['similarity_score']:.4f} | {cand['fault_description'][:60]}...{flag}")
        print(f"  Result:   Top-1: {'[PASS]' if top1_match else '[FAIL]'} | Top-3: {'[PASS]' if top3_match else '[FAIL]'}")

    # Summary calculations
    total = len(queries)
    top1_acc = (top1_hits / total) * 100 if total > 0 else 0.0
    top3_acc = (top3_hits / total) * 100 if total > 0 else 0.0

    print("\n" + "=" * 80)
    print(" EVALUATION SUMMARY REPORT")
    print("=" * 80)
    print(f" Total Queries Tested:  {total}")
    print(f" Top-1 Hits:            {top1_hits}/{total} ({top1_acc:.2f}%)")
    print(f" Top-3 Hits:            {top3_hits}/{total} ({top3_acc:.2f}%)")
    print("-" * 80)

    return {
        "total_queries": total,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "top1_accuracy_percent": round(top1_acc, 2),
        "top3_accuracy_percent": round(top3_acc, 2),
        "details": detailed_rows
    }


if __name__ == "__main__":
    run_evaluation()
