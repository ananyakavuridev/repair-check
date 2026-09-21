"""
test_retrieval.py
=================
Demonstration script for RepairCheck Segment 2 Retrieval Layer.

Demonstrates:
1. Exact error-code queries (e.g., 'Voltas E1', 'LG CH05')
2. Vague symptom queries (e.g., 'AC running but not cooling the room')
3. Brand-specific queries (e.g., 'indoor fan motor fault' with brand='Daikin')

Note: This script demonstrates retrieval functionality and does not invent
or report any synthetic accuracy metrics.
"""

from retriever import retrieve


def format_result(idx: int, item: dict):
    cost_str = (
        f"₹{int(item['cost_min']):,} – ₹{int(item['cost_max']):,}"
        if item["cost_min"] >= 0 and item["cost_max"] >= 0
        else "Cost varies / Diagnostic required"
    )
    print(f"   [Rank {idx}] {item['brand']} | Code: {item['code']} (Doc ID: {item['document_id']})")
    print(f"      • Fault: {item['fault_description']}")
    print(f"      • Fault Type: {item['fault_type']} | Component: {item['component_category']} | Difficulty: {item['difficulty']}")
    print(f"      • Estimated Cost: {cost_str} (Confidence: {item['cost_confidence']})")
    print(f"      • Match: {item['match_type']} | Similarity: {item['similarity_score']:.4f} (Distance: {item['distance']:.4f})")
    print()


def run_demo():
    print("=" * 75)
    print(" RepairCheck - Segment 2 (Retrieval Layer Demonstration)")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. Exact Error-Code Query
    # -------------------------------------------------------------
    print("\n--- [TEST 1] Exact Error-Code Query ---")
    query_1 = "Voltas AC displaying error code E1"
    print(f"Query: \"{query_1}\"")
    print(f"Filter: None (Exact code detected automatically)")
    results_1 = retrieve(query=query_1, k=3)
    for i, res in enumerate(results_1, 1):
        format_result(i, res)

    # -------------------------------------------------------------
    # 2. Vague Symptom Query
    # -------------------------------------------------------------
    print("\n--- [TEST 2] Vague Symptom Query ---")
    query_2 = "AC running but not cooling the room properly"
    print(f"Query: \"{query_2}\"")
    print(f"Filter: None (Pure semantic vector similarity)")
    results_2 = retrieve(query=query_2, k=3)
    for i, res in enumerate(results_2, 1):
        format_result(i, res)

    # -------------------------------------------------------------
    # 3. Brand-Specific Query
    # -------------------------------------------------------------
    print("\n--- [TEST 3] Brand-Specific Query ---")
    query_3 = "indoor fan motor fault"
    brand_3 = "Daikin"
    print(f"Query: \"{query_3}\"")
    print(f"Filter: brand='{brand_3}'")
    results_3 = retrieve(query=query_3, k=3, brand=brand_3)
    for i, res in enumerate(results_3, 1):
        format_result(i, res)

    print("=" * 75)
    print(" Demonstration completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
