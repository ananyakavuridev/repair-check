from reasoning.prompt import build_prompt
from reasoning.llm import generate_diagnosis
from reasoning.quote_checker import check_quote
from retriever import retrieve


def diagnose(query, quote=None, brand=None, k=3):
    rows = retrieve(
        query=query,
        k=k,
        brand=brand
    )

    if not rows:
        return {
            "error": "No relevant knowledge-base evidence found."
        }

    normalized_rows = []

    for row in rows:
        normalized_rows.append({
            "document_id": row.get("document_id"),
            "brand": row.get("brand"),
            "code": row.get("code"),
            "fault": row.get("fault_description"),
            "fault_description": row.get("fault_description"),
            "severity": row.get("severity"),
            "fault_type": row.get("fault_type"),
            "component_category": row.get("component_category"),
            "difficulty": row.get("difficulty"),
            "cost_min": row.get("cost_min"),
            "cost_max": row.get("cost_max"),
            "cost_confidence": row.get("cost_confidence"),
            "match_type": row.get("match_type"),
            "similarity_score": row.get("similarity_score"),
            "distance": row.get("distance"),
            "document": row.get("document")
        })

    prompt = build_prompt(
        query=query,
        retrieved_rows=normalized_rows
    )

    diagnosis = generate_diagnosis(prompt)

    top_row = normalized_rows[0]

    cost_min = top_row.get("cost_min")
    cost_max = top_row.get("cost_max")

    diagnosis = check_quote(
        diagnosis,
        quote,
        cost_min,
        cost_max
    )

    return diagnosis.model_dump()


if __name__ == "__main__":
    result = diagnose(
        "My Daikin AC is showing E7"
    )
    print(result)
