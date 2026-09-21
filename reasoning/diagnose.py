from prompt import build_prompt
from llm import generate_diagnosis
from quote_checker import check_quote


# TEMPORARY MOCK
# Replace this with Segment 2's retrieve() later.
def retrieve(query, k=3, brand=None):
    return [
        {
            "document_id": "DEMO-001",
            "brand": "Daikin",
            "code": "E7",
            "fault": "Outdoor fan malfunction",
            "severity": "High",
            "cost_min": 1500,
            "cost_max": 3500,
            "cost_confidence": "sourced"
        }
    ]


def diagnose(query, quote=None, brand=None, k=3):

    # 1. Retrieve relevant KB rows
    rows = retrieve(
        query=query,
        k=k,
        brand=brand
    )

    if not rows:
        return {
            "error": "No relevant knowledge-base evidence found."
        }

    # 2. Build grounded prompt
    prompt = build_prompt(
        query=query,
        retrieved_rows=rows
    )

    # 3. Ask LLM
    diagnosis = generate_diagnosis(prompt)

    # 4. Get cost information from top retrieved row
    top_row = rows[0]

    cost_min = top_row.get("cost_min")
    cost_max = top_row.get("cost_max")

    # 5. Check technician quote
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