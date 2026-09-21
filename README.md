# RepairCheck - Segment 2: Retrieval Layer

Welcome to **Segment 2 (Retrieval Layer)** of the **RepairCheck** project.

This layer serves as the knowledge retrieval engine for an AI-assisted air conditioner repair diagnostics system. It provides vector similarity search and exact error code lookup over 201 AC fault records across major brands (Voltas, Daikin, LG, Samsung, Blue Star, Carrier, Hitachi, Haier, Godrej, Lloyd, Whirlpool, etc.).

---

## 📁 File Structure & Purpose

| File | Purpose |
| :--- | :--- |
| `repaircheck_final_dataset.xlsx` | The official Excel dataset containing the AC knowledge base (sheet: `Combined_Dataset`). Loaded strictly in read-only mode. |
| `data_loader.py` | Ingests the Excel file, creates clean document text representations for embedding, and formats type-safe metadata for vector storage. |
| `retriever.py` | The core retrieval engine. Handles ChromaDB vector indexing with `sentence-transformers/all-MiniLM-L6-v2`, exact error-code extraction, brand filtering, and exposes `retrieve(query, k=3, brand=None)`. |
| `test_retrieval.py` | Demonstration test script showcasing exact error-code queries, vague symptom queries, and brand-filtered queries. |
| `evaluate.py` | Evaluation harness to calculate **Top-1** and **Top-3** retrieval accuracy and generate a readable comparison report. |
| `requirements.txt` | Minimal dependencies (`pandas`, `openpyxl`, `chromadb`, `sentence-transformers`). |

---

## 🚀 Setup & Installation

### 1. Requirements
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
Open a terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

---

## 🛠️ How to Run

### 1. Run the Test Demonstration Script
Demonstrates three key query scenarios without inventing any accuracy numbers:
1. **Exact error-code query**: e.g. `"Voltas AC displaying error code E1"`
2. **Vague symptom query**: e.g. `"AC running but not cooling the room properly"`
3. **Brand-specific query**: e.g. `"indoor fan motor fault"` with `brand="Daikin"`

```bash
python test_retrieval.py
```

### 2. Run the Evaluation Benchmark
Calculates Top-1 and Top-3 accuracy against ground-truth evaluation queries (loaded from the `Test_Queries` sheet or the in-file `BENCHMARK_QUERIES` list):
```bash
python evaluate.py
```

To insert new queries from the knowledge base team:
- Edit the `BENCHMARK_QUERIES` list in `evaluate.py`, OR
- Add rows to the `Test_Queries` sheet in `repaircheck_final_dataset.xlsx`.

---

## 🧩 Modular Integration (Calling from Segment 3)

When building Segment 3 (the LLM reasoning layer), you can import and call `retrieve()` directly from `retriever.py`:

```python
from retriever import retrieve

# 1. Symptom diagnosis
matches = retrieve("water leaking from indoor unit", k=3)

# 2. Brand-specific diagnostic query
matches = retrieve("E1 error code", k=3, brand="Voltas")

# Accessing retrieved information:
for item in matches:
    print(item["brand"], item["code"])
    print("Fault:", item["fault_description"])
    print("Cost Range:", f"₹{item['cost_min']} - ₹{item['cost_max']}")
    print("Similarity Score:", item["similarity_score"])
```

---

## 📋 Document Representation & Metadata Schema

### Document Text Format
Each knowledge base row is embedded as a structured string:
```text
Brand: {Brand} | Code: {Code} | Fault Description: {Fault Description} | Fault Type: {Fault Type} | Component Category: {Component Category} | Difficulty: {Difficulty}
```

### Stored Metadata
- `document_id` (int)
- `brand` (str)
- `code` (str)
- `code_normalized` (str)
- `alt_codes` (str)
- `fault_description` (str)
- `fault_type` (str)
- `component_category` (str)
- `difficulty` (str)
- `cost_min` (float)
- `cost_max` (float)
- `cost_confidence` (str)
