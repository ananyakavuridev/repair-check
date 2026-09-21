"""
data_loader.py
==============
Data loader and preprocessor for RepairCheck Segment 2 (Retrieval Layer).

Responsibilities:
1. Load the 'Combined_Dataset' sheet from repaircheck_final_dataset.xlsx in read-only mode.
2. Build a clean, structured document text representation for each row using:
   - Brand
   - Code
   - Fault Description
   - Fault Type
   - Component Category
   - Difficulty
3. Extract clean, type-safe metadata for each row:
   - Brand
   - Code
   - Fault Type
   - Component Category
   - Difficulty
   - cost_min
   - cost_max
   - cost_confidence
   - document_id
   - Fault Description
"""

import os
import pandas as pd
from typing import Tuple, List, Dict, Any


def load_dataset(
    file_path: str = "repaircheck_final_dataset.xlsx",
    sheet_name: str = "Combined_Dataset"
) -> pd.DataFrame:
    """
    Reads the specified sheet from the Excel file in read-only mode.
    Does NOT modify the original file under any circumstance.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file '{file_path}' not found. Please ensure the Excel file exists."
        )

    # Read using openpyxl engine in read-only manner
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    return df


def create_document_text(row: pd.Series) -> str:
    """
    Creates a clean document string representation for semantic embedding.
    Uses Brand, Code, Fault Description, Fault Type, Component Category, Difficulty.
    """
    brand = str(row.get("Brand", "")).strip() if pd.notna(row.get("Brand")) else "Unknown Brand"
    code = str(row.get("Code", "")).strip() if pd.notna(row.get("Code")) else "No Code"
    fault_desc = str(row.get("Fault Description", "")).strip() if pd.notna(row.get("Fault Description")) else ""
    fault_type = str(row.get("Fault Type", "")).strip() if pd.notna(row.get("Fault Type")) else ""
    comp_cat = str(row.get("Component Category", "")).strip() if pd.notna(row.get("Component Category")) else ""
    difficulty = str(row.get("Difficulty", "")).strip() if pd.notna(row.get("Difficulty")) else ""

    parts = [
        f"Brand: {brand}",
        f"Code: {code}",
        f"Fault Description: {fault_desc}",
        f"Fault Type: {fault_type}",
        f"Component Category: {comp_cat}",
        f"Difficulty: {difficulty}"
    ]
    return " | ".join(parts)


def create_metadata(row: pd.Series) -> Dict[str, Any]:
    """
    Extracts metadata for Chroma vector store.
    Chroma requires metadata values to be primitive types: str, int, float, or bool.
    Nulls/NaNs are replaced with safe defaults.
    """
    def _clean_str(val: Any, default: str = "") -> str:
        if pd.isna(val):
            return default
        return str(val).strip()

    def _clean_float(val: Any, default: float = -1.0) -> float:
        if pd.isna(val):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _clean_int(val: Any, default: int = 0) -> int:
        if pd.isna(val):
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    brand = _clean_str(row.get("Brand"))
    code = _clean_str(row.get("Code"))
    code_normalized = _clean_str(row.get("code_normalized", code))
    alt_codes = _clean_str(row.get("alt_codes"))
    fault_desc = _clean_str(row.get("Fault Description"))
    fault_type = _clean_str(row.get("Fault Type"))
    comp_cat = _clean_str(row.get("Component Category"))
    difficulty = _clean_str(row.get("Difficulty"))
    cost_min = _clean_float(row.get("cost_min"))
    cost_max = _clean_float(row.get("cost_max"))
    cost_confidence = _clean_str(row.get("cost_confidence"))
    doc_id = _clean_int(row.get("document_id"))
    entry_type = _clean_str(row.get("entry_type", "error_code"))

    metadata = {
        "document_id": doc_id,
        "brand": brand,
        "code": code,
        "code_normalized": code_normalized,
        "alt_codes": alt_codes,
        "entry_type": entry_type,
        "fault_description": fault_desc,
        "fault_type": fault_type,
        "component_category": comp_cat,
        "difficulty": difficulty,
        "cost_min": cost_min,
        "cost_max": cost_max,
        "cost_confidence": cost_confidence,
    }
    return metadata


def prepare_documents_and_metadata(
    df: pd.DataFrame
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Processes all rows of the DataFrame and returns:
    - documents: list of formatted text strings
    - metadatas: list of metadata dicts
    - ids: list of unique string IDs (e.g., 'doc_1', 'doc_2', ...)
    """
    documents = []
    metadatas = []
    ids = []

    for _, row in df.iterrows():
        doc_text = create_document_text(row)
        meta = create_metadata(row)
        doc_id = str(meta["document_id"]) if meta["document_id"] > 0 else str(len(ids) + 1)
        
        documents.append(doc_text)
        metadatas.append(meta)
        ids.append(f"doc_{doc_id}")

    return documents, metadatas, ids


if __name__ == "__main__":
    print("Testing data_loader.py...")
    dataset = load_dataset()
    print(f"Successfully loaded {len(dataset)} rows from Combined_Dataset sheet.")
    docs, metas, ids = prepare_documents_and_metadata(dataset)
    print(f"Sample ID: {ids[0]}")
    print(f"Sample Document:\n  {docs[0]}")
    print(f"Sample Metadata:\n  {metas[0]}")
