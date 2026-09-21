"""
retriever.py
============
Core Retrieval Layer for RepairCheck (Segment 2).

Key Features:
1. Ingests clean document representations & metadata from Combined_Dataset sheet.
2. Uses sentence-transformers/all-MiniLM-L6-v2 embedding model via ChromaDB.
3. Persistent vector collection in './chroma_db' for fast repeated queries.
4. Exact error code detection & lookup:
   - Identifies exact manufacturer error codes in user queries (e.g., E1, CH05, A1, U4, F9).
   - Returns exact KB matches with top priority so exact code queries do not rely purely on semantic vectors.
5. Brand filtering support (e.g., brand='Voltas', brand='Daikin').
6. Modular API: retrieve(query, k=3, brand=None) for direct use in test, evaluation, and future diagnose() workflows.
"""

import os
import re

# Enable offline mode for Hugging Face so cached weights are loaded instantly without network checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from typing import List, Dict, Any, Optional
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

from data_loader import load_dataset, prepare_documents_and_metadata


class RepairCheckRetriever:
    """
    Manages vector indexing and hybrid retrieval (exact code + semantic search)
    for RepairCheck AC error codes and symptoms.
    """

    def __init__(
        self,
        dataset_path: str = "repaircheck_final_dataset.xlsx",
        sheet_name: str = "Combined_Dataset",
        db_dir: str = "./chroma_db",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.dataset_path = dataset_path
        self.sheet_name = sheet_name
        self.db_dir = db_dir
        self.model_name = model_name

        # Load raw dataset
        self.df = load_dataset(self.dataset_path, self.sheet_name)
        self.documents, self.metadatas, self.ids = prepare_documents_and_metadata(self.df)

        # Build in-memory lookup table for fast exact-code matching
        self._build_code_index()

        # Initialize ChromaDB persistent client & collection
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(
            name="repaircheck_kb",
            embedding_function=self.embedding_fn,
            metadata={"description": "RepairCheck Segment 2 Knowledge Base"}
        )

        # Index documents if collection is empty or row count mismatch
        self._ensure_indexed()

    def _build_code_index(self):
        """
        Builds a normalized error code lookup mapping each error code
        to its matching row metadatas and document texts.
        Only indexes entries where entry_type is 'error_code' to prevent
        symptom phrases (like 'not cooling') from acting as exact error codes.
        """
        self.code_index: Dict[str, List[Dict[str, Any]]] = {}
        self.known_brands = set()

        for meta, doc in zip(self.metadatas, self.documents):
            entry = {"metadata": meta, "document": doc}
            brand = meta["brand"].strip()
            if brand:
                self.known_brands.add(brand.lower())

            # Only index error codes from genuine error_code entries
            if meta.get("entry_type") != "error_code":
                continue

            raw_code = meta.get("code", "")
            code_norm = meta.get("code_normalized", "")
            alt_codes = meta.get("alt_codes", "")

            # Extract individual codes from code strings
            code_candidates = set()
            for text in [raw_code, code_norm, alt_codes]:
                if not text:
                    continue
                parts = re.split(r"[/, ]+", text)
                for p in parts:
                    clean_p = p.strip().upper()
                    if clean_p and clean_p not in ["NO", "CODE", "SYMPTOM", "NA", "N/A"]:
                        code_candidates.add(clean_p)
                if "/" in text or (len(text.strip()) <= 8 and " " not in text.strip()):
                    code_candidates.add(text.strip().upper())

            for code in code_candidates:
                if code not in self.code_index:
                    self.code_index[code] = []
                self.code_index[code].append(entry)

    def _ensure_indexed(self):
        """
        Indexes documents into ChromaDB collection if not already populated.
        """
        current_count = self.collection.count()
        expected_count = len(self.documents)
        needs_reindex = (current_count != expected_count)

        if current_count > 0 and not needs_reindex:
            sample = self.collection.get(limit=1)
            if sample and sample.get("metadatas") and len(sample["metadatas"]) > 0:
                if "entry_type" not in sample["metadatas"][0]:
                    needs_reindex = True

        if needs_reindex:
            print(f"[Indexer] Updating collection ({expected_count} rows)...")
            try:
                self.client.delete_collection("repaircheck_kb")
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name="repaircheck_kb",
                embedding_function=self.embedding_fn,
                metadata={"description": "RepairCheck Segment 2 Knowledge Base"}
            )
            batch_size = 50
            for i in range(0, expected_count, batch_size):
                self.collection.upsert(
                    documents=self.documents[i : i + batch_size],
                    metadatas=self.metadatas[i : i + batch_size],
                    ids=self.ids[i : i + batch_size],
                )
            print(f"[Indexer] Successfully indexed {self.collection.count()} rows into Chroma collection.")
        else:
            print(f"[Indexer] Reusing existing Chroma collection ({current_count} documents indexed).")

    def _extract_brand_from_query(self, query: str) -> Optional[str]:
        """
        Detects if a known brand is mentioned in the query text.
        """
        query_lower = query.lower()
        for brand in self.known_brands:
            if re.search(rf"\b{re.escape(brand)}\b", query_lower):
                # Return canonical case from dataframe
                for meta in self.metadatas:
                    if meta["brand"].lower() == brand:
                        return meta["brand"]
        return None

    def _extract_exact_codes(self, query: str) -> List[str]:
        """
        Finds any exact error codes mentioned in the query that match known codes in the KB.
        """
        tokens = re.findall(r"\b[A-Za-z0-9]+(?:/[A-Za-z0-9]+)?\b", query)
        detected_codes = []
        ignored_words = {
            "AC", "NO", "CODE", "CODES", "PCB", "DC", "ON", "OFF", "IN", "OUT",
            "AN", "AS", "AT", "BY", "FOR", "IF", "IS", "IT", "MY", "OF", "OR",
            "TO", "UP", "THE", "AND", "NOT", "HAS", "UNIT", "WITH", "SHOWING",
            "SHOWS", "ERROR", "ERRORS", "TROUBLE", "PROBLEM", "FAULT"
        }

        for token in tokens:
            up_tok = token.upper()
            if up_tok in self.code_index and up_tok not in ignored_words:
                if up_tok not in detected_codes:
                    detected_codes.append(up_tok)

        return detected_codes

    def retrieve(
        self,
        query: str,
        k: int = 3,
        brand: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k matching KB records for a query.

        Parameters:
        - query (str): User problem statement or error code (e.g., 'E1' or 'water dripping from AC').
        - k (int): Number of top results to return (default: 3).
        - brand (str, optional): Brand filter (e.g., 'Voltas', 'Daikin'). If None, infers from query.

        Returns:
        - List of dicts with document text, similarity score, distance, and metadata.
        """
        target_brand = brand if brand else self._extract_brand_from_query(query)
        detected_codes = self._extract_exact_codes(query)

        results = []
        seen_doc_ids = set()

        # -------------------------------------------------------------
        # 1. Exact Code Matching (High Priority)
        # -------------------------------------------------------------
        if detected_codes:
            for code in detected_codes:
                matching_entries = self.code_index.get(code, [])
                for entry in matching_entries:
                    meta = entry["metadata"]
                    doc_id = meta["document_id"]
                    row_brand = meta["brand"]

                    # Filter by brand if specified
                    if target_brand:
                        if row_brand.lower() != target_brand.lower() and row_brand.lower() != "cross-brand":
                            continue

                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        results.append({
                            "document_id": doc_id,
                            "brand": row_brand,
                            "code": meta["code"],
                            "fault_description": meta["fault_description"],
                            "fault_type": meta["fault_type"],
                            "component_category": meta["component_category"],
                            "difficulty": meta["difficulty"],
                            "cost_min": meta["cost_min"],
                            "cost_max": meta["cost_max"],
                            "cost_confidence": meta["cost_confidence"],
                            "document": entry["document"],
                            "distance": 0.0,
                            "similarity_score": 1.0,
                            "match_type": "exact_code",
                        })

        # If exact matching already satisfied top-k, return top-k
        if len(results) >= k:
            return results[:k]

        # -------------------------------------------------------------
        # 2. Semantic Similarity Vector Search via Chroma
        # -------------------------------------------------------------
        remaining_slots = k - len(results)
        # Fetch more candidates from Chroma to allow deduplication against exact matches
        fetch_n = k + len(seen_doc_ids) + 5

        chroma_where = None
        if target_brand:
            chroma_where = {"brand": target_brand}

        try:
            chroma_res = self.collection.query(
                query_texts=[query],
                n_results=min(fetch_n, self.collection.count()),
                where=chroma_where
            )
        except Exception as e:
            # Fallback if brand filter matched 0 records in Chroma metadata
            chroma_res = self.collection.query(
                query_texts=[query],
                n_results=min(fetch_n, self.collection.count())
            )

        if chroma_res and chroma_res["documents"] and chroma_res["documents"][0]:
            docs = chroma_res["documents"][0]
            metas = chroma_res["metadatas"][0]
            distances = chroma_res["distances"][0]

            for doc, meta, dist in zip(docs, metas, distances):
                doc_id = meta["document_id"]
                if doc_id in seen_doc_ids:
                    continue

                seen_doc_ids.add(doc_id)
                # Compute normalized similarity score from L2 distance: 1 / (1 + distance)
                sim_score = round(1.0 / (1.0 + float(dist)), 4)

                results.append({
                    "document_id": doc_id,
                    "brand": meta["brand"],
                    "code": meta["code"],
                    "fault_description": meta["fault_description"],
                    "fault_type": meta["fault_type"],
                    "component_category": meta["component_category"],
                    "difficulty": meta["difficulty"],
                    "cost_min": meta["cost_min"],
                    "cost_max": meta["cost_max"],
                    "cost_confidence": meta["cost_confidence"],
                    "document": doc,
                    "distance": round(float(dist), 4),
                    "similarity_score": sim_score,
                    "match_type": "semantic",
                })

                if len(results) >= k:
                    break

        return results[:k]


# ----------------------------------------------------------------------
# Modular global interface for external callers (e.g. diagnose(query))
# ----------------------------------------------------------------------
_retriever_instance: Optional[RepairCheckRetriever] = None


def get_retriever(dataset_path: str = "repaircheck_final_dataset.xlsx") -> RepairCheckRetriever:
    """
    Returns the singleton instance of RepairCheckRetriever.
    Initializes on first call.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RepairCheckRetriever(dataset_path=dataset_path)
    return _retriever_instance


def retrieve(
    query: str,
    k: int = 3,
    brand: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to perform retrieval directly.
    Can be imported and called by Segment 3 diagnose(query) function:
    
        from retriever import retrieve
        matches = retrieve('Voltas E1', k=3)
    """
    return get_retriever().retrieve(query=query, k=k, brand=brand)
