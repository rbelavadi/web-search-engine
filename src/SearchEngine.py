import time
import pickle
import os

MERGED_INDEX_PATH = "index/merged_index.txt"
OFFSET_INDEX_PATH = "index/offset_index.pkl"
DOC_ID_MAP_PATH = "index/doc_id_map.pkl"

class SearchEngine:
    def __init__(self):
        with open(OFFSET_INDEX_PATH, "rb") as f:
            self.offsets = pickle.load(f)

        with open(DOC_ID_MAP_PATH, "rb") as f:
            self.doc_id_map = pickle.load(f)

    def search(self, query):
        start_time = time.time()

        results = []
        query_terms = query.lower().split()

        with open(MERGED_INDEX_PATH, "r") as file:
            for term in query_terms:
                if not term in self.offsets:
                    continue

                file.seek(self.offsets[term])
                line = file.readline().strip()
                if not line.startswith(term):
                    continue
                
                parts = line.split()
                postings = parts[1:]
                for p in postings:
                    doc_id, score = p.split(":")
                    results.append((int(doc_id), float(score)))

        results.sort(key=lambda x: -x[1])
        end_time = time.time()

        print(f"{query}:  Query took {end_time - start_time:.4f} seconds")
        return results
    
    def print_results(self, query, top_k = 10):
        results = self.search(query)
        if not results:
            print(f"No results were found for {query}")
        else:
            for rank, (doc_id, score) in enumerate(results[:top_k], 1):
                url = self.doc_id_map[doc_id]
                print(f"{rank}. {url} (score: {score:.4f})")
