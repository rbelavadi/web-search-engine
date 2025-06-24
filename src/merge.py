import os
import pickle
import math
from collections import defaultdict

PARTIAL_DIR = "index/"
MERGED_INDEX = "index/merged_index.txt"
OFFSET_INDEX = "index/offset_index.pkl"

def merge_index(N):
    with open("index/page_rank.pkl", "rb") as f:
        page_rank = pickle.load(f)

    partial_files = sorted(
        [os.path.join(PARTIAL_DIR, f) for f in os.listdir(PARTIAL_DIR)
         if f.startswith("partial_index_") and f.endswith(".pkl")]
    )

    partial_indexes = []
    for path in partial_files:
        with open(path, "rb") as f:
            partial_indexes.append(pickle.load(f))

    all_terms = sorted({term for p in partial_indexes for term in p.keys()})

    offsets = {}
    with open(MERGED_INDEX, "w", encoding="utf-8") as out:
        for term in all_terms:
            doc_freqs = defaultdict(int)

            for p_index in partial_indexes:
                if term in p_index:
                    for entry in p_index[term]:
                        doc_id = entry["doc_id"]
                        freq = entry["freq"]
                        boost = entry.get("boost", 1.0)
                        doc_freqs[doc_id] += freq * boost

            df = len(doc_freqs)
            idf = math.log10(N / df) if df > 0 else 0

            tfidf_postings = []
            for doc_id, boosted_freq in doc_freqs.items():
                tf = 1 + math.log10(boosted_freq)
                tfidf = tf * idf
                pr_score = page_rank.get(doc_id, 0)
                alpha = 0.1
                final_score = tfidf + alpha * pr_score
                tfidf_postings.append((doc_id, round(final_score, 6)))


            tfidf_postings.sort()
            offsets[term] = out.tell()

            posting_str = " ".join(f"{doc_id}:{score}" for doc_id, score in tfidf_postings)
            out.write(f"{term} {posting_str}\n")

    with open(OFFSET_INDEX, "wb") as f:
        pickle.dump(offsets, f)