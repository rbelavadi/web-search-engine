from posting import build_partial_index
from pagerank import compute_pagerank
from merge import merge_index

total_docs = build_partial_index()
compute_pagerank()

merge_index(total_docs)
