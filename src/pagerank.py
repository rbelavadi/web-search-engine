import pickle

DAMPING = 0.85
ITERATIONS = 20

def compute_pagerank():
    with open("index/link_graph.pkl", "rb") as f:
        link_graph = pickle.load(f)

    N = len(link_graph)
    ranks = {doc: 1.0 / N for doc in link_graph}
    for _ in range(ITERATIONS):
        new_ranks = {}
        for doc in link_graph:
            rank_sum = sum(ranks[other] / len(link_graph[other]) for other in link_graph if doc in link_graph[other])
            new_ranks[doc] = (1 - DAMPING) / N + DAMPING * rank_sum
        ranks = new_ranks

    with open("index/page_rank.pkl", "wb") as f:
        pickle.dump(ranks, f)

