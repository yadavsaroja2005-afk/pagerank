import numpy as np

def page_rank(graph, damping_factor=0.85, max_iterations=100, tolerance=1e-6):

    # Number of pages (nodes)
    num_nodes = len(graph)

    # Initialize PageRank values equally
    page_ranks = np.ones(num_nodes) / num_nodes

    # Iterative computation
    for _ in range(max_iterations):

        prev_page_ranks = page_ranks.copy()

        for node in range(num_nodes):

            # Find incoming links
            incoming_links = [i for i in range(num_nodes) if node in graph[i]]

            # Base PageRank value
            rank_sum = 0

            for link in incoming_links:
                rank_sum += prev_page_ranks[link] / len(graph[link])

            page_ranks[node] = ((1 - damping_factor) / num_nodes) + (damping_factor * rank_sum)

        # Check convergence
        if np.linalg.norm(page_ranks - prev_page_ranks) < tolerance:
            break

    return page_ranks


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    # Directed graph (Adjacency list)
    web_graph = [
        [1, 2],   # Page 0 links to Page 1 and 2
        [0, 2],   # Page 1 links to Page 0 and 2
        [0, 1],   # Page 2 links to Page 0 and 1
        [1, 2]    # Page 3 links to Page 1 and 2
    ]

    # Calculate PageRank
    ranks = page_rank(web_graph)

    # Print results
    print("PageRank Values:\n")

    for i, rank in enumerate(ranks):
        print(f"Page {i}: {rank:.4f}")
