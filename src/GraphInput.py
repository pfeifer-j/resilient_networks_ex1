import os
from typing import Optional
import igraph
import abc


def validate_graph_exists(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        if args[0]._graph is None:
            raise Exception("Graph not initialized")
        return func(*args, **kwargs)

    return wrapper


class GraphInput(abc.ABC):
    def __init__(self):
        self._graph: Optional[Graph] = None

    @abc.abstractmethod
    def read_input_file_and_convert(self, filename) -> igraph.Graph:
        pass

    @validate_graph_exists
    def computed_density(self) -> float:
        return self._graph.density()

    @validate_graph_exists
    def compute_average_path_length(self) -> float:
        return self._graph.average_path_length()

    @validate_graph_exists
    def compute_average_clustering_coefficient(self) -> float:
        # The average clustering coefficient (C_avg) of a graph can be calculated using the following formula:
        # C_avg = (1 / n) * Σ Ci

        # To calculate the clustering coefficient (Ci) for an individual node, you can use the following formula:
        # Ci = (2 * Ei) / (ki * (ki - 1))
        # clustering_coefficient = 2 * number of edges between neighbors of i / ((degree of i) * (degree of i-1))

        total_number_of_nodes = len(self._graph.vs)
        total_clustering_coefficient = 0.0

        for node in self._graph.vs:
            neighbors = node.neighbors()
            number_of_neighbors = len(neighbors)
            if number_of_neighbors < 2:
                continue  # Skip nodes with degree less than 2 to avoid division by zero

            edges_between_neighbors = self.count_edges_between_neighbors(
                node, neighbors
            )

            # Calculate clustering_coefficient for each node
            clustering_coefficient = (2 * edges_between_neighbors) / (
                number_of_neighbors * (number_of_neighbors - 1)
            )
            total_clustering_coefficient += clustering_coefficient

        # Calculate average clustering_coefficient
        average_clustering_coefficient = (
            total_clustering_coefficient / total_number_of_nodes
        )
        return average_clustering_coefficient

    def count_edges_between_neighbors(self, node, neighbors):
        edge_count = 0

        for neighbor in neighbors:
            if self._graph.are_connected(node.index, neighbor.index):
                edge_count += 1

        return edge_count


class CsvFileGraphInput(GraphInput):
    def __init__(self):
        super().__init__()

    def read_input_file_and_convert(self, filename, directed=False) -> None:
        return igraph.Graph.Read_Ncol(filename, directed=directed)


class TxtFileGraphInput(GraphInput):
    def __init__(self):
        super().__init__()

    def read_input_file_and_convert(self, filename, directed=False) -> None:
        edges = [
            (int(line.split()[0]), int(line.split()[1]))
            for line in open(filename)
            if not line.startswith("#")
        ]
        self._graph = igraph.Graph(edges=edges, directed=directed)
        self._graph.delete_vertices(
            [v.index for v in self._graph.vs if v.degree() == 0]
        )
