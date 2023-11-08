import copy
import os
import random
from typing import Optional, List, Dict, Iterable
import igraph
import csv
import abc
from collections import Counter


def validate_graph_exists(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        if args[0]._graph is None:
            raise Exception("Graph not initialized")
        return func(*args, **kwargs)

    return wrapper


class GraphInput(abc.ABC):
    def __init__(self):
        self._graph: Optional[igraph.Graph] = None

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
    def compute_clustering_coefficient(self) -> (float, Dict[float, int]):
        # Calculate the average clustering coefficient (C_avg) of a graph and its distribution.
        # C_avg = (1 / n) * Σ Ci
        # Ci = (2 * Ei) / (ki * (ki - 1))
        # clustering_coefficient = 2 * number of edges between neighbors of i / ((degree of i) * (degree of i-1))

        total_number_of_nodes = len(self._graph.vs)
        total_clustering_coefficient = 0.0
        clustering_coefficients = (
            []
        )  # List to store clustering coefficients for all nodes

        for node in self._graph.vs:
            neighbors = node.neighbors()
            number_of_neighbors = len(neighbors)
            if number_of_neighbors < 2:
                continue  # Skip nodes with a degree less than 2 to avoid division by zero

            edges_between_neighbors = self.count_edges_between_neighbors(
                node, neighbors
            )

            # Calculate clustering_coefficient for each node
            clustering_coefficient = (2 * edges_between_neighbors) / (
                    number_of_neighbors * (number_of_neighbors - 1)
            )
            total_clustering_coefficient += clustering_coefficient

            # Append the clustering coefficient to the list
            clustering_coefficients.append(clustering_coefficient)

        # Calculate average clustering_coefficient
        average_clustering_coefficient = (
                total_clustering_coefficient / total_number_of_nodes
        )

        # Calculate the distribution of clustering coefficients
        clustering_coefficient_distribution = Counter(clustering_coefficients)

        return average_clustering_coefficient, clustering_coefficient_distribution

    def count_edges_between_neighbors(self, node, neighbors):
        edge_count = 0

        for neighbor in neighbors:
            if self._graph.are_connected(node.index, neighbor.index):
                edge_count += 1

        return edge_count

    @validate_graph_exists
    def compute_cpl_for_all_nodes(self) -> List[float]:
        cpls = []
        for node in self._graph.vs:
            cpl = self.compute_characteristic_path_length(node)
            cpls.append(cpl)
        return cpls

    @validate_graph_exists
    def compute_characteristic_path_length(self, node) -> float:
        shortest_paths = self._graph.shortest_paths(node, mode=igraph.ALL)
        cpl_values = [sum(sp) for sp in shortest_paths if sp != float("inf")]

        return sum(cpl_values) / len(cpl_values) if cpl_values else 0.0

    @validate_graph_exists
    def compute_cohesiveness_for_each_node(self) -> Dict[int, int]:
        cohesiveness_values = {}

        for node in self._graph.vs:
            # Calculate k(G) (original edge connectivity)
            edge_connectivity_G = self._graph.edge_disjoint_paths()

            # Remove the vertex from the graph and calculate k(G - v)
            subgraph = self._graph.copy()
            subgraph.delete_vertices(node.index)
            edge_connectivity_G_minus_v = subgraph.edge_disjoint_paths()

            # Calculate cohesiveness c(v) = k(G) - k(G - v)
            cohesiveness = edge_connectivity_G - edge_connectivity_G_minus_v

            cohesiveness_values[node.index] = cohesiveness

        return cohesiveness_values

    @validate_graph_exists
    def compute_edge_persistence_under_greedy_attack(self, runs_without_result=10, total_number_of_runs=100) -> Dict:
        graph = copy.deepcopy(self._graph)
        diameters = {}
        diameters[0] = graph.diameter()
        for i in range(total_number_of_runs * runs_without_result):
            degrees = graph.degree()
            index_of_node_with_max_degree = degrees.index(max(degrees))
            graph.delete_vertices(index_of_node_with_max_degree)
            proportional_removed_nodes = 1 - (len(graph.vs) / len(self._graph.vs))
            if i % runs_without_result == 1:
                diameters[proportional_removed_nodes] = graph.diameter()
        return diameters

    @validate_graph_exists
    def compute_edge_persistence_under_random_attack(self, runs_without_result=10, total_number_of_runs=100) -> Dict:
        graph = copy.deepcopy(self._graph)
        diameters = {}
        random.seed(42)
        diameters[0] = graph.diameter()
        for i in range(total_number_of_runs * runs_without_result):
            degrees = graph.degree()
            random_node = random.randrange(0, len(degrees))
            graph.delete_vertices(random_node)
            proportional_removed_nodes = 1 - (len(graph.vs) / len(self._graph.vs))
            if i % runs_without_result == 1:
                diameters[proportional_removed_nodes] = graph.diameter()
        return diameters


class CsvFileGraphInput(GraphInput):
    def __init__(self):
        super().__init__()

    def read_input_file_and_convert(self, filename, directed=False) -> None:
        edges = []

        with open(filename, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                node_1, node_2 = int(row[0]), int(row[1])
                edges.append((node_1, node_2))

        self._graph = igraph.Graph(edges=edges, directed=directed)
        self._graph.delete_vertices(
            [v.index for v in self._graph.vs if v.degree() == 0]
        )


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
