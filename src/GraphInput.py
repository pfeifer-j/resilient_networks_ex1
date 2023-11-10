import copy
import random
from typing import Optional, List, Dict, Iterable
import igraph
import csv
import abc
from collections import Counter


# Checks for the initialization of a graph
def validate_graph_exists(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        if args[0]._graph is None:
            raise Exception("Graph not initialized")
        return func(*args, **kwargs)

    return wrapper


# This class contains the calculation of the metrics
class GraphInput(abc.ABC):
    def __init__(self):
        self._graph: Optional[igraph.Graph] = None

    @abc.abstractmethod
    def read_input_file_and_convert(self, filename) -> igraph.Graph:
        pass

    # Metric 1: Density
    @validate_graph_exists
    def computed_density(self) -> float:
        return self._graph.density()

    # Metric 2: Average path length
    @validate_graph_exists
    def compute_average_path_length(self) -> float:
        return self._graph.average_path_length()

    # Metric 3, 4: Clustering coefficient and its distribution
    @validate_graph_exists
    def compute_clustering_coefficient(self) -> (float, Dict[float, int]):
        total_number_of_nodes = len(self._graph.vs)
        total_clustering_coefficient = 0.0
        clustering_coefficients = []

        for node in self._graph.vs:
            neighbors = node.neighbors()
            number_of_neighbors = len(neighbors)
            if number_of_neighbors < 2:
                continue

            edges_between_neighbors = self.count_edges_between_neighbors(
                node, neighbors
            )

            # Calculate clustering_coefficient for each node
            clustering_coefficient = (2 * edges_between_neighbors) / (
                number_of_neighbors * (number_of_neighbors - 1)
            )
            total_clustering_coefficient += clustering_coefficient
            clustering_coefficients.append(clustering_coefficient)

        # Calculate average clustering_coefficient
        average_clustering_coefficient = (
            total_clustering_coefficient / total_number_of_nodes
        )
        clustering_coefficient_distribution = Counter(clustering_coefficients)

        return average_clustering_coefficient, clustering_coefficient_distribution

    @staticmethod
    def count_edges_between_neighbors(self, node, neighbors):
        edge_count = 0
        for neighbor in neighbors:
            if self._graph.are_connected(node.index, neighbor.index):
                edge_count += 1

        return edge_count

    # Metric 5: Distribution of the cohesiveness
    def compute_cohesiveness_distribution(self) -> Dict[int, int]:
        cohesiveness_values = []

        for vertex in self._graph.vs:
            cohesiveness_values.append(
                self.cohesiveness_for_a_node(self._graph, vertex.index)
            )

        cohesiveness_distribution = dict(Counter(cohesiveness_values))

        return cohesiveness_distribution

    @staticmethod
    def cohesiveness_for_a_node(graph, node):
        # Calculate k(G)
        connectivity_of_G = graph.vertex_connectivity()

        # Remove the vertex v
        modified_graph = graph.copy()
        modified_graph.delete_vertices(node)

        # Calculate k(G-v)
        connectivity_of_G_minus_v = modified_graph.vertex_connectivity()

        # Calculate cohesiveness c(v)
        cohesiveness = connectivity_of_G - connectivity_of_G_minus_v

        return cohesiveness

    # Metric 6: Edge persistence under greedy attack
    @validate_graph_exists
    def compute_edge_persistence_under_greedy_attack(self) -> int:
        graph = copy.deepcopy(self._graph)
        diameter = graph.diameter()
        removed_nodes = 0
        while graph.diameter() == diameter:
            degrees = graph.degree()
            index_of_node_with_max_degree = degrees.index(max(degrees))
            graph.delete_vertices(index_of_node_with_max_degree)
            removed_nodes += 1
        return removed_nodes

    # Metric 7: Edge persistence under random attack
    @validate_graph_exists
    def compute_edge_persistence_under_random_attack(self) -> int:
        graph = copy.deepcopy(self._graph)
        random.seed(42)
        diameter = graph.diameter()
        removed_nodes = 0
        while diameter == graph.diameter():
            degrees = graph.degree()
            random_node = random.randrange(0, len(degrees))
            graph.delete_vertices(random_node)
            removed_nodes += 1
        return removed_nodes


# CSV File Reader
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


# Text File Reader
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
