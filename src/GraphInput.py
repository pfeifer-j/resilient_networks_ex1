import abc

import igraph
from igraph import *


def validate_graph_exists(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        if args[0]._graph is None:
            raise Exception("Graph not initialized")
        return func(*args, **kwargs)

    return wrapper


class GraphInput(abc.ABC):
    def __init__(self):
        self._graph = None

    @abc.abstractmethod
    def read_input_file_and_convert(self, filename) -> igraph.Graph:
        pass

    @validate_graph_exists
    def computed_density(self) -> float:
        return self._graph.density()

    @validate_graph_exists
    def compute_average_path_length(self) -> float:
        return self._graph.average_path_length()


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
