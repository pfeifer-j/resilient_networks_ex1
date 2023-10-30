import abc

import igraph
from igraph import *


class GraphInput(abc.ABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._graph = None

    @staticmethod
    def read_input_file_and_convert(filename) -> igraph.Graph:
        pass


class CsvFileGraphInput(GraphInput):
    def __init__(self):
        super().__init__()

    def read_input_file_and_convert(self, filename, directed=False) -> None:
        return igraph.Graph.Read_Ncol(filename, directed=directed)


class TxtFileGraphInput(GraphInput):
    def __init__(self):
        super().__init__()

    def read_input_file_and_convert(self, filename, directed=False) -> None:
        edges = [(int(line.split()[0]), int(line.split()[1])) for line in open(filename) if not line.startswith("#")]
        self._graph = igraph.Graph(edges=edges, directed=directed)
        self._graph.delete_vertices([v.index for v in self._graph.vs if v.degree() == 0])
