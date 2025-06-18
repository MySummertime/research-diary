# Projects/Risk_Assessment_Models/backend/core/graph.py

from typing import List, Dict, Type, TypeVar, Set, Tuple, Optional

# A type variable for proper type hinting of the class method 'from_json'
G = TypeVar('G', bound='Graph')


class Node:
    """
    Represents a node in the graph, matching the provided JSON structure.
    """

    def __init__(self, id: int, label: str, x: float, y: float, prob: float, cons: float):
        """Initializes a Node object."""
        self.id: int = id
        self.label: str = label
        self.x: float = x
        self.y: float = y
        self.prob: float = prob
        self.cons: float = cons

    def __repr__(self) -> str:
        return f"Node(id={self.id}, label='{self.label}')"


class Edge:
    """
    Represents a structured, directed edge in the graph.
    """

    def __init__(self, source: int, target: int, prob: float, cons: float, length: float):
        """Initializes an Edge object."""
        self.source: int = source
        self.target: int = target
        self.prob: float = prob
        self.cons: float = cons
        self.length: float = length

    def __repr__(self) -> str:
        return f'Edge(source={self.source}, target={self.target})'


class Graph:
    """
    Represents the graph structure, compatible with both directed and undirected graphs.
    """

    def __init__(self, directed: bool = True):
        """
        Initializes a Graph object.

        Args:
            directed: If True, the graph is treated as a directed graph.
                      If False, adding an edge (A, B) will also implicitly
                      create an edge (B, A), making it undirected.
        """
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
        self.directed: bool = directed
        self._adj: Optional[Dict[int, List[Edge]]] = None

    @property
    def adjacency_list(self) -> Dict[int, List[Edge]]:
        """
        Provides an adjacency list representation of the graph.

        The list is generated on first access and then cached for efficiency.
        It maps each node ID to a list of its outgoing Edge objects.
        """
        if self._adj is None:
            # Build the adjacency list if it hasn't been built yet
            self._adj = {node_id: [] for node_id in self.nodes}
            for edge in self.edges:
                if edge.source in self._adj:
                    self._adj[edge.source].append(edge)
        return self._adj

    def add_node(self, node: Node) -> None:
        """
        Adds a Node object to the graph.
        """
        self.nodes[node.id] = node
        self._adj = None  # Invalidate cache when nodes change

    def add_edge(self, edge: Edge) -> None:
        """
        Adds an edge to the graph. If the graph is undirected,
        a symmetric reverse edge is also added automatically.
        """
        self.edges.append(edge)
        self._adj = None  # Invalidate cache when nodes change

        if not self.directed:
            reverse_edge = Edge(
                source=edge.target,
                target=edge.source,
                prob=edge.prob,
                cons=edge.cons,
                length=edge.length,
            )
            self.edges.append(reverse_edge)

    @classmethod
    def from_json(cls: Type[G], data: Dict) -> G:
        """
        Creates a Graph instance from a JSON-like dictionary structure,
        automatically inferring if the graph is directed or undirected.
        """
        # --- Graph Type Inference Logic ---
        is_directed_graph = False
        edge_tuples: Set[Tuple[int, int]] = set()

        if 'edges' in data:
            for edge_data in data['edges']:
                edge_tuples.add((edge_data['source'], edge_data['target']))

        # Iterating over the set to check for symmetry
        for u, v in edge_tuples:
            if (v, u) not in edge_tuples:
                is_directed_graph = True
                break

        # Instantiate the graph with the inferred type
        graph_instance = cls(directed=is_directed_graph)

        # --- Populate Graph ---
        if 'nodes' in data:
            for node_data in data['nodes']:
                graph_instance.add_node(Node(**node_data))

        processed_undirected_edges: Set[Tuple[int, int]] = set()

        if 'edges' in data:
            for edge_data in data['edges']:
                if not graph_instance.directed:
                    edge_pair = tuple(sorted((edge_data['source'], edge_data['target'])))
                    if edge_pair in processed_undirected_edges:
                        continue  # This symmetric pair has already been processed.
                    processed_undirected_edges.add(edge_pair)
                graph_instance.add_edge(Edge(**edge_data))

        return graph_instance

    def __repr__(self) -> str:
        graph_type = 'Directed' if self.directed else 'Undirected'
        return f'{graph_type} Graph(nodes={len(self.nodes)}, edges={len(self.edges)})'
