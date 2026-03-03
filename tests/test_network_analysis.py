# tests/test_network_analysis.py

import pytest
from analysis.network_analysis import CoauthorNetworkAnalyzer


class TestCoauthorNetworkAnalyzer:

    def test_init(self):
        analyzer = CoauthorNetworkAnalyzer()
        assert analyzer is not None

    def test_build_graph_empty(self):
        analyzer = CoauthorNetworkAnalyzer()
        G = analyzer.build_graph([])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_find_bridge_researchers_empty(self):
        import networkx as nx
        analyzer = CoauthorNetworkAnalyzer()
        G = nx.Graph()
        result = analyzer.find_bridge_researchers(G, top_n=10)
        assert result == []
