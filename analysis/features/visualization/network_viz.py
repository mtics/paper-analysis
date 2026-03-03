"""Interactive network visualization using PyVis."""

import logging
from pathlib import Path
from typing import Optional, List, Any
import warnings

logger = logging.getLogger(__name__)

# Try to import pyvis
PYVIS_AVAILABLE = False
Net = None
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
    Net = Network
except ImportError:
    warnings.warn("pyvis not available, network visualization will be skipped")
    logger.warning("pyvis not available")


class NetworkViz:
    """Generate interactive network visualizations using PyVis."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("output/visualizations/network")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pyvis_available = PYVIS_AVAILABLE
        self.Net = Net if PYVIS_AVAILABLE else None

    def plot_coauthor_network(
        self,
        G,
        output_name: str = "coauthor_network",
        min_weight: int = 2,
        max_nodes: int = 500,
        show_weights: bool = True
    ) -> Optional[Path]:
        """Plot coauthor network as interactive HTML.

        Args:
            G: NetworkX graph with 'weight' edge attribute
            output_name: Output file name (without extension)
            min_weight: Minimum edge weight to show
            max_nodes: Maximum number of nodes to display (top by degree)
            show_weights: Whether to show edge weights

        Returns:
            Path to saved HTML file, or None if skipped
        """
        if not self.pyvis_available:
            logger.warning("pyvis not available, skipping network visualization")
            return None

        # Filter edges by weight
        edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w < min_weight]
        G_filtered = G.copy()
        G_filtered.remove_edges_from(edges_to_remove)

        # Remove isolated nodes
        G_filtered.remove_nodes_from(list(__import__('networkx').isolates(G_filtered)))

        # If too many nodes, keep top by degree
        if G_filtered.number_of_nodes() > max_nodes:
            degrees = dict(G_filtered.degree())
            top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:max_nodes]
            nodes_to_remove = set(G_filtered.nodes()) - set(top_nodes)
            G_filtered.remove_nodes_from(nodes_to_remove)
            logger.info(f"Network reduced to {max_nodes} nodes for visualization")

        # Create PyVis network
        net = self.Net(
            height="600px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#333333",
            directed=False
        )

        # Add nodes
        for node in G_filtered.nodes():
            degree = G_filtered.degree(node)
            # Scale node size by degree
            size = 10 + min(degree * 2, 30)
            net.add_node(
                node,
                label=node[:20],  # Truncate long names
                title=f"{node}\nConnections: {degree}",
                size=size
            )

        # Add edges
        for u, v, data in G_filtered.edges(data=True):
            weight = data.get('weight', 1)
            net.add_edge(
                u,
                v,
                value=weight,
                title=f"Co-authored papers: {weight}"
            )

        # Set physics options for better layout
        net.set_options("""
        {
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 4
            },
            "edges": {
                "color": {
                    "inherit": "both"
                },
                "smooth": false
            },
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                    "enabled": true,
                    "iterations": 200
                }
            }
        }
        """)

        # Save HTML
        output_path = self.output_dir / f"{output_name}.html"
        net.save_graph(str(output_path))

        logger.info(f"Saved interactive network to {output_path}")
        return output_path

    def export_network_json(
        self,
        G,
        output_name: str = "network_data"
    ) -> Optional[Path]:
        """Export network data as JSON for external visualization tools.

        Args:
            G: NetworkX graph
            output_name: Output file name

        Returns:
            Path to saved JSON file
        """
        import json

        nodes = []
        for node in G.nodes():
            nodes.append({
                "id": node,
                "label": node[:20],
                "degree": G.degree(node)
            })

        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": data.get('weight', 1)
            })

        data = {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges()
            }
        }

        output_path = self.output_dir / f"{output_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved network JSON to {output_path}")
        return output_path
