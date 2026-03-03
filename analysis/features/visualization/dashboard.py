"""HTML dashboard generator."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Generate HTML dashboard combining all visualizations."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("output/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.charts_dir = self.output_dir / "charts"
        self.network_dir = self.output_dir / "network"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.network_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: Dict[str, Any],
        papers: List,
        timestamp: Optional[str] = None
    ) -> Path:
        """Generate complete dashboard HTML.

        Args:
            results: Full analysis results dictionary
            papers: List of paper objects
            timestamp: Optional timestamp for the report

        Returns:
            Path to generated dashboard HTML
        """
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        # First generate charts
        chart_files = self._generate_charts(results, papers)

        # Generate network visualization
        network_file = self._generate_network(results, papers)

        # Generate HTML dashboard
        html_path = self._generate_html(results, chart_files, network_file, timestamp)

        logger.info(f"Dashboard generated: {html_path}")
        return html_path

    def _generate_charts(
        self,
        results: Dict[str, Any],
        papers: List
    ) -> Dict[str, str]:
        """Generate static charts."""
        chart_files = {}

        # Try matplotlib charts
        try:
            from analysis.features.visualization.charts import TrendCharts
            charts = TrendCharts(self.charts_dir)

            # Yearly distribution
            yearly_data = results.get('yearly_distribution', {})
            if yearly_data:
                path = charts.plot_yearly_distribution(yearly_data)
                if path:
                    chart_files['yearly_distribution'] = path.name

            # Venue distribution
            venue_data = results.get('venue_distribution', {})
            if venue_data:
                path = charts.plot_venue_distribution(venue_data)
                if path:
                    chart_files['venue_distribution'] = path.name

            # Keyword trends - use domain-specific keywords if available
            keyword_data = {}
            domain_analysis = results.get('domain_analysis', {})

            # Try to get keywords from first domain analysis
            if domain_analysis:
                first_domain = list(domain_analysis.keys())[0]
                domain_data = domain_analysis[first_domain]
                if 'top_keywords' in domain_data and domain_data['top_keywords']:
                    # Convert list of [keyword, count] pairs to dict
                    for kw, count in domain_data['top_keywords'][:30]:
                        keyword_data[kw] = count

            # Fallback to generic keywords if no domain keywords
            if not keyword_data:
                keyword_data = results.get('top_keywords', {})

            if keyword_data:
                path = charts.plot_keyword_trends(keyword_data, top_n=20)
                if path:
                    chart_files['keyword_trends'] = path.name

            # Venue-year heatmap
            if papers:
                path = charts.plot_venue_year_heatmap(papers)
                if path:
                    chart_files['venue_year_heatmap'] = path.name

            # Domain lifecycle charts
            domain_analysis = results.get('domain_analysis', {})
            for domain, data in domain_analysis.items():
                if 'lifecycle' in data and 'yearly_trends' in data:
                    path = charts.plot_lifecycle_scurve(
                        data['yearly_trends'],
                        data['lifecycle'],
                        output_name=f"lifecycle_{domain.replace(' ', '_')[:20]}"
                    )
                    if path:
                        chart_files[f'lifecycle_{domain}'] = path.name

            logger.info(f"Generated {len(chart_files)} charts")

        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")

        return chart_files

    def _generate_network(
        self,
        results: Dict[str, Any],
        papers: List
    ) -> Optional[str]:
        """Generate network visualization."""
        try:
            from analysis.features.visualization.network_viz import NetworkViz
            from analysis.features.network import CoauthorNetworkAnalyzer

            viz = NetworkViz(self.network_dir)

            # Build network
            analyzer = CoauthorNetworkAnalyzer()
            G = analyzer.build_graph(papers)

            if G.number_of_nodes() > 0:
                # Generate interactive HTML
                path = viz.plot_coauthor_network(
                    G,
                    min_weight=3,
                    max_nodes=300
                )
                if path:
                    # Also export JSON
                    viz.export_network_json(G)
                    return path.name

        except Exception as e:
            logger.warning(f"Network visualization failed: {e}")

        return None

    def _generate_html(
        self,
        results: Dict[str, Any],
        chart_files: Dict[str, str],
        network_file: Optional[str],
        timestamp: str
    ) -> Path:
        """Generate HTML dashboard."""

        # Helper function to render sections
        def render_chart_section(title: str, chart_file: Optional[str]) -> str:
            if not chart_file:
                return ""
            return f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="chart-item">
                <img src="charts/{chart_file}" alt="{title}">
            </div>
        </div>
            """

        def render_network_section(net_file: Optional[str]) -> str:
            if not net_file:
                return ""
            return f"""
        <div class="section">
            <h2>Author Collaboration Network</h2>
            <div class="network-container">
                <iframe src="network/{net_file}" title="Author Network"></iframe>
            </div>
        </div>
            """

        def render_domain_analysis(domain_analysis: Dict[str, Any]) -> str:
            if not domain_analysis:
                return "<p>No domain analysis results.</p>"

            html = '<div class="chart-grid">'

            for domain, data in domain_analysis.items():
                if 'error' in data:
                    continue

                papers = data.get('total_papers', 0)
                lifecycle = data.get('lifecycle', {}).get('stage', 'unknown')

                html += f"""
            <div class="chart-item">
                <h3>{domain}</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="value">{papers}</div>
                        <div class="label">Papers</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">{lifecycle}</div>
                        <div class="label">Stage</div>
                    </div>
                </div>
            </div>
                """

            html += '</div>'
            return html

        # Generate chart sections HTML
        yearly_section = render_chart_section("Yearly Distribution", chart_files.get('yearly_distribution'))
        venue_section = render_chart_section("Conference Distribution", chart_files.get('venue_distribution'))
        keyword_section = render_chart_section("Top Keywords", chart_files.get('keyword_trends'))
        heatmap_section = render_chart_section("Venue-Year Heatmap", chart_files.get('venue_year_heatmap'))
        network_section = render_network_section(network_file)
        domain_html = render_domain_analysis(results.get('domain_analysis', {}))

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCF-A Paper Analysis Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        .header p {{
            opacity: 0.9;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #1a365d;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        .stat-card {{
            background: #f7fafc;
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: #2c5282;
        }}
        .stat-card .label {{
            color: #718096;
            font-size: 0.875rem;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }}
        .chart-item {{
            background: #f7fafc;
            border-radius: 6px;
            padding: 1rem;
        }}
        .chart-item img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .chart-item h3 {{
            margin-bottom: 0.5rem;
            color: #2d3748;
        }}
        .network-container {{
            background: #f7fafc;
            border-radius: 6px;
            padding: 1rem;
        }}
        .network-container iframe {{
            width: 100%;
            height: 600px;
            border: none;
            border-radius: 4px;
        }}
        .keyword-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .keyword-tag {{
            background: #e2e8f0;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.875rem;
        }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #718096;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CCF-A Conference Paper Analysis Dashboard</h1>
        <p>Generated: {timestamp} | Analysis Period: {results.get('years', 'N/A')}</p>
    </div>

    <div class="container">
        <!-- Overview Stats -->
        <div class="section">
            <h2>Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="value">{results.get('total_papers', 0):,}</div>
                    <div class="label">Total Papers</div>
                </div>
                <div class="stat-card">
                    <div class="value">{len(results.get('conferences', []))}</div>
                    <div class="label">Conferences</div>
                </div>
                <div class="stat-card">
                    <div class="value">{len(results.get('years', []))}</div>
                    <div class="label">Years</div>
                </div>
            </div>
        </div>

        {yearly_section}
        {venue_section}
        {keyword_section}
        {heatmap_section}

        <!-- Domain Analysis -->
        <div class="section">
            <h2>Domain Analysis</h2>
            {domain_html}
        </div>

        {network_section}

    </div>

    <div class="footer">
        <p>CCF-A Paper Analysis System | Powered by Python, NetworkX, Matplotlib, PyVis</p>
    </div>
</body>
</html>"""

        html_path = self.output_dir / "dashboard.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path
