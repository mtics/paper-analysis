"""Static chart generation using matplotlib/seaborn."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None
    warnings.warn("matplotlib/seaborn not available, charts will be skipped")
    logger.warning("matplotlib/seaborn not available")


class TrendCharts:
    """Generate static charts using matplotlib/seaborn."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("output/visualizations/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        self.plt = plt
        self.sns = sns

        if self.matplotlib_available:
            # Set style
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (12, 6)
            plt.rcParams['font.size'] = 10

    def plot_yearly_distribution(
        self,
        yearly_data: Dict[int, int],
        output_name: str = "yearly_distribution"
    ) -> Optional[Path]:
        """Plot yearly paper distribution as bar chart."""
        if not self.matplotlib_available:
            logger.warning("matplotlib not available, skipping yearly distribution chart")
            return None

        years = sorted(yearly_data.keys())
        counts = [yearly_data[y] for y in years]

        fig, ax = self.plt.subplots(figsize=(10, 6))
        bars = ax.bar(years, counts, color='steelblue', edgecolor='navy', alpha=0.8)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{count:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title('Yearly Paper Distribution (CCF-A Conferences)', fontsize=14, fontweight='bold')
        ax.set_xticks(years)

        self.plt.tight_layout()

        output_path = self.output_dir / f"{output_name}.png"
        self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.plt.close()

        logger.info(f"Saved yearly distribution chart to {output_path}")
        return output_path

    def plot_venue_distribution(
        self,
        venue_data: Dict[str, int],
        output_name: str = "venue_distribution"
    ) -> Optional[Path]:
        """Plot conference paper distribution as pie chart."""
        if not self.matplotlib_available:
            logger.warning("matplotlib not available, skipping venue distribution chart")
            return None

        # Sort and take top 10
        sorted_venues = sorted(venue_data.items(), key=lambda x: x[1], reverse=True)[:10]
        venues = [v[0].upper() for v in sorted_venues]
        counts = [v[1] for v in sorted_venues]

        # Use a color palette
        colors = sns.color_palette("husl", len(venues))

        fig, ax = self.plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=venues,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            pctdistance=0.75
        )

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)

        ax.set_title('Conference Paper Distribution (Top 10)', fontsize=14, fontweight='bold')

        self.plt.tight_layout()

        output_path = self.output_dir / f"{output_name}.png"
        self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.plt.close()

        logger.info(f"Saved venue distribution chart to {output_path}")
        return output_path

    def plot_keyword_trends(
        self,
        keyword_data: Dict[str, int],
        top_n: int = 20,
        output_name: str = "keyword_trends"
    ) -> Optional[Path]:
        """Plot top keywords as horizontal bar chart."""
        if not self.matplotlib_available:
            logger.warning("matplotlib not available, skipping keyword trends chart")
            return None

        # Sort and take top N
        sorted_keywords = sorted(keyword_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
        keywords = [k[0] for k in sorted_keywords]
        counts = [k[1] for k in sorted_keywords]

        fig, ax = self.plt.subplots(figsize=(12, max(8, top_n * 0.4)))
        y_pos = range(len(keywords))

        bars = ax.barh(y_pos, counts, color='teal', edgecolor='darkslategray', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(keywords)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title(f'Top {top_n} Keywords in CCF-A Papers', fontsize=14, fontweight='bold')

        # Add value labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.annotate(f'{count:,}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=8)

        self.plt.tight_layout()

        output_path = self.output_dir / f"{output_name}.png"
        self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.plt.close()

        logger.info(f"Saved keyword trends chart to {output_path}")
        return output_path

    def plot_venue_year_heatmap(
        self,
        papers: List,
        output_name: str = "venue_year_heatmap"
    ) -> Optional[Path]:
        """Plot conference-year heatmap showing paper counts."""
        if not self.matplotlib_available:
            logger.warning("matplotlib not available, skipping venue-year heatmap")
            return None

        from collections import defaultdict
        import pandas as pd
        import numpy as np

        # Collect data
        venue_year = defaultdict(lambda: defaultdict(int))
        years = set()
        venues = set()

        # Collect data from papers
        for paper in papers:
            venue = paper.venue.upper()
            year = paper.year
            venue_year[venue][year] += 1
            years.add(year)
            venues.add(venue)

        years = sorted(years)
        venues = sorted(venues)

        # Create matrix
        matrix = np.zeros((len(venues), len(years)))
        for i, venue in enumerate(venues):
            for j, year in enumerate(years):
                matrix[i, j] = venue_year[venue].get(year, 0)

        # Create DataFrame
        df = pd.DataFrame(matrix, index=venues, columns=years)

        fig, ax = self.plt.subplots(figsize=(max(10, len(years) * 1.2), max(6, len(venues) * 0.3)))

        sns.heatmap(
            df,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': 'Number of Papers'}
        )

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Conference', fontsize=12)
        ax.set_title('Conference-Year Paper Distribution', fontsize=14, fontweight='bold')

        self.plt.tight_layout()

        output_path = self.output_dir / f"{output_name}.png"
        self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.plt.close()

        logger.info(f"Saved venue-year heatmap to {output_path}")
        return output_path

    def plot_lifecycle_scurve(
        self,
        yearly_trends: Dict[int, int],
        lifecycle_data: Dict[str, Any],
        output_name: str = "lifecycle_scurve"
    ) -> Optional[Path]:
        """Plot S-curve lifecycle visualization."""
        if not self.matplotlib_available:
            logger.warning("matplotlib not available, skipping lifecycle chart")
            return None

        years = sorted(yearly_trends.keys())
        counts = [yearly_trends[y] for y in years]

        fig, ax = self.plt.subplots(figsize=(10, 6))

        # Plot actual data
        ax.plot(years, counts, 'o-', color='steelblue', linewidth=2, markersize=8, label='Actual')

        # Plot fitted S-curve if available
        if 'fitted_values' in lifecycle_data and lifecycle_data['fitted_values']:
            fitted = lifecycle_data['fitted_values']
            ax.plot(years, fitted, '--', color='coral', linewidth=2, label='S-Curve Fit')

        # Add stage annotation
        stage = lifecycle_data.get('stage', 'unknown')
        ax.annotate(
            f'Stage: {stage}',
            xy=(0.02, 0.98),
            xycoords='axes fraction',
            fontsize=12,
            fontweight='bold',
            va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title('Domain Lifecycle Analysis (S-Curve)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.plt.tight_layout()

        output_path = self.output_dir / f"{output_name}.png"
        self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.plt.close()

        logger.info(f"Saved lifecycle chart to {output_path}")
        return output_path

    # =========================================================================
    # 洞见可视化 (Insight-Generating Visualizations)
    # =========================================================================

    def plot_keyword_cooccurrence(
        self,
        papers: List,
        top_n: int = 50,
        output_name: str = "keyword_cooccurrence"
    ) -> Optional[Path]:
        """Plot keyword co-occurrence network - shows which keywords appear together."""
        if not self.matplotlib_available:
            logger.warning("matplotlib not available, skipping keyword co-occurrence")
            return None

        try:
            from collections import Counter
            import re
            from itertools import combinations

            # Extract keywords from papers
            keyword_counter = Counter()
            cooccurrence = Counter()

            for paper in papers:
                # Get keywords from paper
                keywords = set()
                if hasattr(paper, 'keywords') and paper.keywords:
                    for kw in paper.keywords:
                        if isinstance(kw, str):
                            keywords.add(kw.lower().strip())
                        elif isinstance(kw, dict):
                            kw_name = kw.get('keyword', kw.get('name', ''))
                            if kw_name:
                                keywords.add(kw_name.lower().strip())

                # Also extract from title if no keywords
                if not keywords and hasattr(paper, 'title') and paper.title:
                    words = re.findall(r'\b[a-z]{4,}\b', paper.title.lower())
                    # Filter common words
                    stopwords = {'from', 'with', 'that', 'this', 'using', 'base', 'method', 'approach', 'paper', 'work'}
                    words = [w for w in words if w not in stopwords]
                    keywords = set(words[:5])

                # Count individual keywords
                for kw in keywords:
                    keyword_counter[kw] += 1

                # Count co-occurrences
                keywords_list = list(keywords)[:15]  # Limit to avoid explosion
                for kw1, kw2 in combinations(sorted(keywords_list), 2):
                    cooccurrence[(kw1, kw2)] += 1

            # Get top keywords
            top_keywords = set([kw for kw, _ in keyword_counter.most_common(top_n)])

            # Build network data for visualization
            edges = []
            for (kw1, kw2), count in cooccurrence.items():
                if kw1 in top_keywords and kw2 in top_keywords and count >= 2:
                    edges.append((kw1, kw2, count))

            if not edges:
                logger.warning("No keyword co-occurrences found")
                return None

            # Create network visualization using matplotlib
            try:
                import networkx as nx
                G = nx.Graph()
                for kw, cnt in keyword_counter.most_common(top_n):
                    G.add_node(kw, size=cnt)
                for kw1, kw2, cnt in edges:
                    G.add_edge(kw1, kw2, weight=cnt)

                # Simple layout
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

                fig, ax = self.plt.subplots(figsize=(14, 12))

                # Node sizes based on frequency
                node_sizes = [keyword_counter.get(n, 10) * 10 for n in G.nodes()]
                node_sizes = [min(s, 2000) for s in node_sizes]  # Cap size

                # Edge widths based on co-occurrence
                edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
                edge_widths = [min(w * 0.5, 5) for w in edge_weights]

                # Draw
                nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_widths, ax=ax)
                nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

                ax.set_title(f'Keyword Co-occurrence Network (Top {top_n})', fontsize=14, fontweight='bold')
                ax.axis('off')

                self.plt.tight_layout()
                output_path = self.output_dir / f"{output_name}.png"
                self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
                self.plt.close()

                logger.info(f"Saved keyword co-occurrence network to {output_path}")
                return output_path
            except ImportError:
                logger.warning("networkx not available for co-occurrence")
                return None

        except Exception as e:
            logger.warning(f"Keyword co-occurrence failed: {e}")
            return None

    def plot_keyword_trend_comparison(
        self,
        yearly_keyword_data: Dict[int, Dict[str, int]],
        top_keywords: List[str] = None,
        output_name: str = "keyword_trend_comparison"
    ) -> Optional[Path]:
        """Plot keyword growth/decline comparison - shows how keywords change over time."""
        if not self.matplotlib_available:
            return None

        if not yearly_keyword_data or len(yearly_keyword_data) < 2:
            logger.warning("Need at least 2 years for trend comparison")
            return None

        years = sorted(yearly_keyword_data.keys())
        first_year = yearly_keyword_data[years[0]]
        last_year = yearly_keyword_data[years[-1]]

        # Calculate growth rate for each keyword
        growth_rates = {}
        for kw in set(first_year.keys()) | set(last_year.keys()):
            first_count = first_year.get(kw, 0)
            last_count = last_year.get(kw, 0)
            if first_count > 0:
                growth = ((last_count - first_count) / first_count) * 100
                growth_rates[kw] = {
                    'growth': growth,
                    'first': first_count,
                    'last': last_count
                }

        # Get top growing and declining keywords
        sorted_by_growth = sorted(growth_rates.items(), key=lambda x: x[1]['growth'], reverse=True)
        top_growing = [kw for kw, _ in sorted_by_growth[:8] if kw in (top_keywords or {})]
        top_declining = [kw for kw, _ in sorted_by_growth[-8:] if kw in (top_keywords or {})]

        if not top_growing and not top_declining:
            # Use all keywords if no filter
            top_growing = [kw for kw, _ in sorted_by_growth[:6]]
            top_declining = [kw for kw, _ in sorted_by_growth[-6:]]

        selected_keywords = list(set(top_growing + top_declining))[:10]

        # Plot
        fig, ax = self.plt.subplots(figsize=(12, 6))

        for kw in selected_keywords:
            values = [yearly_keyword_data.get(y, {}).get(kw, 0) for y in years]
            growth = growth_rates.get(kw, {}).get('growth', 0)
            color = 'green' if growth > 0 else 'red'
            ax.plot(years, values, 'o-', label=f'{kw} ({growth:+.0f}%)', linewidth=2, markersize=6)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Keyword Frequency', fontsize=12)
        ax.set_title(f'Keyword Trend Comparison ({years[0]} → {years[-1]})', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        self.plt.tight_layout()
        output_path = self.output_dir / f"{output_name}.png"
        self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.plt.close()

        logger.info(f"Saved keyword trend comparison to {output_path}")
        return output_path

    def plot_emerging_keywords(
        self,
        yearly_keyword_data: Dict[int, Dict[str, int]],
        min_growth: float = 50.0,
        output_name: str = "emerging_keywords"
    ) -> Optional[Path]:
        """Plot emerging keywords - shows rising stars in research."""
        if not self.matplotlib_available:
            return None

        if not yearly_keyword_data or len(yearly_keyword_data) < 2:
            logger.warning("Need at least 2 years for emerging keywords")
            return None

        years = sorted(yearly_keyword_data.keys())
        first_year_data = yearly_keyword_data[years[0]]
        last_year_data = yearly_keyword_data[years[-1]]

        # Find emerging keywords
        emerging = []
        all_keywords = set(first_year_data.keys()) | set(last_year_data.keys())

        for kw in all_keywords:
            first_count = first_year_data.get(kw, 0)
            last_count = last_year_data.get(kw, 0)

            # Must have reasonable presence in latest year
            if last_count < 10:
                continue

            # Calculate growth
            if first_count == 0:
                growth = 100.0 if last_count > 0 else 0
                abs_growth = last_count
            else:
                growth = ((last_count - first_count) / first_count) * 100
                abs_growth = last_count - first_count

            if growth >= min_growth:
                emerging.append({
                    'keyword': kw,
                    'growth': growth,
                    'count_2020': first_count,
                    'count_2025': last_count,
                    'abs_growth': abs_growth
                })

        if not emerging:
            logger.warning("No emerging keywords found")
            return None

        # Sort by absolute growth
        emerging = sorted(emerging, key=lambda x: x['abs_growth'], reverse=True)[:15]

        # Plot bubble chart
        fig, ax = self.plt.subplots(figsize=(12, 8))

        # Normalize sizes
        max_growth = max(e['growth'] for e in emerging)
        sizes = [100 + (e['growth'] / max_growth) * 500 for e in emerging]

        colors = plt.cm.Greens([0.3 + 0.7 * (e['growth'] / max_growth) for e in emerging])

        x = range(len(emerging))
        ax.scatter(x, [e['count_2025'] for e in emerging], s=sizes, c=colors, alpha=0.7)

        # Labels
        for i, e in enumerate(emerging):
            ax.annotate(
                f"{e['keyword']}\n(+{e['growth']:.0f}%)",
                (i, e['count_2025']),
                fontsize=8,
                ha='center',
                va='bottom'
            )

        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('Current Frequency (2025)', fontsize=12)
        ax.set_title(f'Emerging Keywords (Growth >= {min_growth}%)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        self.plt.tight_layout()
        output_path = self.output_dir / f"{output_name}.png"
        self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.plt.close()

        logger.info(f"Saved emerging keywords to {output_path}")
        return output_path

    def plot_conference_similarity(
        self,
        papers: List,
        output_name: str = "conference_similarity"
    ) -> Optional[Path]:
        """Plot conference similarity network based on keyword overlap."""
        if not self.matplotlib_available:
            return None

        try:
            from collections import Counter, defaultdict
            import numpy as np
            import networkx as nx

            # Build keyword profiles for each conference
            conf_keywords = defaultdict(Counter)

            for paper in papers:
                venue = getattr(paper, 'venue', 'unknown')
                if hasattr(paper, 'keywords') and paper.keywords:
                    for kw in paper.keywords:
                        if isinstance(kw, str):
                            conf_keywords[venue.lower()][kw.lower().strip()] += 1
                        elif isinstance(kw, dict):
                            kw_name = kw.get('keyword', kw.get('name', ''))
                            if kw_name:
                                conf_keywords[venue.lower()][kw_name.lower().strip()] += 1

            # Get top conferences
            conf_totals = {c: sum(kws.values()) for c, kws in conf_keywords.items()}
            top_confs = sorted(conf_totals.items(), key=lambda x: x[1], reverse=True)[:12]
            top_conf_names = [c for c, _ in top_confs]

            # Calculate Jaccard similarity
            similarities = []
            for i, c1 in enumerate(top_conf_names):
                for c2 in top_conf_names[i+1:]:
                    keywords1 = set(conf_keywords[c1].keys())
                    keywords2 = set(conf_keywords[c2].keys())
                    intersection = len(keywords1 & keywords2)
                    union = len(keywords1 | keywords2)
                    if union > 0:
                        jaccard = intersection / union
                        if jaccard > 0.05:  # Only show meaningful similarities
                            similarities.append((c1, c2, jaccard))

            if not similarities:
                logger.warning("No conference similarities found")
                return None

            # Create network
            G = nx.Graph()
            for c in top_conf_names:
                G.add_node(c.upper(), size=conf_totals.get(c, 100))

            for c1, c2, sim in similarities:
                G.add_edge(c1.upper(), c2.upper(), weight=sim)

            # Plot
            fig, ax = self.plt.subplots(figsize=(12, 10))

            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

            # Node sizes
            node_sizes = [conf_totals.get(c.lower(), 100) * 2 for c in G.nodes()]
            node_sizes = [min(s, 2000) for s in node_sizes]

            # Edge widths
            edge_weights = [G[u][v].get('weight', 0.1) * 10 for u, v in G.edges()]

            nx.draw_networkx_edges(G, pos, alpha=0.4, width=edge_weights, ax=ax)
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

            # Add edge labels for strong connections
            edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True) if d['weight'] > 0.2}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

            ax.set_title('Conference Similarity Network (Jaccard Index)', fontsize=14, fontweight='bold')
            ax.axis('off')

            self.plt.tight_layout()
            output_path = self.output_dir / f"{output_name}.png"
            self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.plt.close()

            logger.info(f"Saved conference similarity to {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"Conference similarity failed: {e}")
            return None

    def plot_topic_radar(
        self,
        domain_data: Dict[str, Any],
        output_name: str = "topic_radar"
    ) -> Optional[Path]:
        """Plot research topic radar chart - multi-dimensional comparison."""
        if not self.matplotlib_available:
            return None

        try:
            from math import pi

            # Define dimensions based on available data
            dimensions = []
            values = []

            # Paper count
            dimensions.append('Paper Volume')
            values.append(domain_data.get('total_papers', 0))

            # Keyword diversity (unique keywords / papers)
            top_kw = domain_data.get('top_keywords', [])
            dimensions.append('Topic Diversity')
            values.append(len(top_kw))

            # Growth indicator
            yearly = domain_data.get('yearly_trends', {})
            growth = 0
            if len(yearly) >= 2:
                years = sorted(yearly.keys())
                growth = (yearly[years[-1]] - yearly[years[0]]) / max(yearly[years[0]], 1) * 100
            dimensions.append('Growth Rate')
            values.append(max(0, growth))  # Normalize

            # Number of venues
            venues = domain_data.get('venue_distribution', {})
            dimensions.append('Venue Coverage')
            values.append(len(venues))

            # Lifecycle stage score
            lifecycle = domain_data.get('lifecycle', {})
            stage = lifecycle.get('stage', 'unknown')
            stage_scores = {'emerging': 100, 'growing': 75, 'mature': 50, 'declining': 25}
            dimensions.append('Maturity')
            values.append(stage_scores.get(stage, 50))

            # Normalize values to 0-100
            max_val = max(values) if max(values) > 0 else 1
            values = [v / max_val * 100 for v in values]

            # Plot radar chart
            N = len(dimensions)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Close the circle
            values = values + values[:1]

            fig, ax = self.plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

            ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
            ax.fill(angles, values, alpha=0.25, color='steelblue')

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dimensions, fontsize=10)
            ax.set_ylim(0, 100)

            domain_name = list(domain_data.keys())[0] if isinstance(domain_data, dict) else 'Domain'
            ax.set_title(f'Research Profile: {domain_name}', fontsize=14, fontweight='bold', pad=20)

            self.plt.tight_layout()
            output_path = self.output_dir / f"{output_name}.png"
            self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.plt.close()

            logger.info(f"Saved topic radar to {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"Topic radar failed: {e}")
            return None

    def plot_ipa_quadrant(
        self,
        yearly_keyword_data: Dict[int, Dict[str, int]],
        output_name: str = "ipa_quadrant"
    ) -> Optional[Path]:
        """Plot IPA (Importance-Performance Analysis) quadrant - strategic keyword positioning."""
        if not self.matplotlib_available:
            return None

        if not yearly_keyword_data or len(yearly_keyword_data) < 2:
            logger.warning("Need at least 2 years for IPA analysis")
            return None

        years = sorted(yearly_keyword_data.keys())
        early = yearly_keyword_data[years[0]]
        late = yearly_keyword_data[years[-1]]

        # Calculate importance (frequency in latest year) and growth
        keyword_stats = []
        all_keywords = set(early.keys()) | set(late.keys())

        for kw in all_keywords:
            early_count = early.get(kw, 0)
            late_count = late.get(kw, 0)

            if late_count < 5:
                continue

            # Importance: frequency in latest year
            importance = late_count

            # Growth rate
            if early_count > 0:
                growth = ((late_count - early_count) / early_count) * 100
            else:
                growth = 100

            keyword_stats.append({
                'keyword': kw,
                'importance': importance,
                'growth': growth
            })

        if not keyword_stats:
            return None

        # Sort by importance
        keyword_stats = sorted(keyword_stats, key=lambda x: x['importance'], reverse=True)[:30]

        # Calculate medians
        median_importance = sum(k['importance'] for k in keyword_stats) / len(keyword_stats)
        median_growth = sum(k['growth'] for k in keyword_stats) / len(keyword_stats)

        # Categorize
        categories = {'Rising Stars': [], 'Mature Leaders': [], 'Declining': [], 'Niche': []}
        for kw in keyword_stats:
            if kw['growth'] > median_growth and kw['importance'] > median_importance:
                categories['Rising Stars'].append(kw)
            elif kw['growth'] > median_growth and kw['importance'] <= median_importance:
                categories['Niche'].append(kw)
            elif kw['growth'] <= median_growth and kw['importance'] > median_importance:
                categories['Mature Leaders'].append(kw)
            else:
                categories['Declining'].append(kw)

        # Plot
        fig, ax = self.plt.subplots(figsize=(12, 10))

        colors = {'Rising Stars': 'green', 'Mature Leaders': 'blue', 'Declining': 'red', 'Niche': 'orange'}

        for cat, kws in categories.items():
            x = [kw['importance'] for kw in kws]
            y = [kw['growth'] for kw in kws]
            ax.scatter(x, y, c=colors[cat], label=f'{cat} ({len(kws)})', s=100, alpha=0.7)

            # Label top keywords in each category
            for kw in sorted(kws, key=lambda x: x['importance'], reverse=True)[:3]:
                ax.annotate(kw['keyword'], (kw['importance'], kw['growth']), fontsize=8)

        # Add quadrant lines
        ax.axvline(x=median_importance, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=median_growth, color='gray', linestyle='--', alpha=0.5)

        # Labels
        ax.set_xlabel('Importance (Frequency)', fontsize=12)
        ax.set_ylabel('Growth Rate (%)', fontsize=12)
        ax.set_title('IPA Keyword Quadrant Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        self.plt.tight_layout()
        output_path = self.output_dir / f"{output_name}.png"
        self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.plt.close()

        logger.info(f"Saved IPA quadrant to {output_path}")
        return output_path
