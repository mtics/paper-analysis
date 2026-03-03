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
