# Visualization Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 CCF-A 论文分析系统添加可视化功能，包括静态图表、交互式网络图和 HTML 仪表盘。

**Architecture:** 在 `analysis/features/` 下创建 `visualization/` 模块，提供 matplotlib 静态图表、PyVis 交互式网络图和 HTML 仪表盘生成能力。与现有 full_analysis 流程集成。

**Tech Stack:** matplotlib, seaborn, pyvis, networkx

---

### Task 1: Create visualization module structure

**Files:**
- Create: `analysis/features/visualization/__init__.py`
- Create: `analysis/features/visualization/charts.py`
- Create: `analysis/features/visualization/network_viz.py`
- Create: `analysis/features/visualization/dashboard.py`

**Step 1: Create visualization/__init__.py**

```python
"""Visualization module for CCF-A paper analysis."""

from analysis.features.visualization.charts import TrendCharts
from analysis.features.visualization.network_viz import NetworkViz
from analysis.features.visualization.dashboard import DashboardGenerator

__all__ = ['TrendCharts', 'NetworkViz', 'DashboardGenerator']
```

**Step 2: Create empty charts.py**

```python
"""Static chart generation using matplotlib/seaborn."""

# TODO: Implement TrendCharts class
```

**Step 3: Create empty network_viz.py**

```python
"""Interactive network visualization using PyVis."""

# TODO: Implement NetworkViz class
```

**Step 4: Create empty dashboard.py**

```python
"""HTML dashboard generator."""

# TODO: Implement DashboardGenerator class
```

**Step 5: Run test to verify imports work**

Run: `python -c "from analysis.features.visualization import TrendCharts, NetworkViz, DashboardGenerator; print('OK')"`
Expected: OK (with warnings about undefined classes)

---

### Task 2: Implement TrendCharts (charts.py)

**Files:**
- Create: `analysis/features/visualization/charts.py`
- Modify: `analysis/features/visualization/__init__.py`

**Step 1: Write the failing test**

```python
# tests/features/visualization/test_charts.py
import pytest
from pathlib import Path
import tempfile

def test_trend_charts_initialization():
    """Test TrendCharts can be initialized with output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        charts = TrendCharts(output_dir=Path(tmpdir))
        assert charts.output_dir == Path(tmpdir)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/features/visualization/test_charts.py::test_trend_charts_initialization -v`
Expected: FAIL with "cannot import name 'TrendCharts'"

**Step 3: Write minimal TrendCharts class**

```python
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

logger = logging.getLogger(__name__)

class TrendCharts:
    """Generate static charts using matplotlib/seaborn."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("output/visualizations/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Try to import matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            self matplotlib_available = True
            self.plt = plt
            self.sns = sns
        except ImportError:
            self.matplotlib_available = False
            warnings.warn("matplotlib/seaborn not available, charts will be skipped")
            logger.warning("matplotlib/seaborn not available")

    def plot_yearly_distribution(self, yearly_data: Dict[int, int], output_name: str = "yearly_distribution"):
        """Plot yearly paper distribution as bar chart."""
        if not self.matplotlib_available:
            logger.warning("matplotlib not available, skipping yearly distribution chart")
            return None

        # Implementation here...
        return self.output_dir / f"{output_name}.png"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/features/visualization/test_charts.py::test_trend_charts_initialization -v`
Expected: PASS

**Step 5: Implement remaining chart methods**

Add: plot_venue_distribution(), plot_keyword_trends(), plot_venue_heatmap()

**Step 6: Commit**

```bash
git add analysis/features/visualization/ tests/features/visualization/
git commit -m "feat(visualization): add TrendCharts class with matplotlib charts"
```

---

### Task 3: Implement NetworkViz (network_viz.py)

**Files:**
- Create: `analysis/features/visualization/network_viz.py`
- Modify: `analysis/features/visualization/__init__.py`

**Step 1: Write the failing test**

```python
def test_network_viz_initialization():
    """Test NetworkViz can be initialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        viz = NetworkViz(output_dir=Path(tmpdir))
        assert viz.output_dir == Path(tmpdir)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/features/visualization/test_network_viz.py::test_network_viz_initialization -v`
Expected: FAIL with "cannot import name 'NetworkViz'"

**Step 3: Write minimal NetworkViz class**

```python
import logging
from pathlib import Path
from typing import Optional
import warnings

logger = logging.getLogger(__name__)

class NetworkViz:
    """Generate interactive network visualizations using PyVis."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("output/visualizations/network")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Try to import pyvis
        try:
            from pyvis.network import Network
            self.pyvis_available = True
            self.Net = Net
        except ImportError:
            self.pyvis_available = False
            warnings.warn("pyvis not available, network visualization will be skipped")
            logger.warning("pyvis not available")

    def plot_coauthor_network(self, G, output_name: str = "coauthor_network"):
        """Plot coauthor network as interactive HTML."""
        if not self.pyvis_available:
            logger.warning("pyvis not available, skipping network visualization")
            return None

        # Implementation here...
        return self.output_dir / f"{output_name}.html"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/features/visualization/test_network_viz.py::test_network_viz_initialization -v`
Expected: PASS

**Step 5: Implement full network visualization with pruning options**

**Step 6: Commit**

```bash
git add analysis/features/visualization/network_viz.py tests/features/visualization/
git commit -m "feat(visualization): add NetworkViz class with PyVis"
```

---

### Task 4: Implement DashboardGenerator (dashboard.py)

**Files:**
- Create: `analysis/features/visualization/dashboard.py`
- Modify: `analysis/features/visualization/__init__.py`

**Step 1: Write the failing test**

```python
def test_dashboard_generator():
    """Test DashboardGenerator can be initialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = DashboardGenerator(output_dir=Path(tmpdir))
        assert gen.output_dir == Path(tmpdir)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/features/visualization/test_dashboard.py::test_dashboard_generator -v`
Expected: FAIL with "cannot import name 'DashboardGenerator'"

**Step 3: Write minimal DashboardGenerator class**

```python
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DashboardGenerator:
    """Generate HTML dashboard combining all visualizations."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("output/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, results: Dict[str, Any], papers: List) -> Path:
        """Generate complete dashboard HTML."""
        # Implementation here...
        return self.output_dir / "dashboard.html"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/features/visualization/test_dashboard.py::test_dashboard_generator -v`
Expected: PASS

**Step 5: Implement full dashboard with chart embedding**

**Step 6: Commit**

```bash
git add analysis/features/visualization/dashboard.py tests/features/visualization/
git commit -m "feat(visualization): add DashboardGenerator for HTML reports"
```

---

### Task 5: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Add visualization dependencies**

Add to requirements.txt:
```
# ── Visualization ───────────────────────────────────────────────────────
matplotlib>=3.7.0
seaborn>=0.12.0
```

**Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add matplotlib and seaborn dependencies"
```

---

### Task 6: Integrate into full_analysis command

**Files:**
- Modify: `analysis/main.py`

**Step 1: Write the failing test**

```python
def test_full_analysis_with_viz():
    """Test full_analysis includes visualization step."""
    # This is a manual verification test
    pass
```

**Step 2: Add visualization import and generation to full_analysis**

In `main.py`, add after line ~817 (after saving JSON reports):

```python
# 6. Visualization
print("\n" + "-" * 70)
print("📊 第六部分：可视化生成")
print("-" * 70)

try:
    from analysis.features.visualization import DashboardGenerator

    viz_output = output_path / "visualizations"
    viz_gen = DashboardGenerator(viz_output)
    viz_gen.generate(results, papers)

    print(f"\n   可视化已保存至: {viz_output}")
except Exception as e:
    print(f"\n   ⚠️ 可视化生成跳过: {str(e)[:50]}")
    logger.warning(f"Visualization failed: {e}")
```

**Step 3: Run integration test**

Run: `python -m analysis full --years 2024,2025 --domains "AI Agent" --output output 2>&1 | tail -20`
Expected: Should complete with visualization section

**Step 4: Commit**

```bash
git add analysis/main.py
git commit -m "feat(main): integrate visualization into full_analysis command"
```

---

### Task 7: Run full integration test

**Files:**
- Test: Full end-to-end execution

**Step 1: Run full analysis with visualization**

```bash
python -m analysis full --years 2024,2025 --domains "AI Agent" --output output
```

**Step 2: Verify output structure**

```bash
ls -la output/analysis/visualizations/
ls -la output/analysis/visualizations/charts/
ls -la output/analysis/visualizations/network/
```

Expected: All directories and files created

**Step 3: Verify dashboard.html exists**

```bash
test -f output/analysis/visualizations/dashboard.html && echo "Dashboard exists!"
```

Expected: Dashboard exists!

**Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete visualization module integration"
```
