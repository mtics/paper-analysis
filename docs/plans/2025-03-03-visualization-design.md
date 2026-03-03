# 可视化模块设计方案

> **For Claude:** Use superpowers:writing-plans to implement this plan.

**Goal:** 为 CCF-A 论文分析系统添加可视化功能，包括静态图表、交互式网络图和 HTML 仪表盘。

**Architecture:** 在 `analysis/features/` 下创建 `visualization/` 模块，提供 matplotlib 静态图表、PyVis 交互式网络图和 HTML 仪表盘生成能力。与现有 full_analysis 流程集成，输出到 `output/analysis/visualizations/`。

**Tech Stack:** matplotlib, seaborn, pyvis, networkx

---

## 1. 模块结构

```
analysis/features/visualization/
├── __init__.py           # 导出主要类
├── charts.py            # matplotlib/seaborn 静态图表
├── network_viz.py       # PyVis 网络可视化
└── dashboard.py         # HTML 仪表盘生成器
```

## 2. 核心功能

### 2.1 TrendCharts (charts.py)

| 方法 | 功能 | 输出 |
|------|------|------|
| `plot_yearly_distribution()` | 年度论文数量柱状图 | PNG + JSON |
| `plot_venue_distribution()` | 会议论文占比饼图 | PNG + JSON |
| `plot_keyword_trends()` | 关键词热度折线图 | PNG + JSON |
| `plot_venue_heatmap()` | 会议-年度热力图 | PNG + JSON |

### 2.2 NetworkViz (network_viz.py)

| 方法 | 功能 | 输出 |
|------|------|------|
| `plot_coauthor_network()` | 作者合作网络图 | HTML |
| `export_json()` | 网络数据 JSON 导出 | JSON |

### 2.3 DashboardGenerator (dashboard.py)

| 方法 | 功能 | 输出 |
|------|------|------|
| `generate()` | 生成完整仪表盘 | HTML |
| `embed_charts()` | 嵌入静态图表 | HTML |
| `embed_network()` | 嵌入交互网络 | HTML |

## 3. 输出结构

```
output/analysis/visualizations/
├── charts/
│   ├── yearly_distribution.png
│   ├── venue_distribution.png
│   ├── keyword_trends.png
│   └── venue_heatmap.png
├── network/
│   └── coauthor_network.html
└── dashboard.html    # 主仪表盘入口
```

## 4. 与 full_analysis 集成

在 `main.py` 的 `full_analysis()` 函数末尾添加：

```python
# 6. Visualization
from analysis.features.visualization import DashboardGenerator

print("\n" + "-" * 70)
print("📊 第六部分：可视化生成")
print("-" * 70)

viz_gen = DashboardGenerator(output_path / "visualizations")
viz_gen.generate(results, papers)

print(f"\n   可视化已保存至: {output_path / 'visualizations'}")
```

## 5. 依赖更新

在 `requirements.txt` 中添加：

```txt
# ── Visualization ───────────────────────────────────────────────────────
matplotlib>=3.7.0
seaborn>=0.12.0
```

注：pyvis 已在可选依赖中。

## 6. 错误处理

- matplotlib/seaborn 不可用时：跳过静态图表，记录 warning
- pyvis 不可用时：跳过网络图，记录 warning
- 图表生成失败时：记录 error 但继续执行，不中断分析流程

## 7. 测试计划

1. 创建 `tests/features/visualization/` 目录
2. 单元测试：每个图表方法独立测试
3. 集成测试：DashboardGenerator 完整流程测试
4. 使用模拟数据进行测试
