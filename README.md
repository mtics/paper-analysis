# CCF-A 论文深度分析系统

> CCF Conference Paper Analysis Tool

对 CCF-A 类顶会论文进行深度分析，追踪技术趋势、发现研究热点、分析学术生态系统。

## 功能特性

### 1. 趋势分析
- 年度论文数量统计
- 关键词趋势识别
- Mann-Kendall 非参数趋势检验
- 新兴关键词发现

### 2. 生态系统分析
- **词汇时间轴**：追踪技术词汇从"首次出现"到"全面扩散"的演变
- **会议相似度矩阵**：基于 TF-IDF 计算会议间主题相似度
- **技术扩散路径**：识别技术如何在会议间传播
- **知识流动图**：基于 PMI 构建关键词共现网络

### 3. 网络分析
- 共作者网络构建
- 桥接研究者识别
- 网络演化分析

### 4. 深度领域分析
- 领域论文检索与相关性评分
- 子方向自动发现
- **S 曲线生命周期拟合**：判断领域处于萌芽/增长/成熟/衰退期
- **研究者稳定性分析**：基于 Jaccard 系数分析作者更替
- **词汇新陈代谢**：追踪领域术语的兴衰

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本用法

```bash
# 列出可用会议
python -m analysis list

# 查看统计信息
python -m analysis stats

# 分析单个会议
python -m analysis analyze aaai --years 2022,2023,2024 --keywords --domains

# 深度领域分析
python -m analysis deep "AI Agent" --years 2023,2024,2025
python -m analysis deep "Federated Learning" --years 2023,2024,2025

# 词汇时间轴
python -m analysis timeline --years 2020,2021,2022 --top-n 50

# 网络分析
python -m analysis network --years 2020,2021
```

### 输出结果

```bash
# 指定输出目录
python -m analysis deep "Transformer" --years 2020,2021,2022 --output output/analysis
```

结果将保存到：
- `output/logs/` - 运行时日志
- `output/analysis/trends/` - 趋势分析结果
- `output/analysis/ecosystem/` - 生态系统分析结果
- `output/analysis/network/` - 网络分析结果
- `output/analysis/deep/` - 深度分析结果

## 项目架构

```
analysis/
├── core/                    # 核心数据层
│   └── data_loader.py      # Paper, ConferenceData
│
├── features/               # 功能模块
│   ├── preprocessing/     # 文本预处理、N-gram 提取
│   ├── trends/            # 趋势分析、Mann-Kendall 检验
│   ├── topics/            # 主题建模、子领域分析
│   ├── ecosystem/         # 词汇时间轴、技术扩散
│   ├── network/           # 共作者网络
│   └── deep/              # 深度分析、S曲线、生命周期
│
├── utils/                 # 工具层
│   ├── logger.py          # 统一日志系统
│   └── output.py          # 输出管理器
│
└── main.py               # CLI 入口
```

## 支持的会议

| 会议 | 领域 | 年份范围 |
|------|------|----------|
| AAAI | AI | 2015-2025 |
| IJCAI | AI | 2015-2025 |
| NeurIPS | AI/ML | 2015-2025 |
| ICLR | ML | 2015-2025 |
| ICML | ML | 2015-2025 |
| ACL | NLP | 2015-2025 |
| CVPR | CV | 2015-2025 |
| ICCV | CV | 2015-2024 |
| KDD | DM | 2015-2025 |
| SIGIR | IR | 2015-2025 |
| SIGMOD | DB | 2015-2025 |
| ICDE | DB | 2015-2025 |
| MM | MM | 2015-2025 |

## 分析示例

### 深度领域分析示例

```bash
# 分析 AI Agent 领域
python -m analysis deep "AI Agent" --years 2020,2021,2022
```

输出：
```
🔍 Analyzing domain: AI Agent
   Years: [2020, 2021, 2022]
   Conferences: all CCF-A

📊 Deep Domain Analysis: AI Agent
================================================================================
📅 Year Range: 2020 - 2022
📄 Total Papers: 28

📈 Yearly Trends
================================================================================
  2020:   13 ██
  2021:    8 █
  2022:    7 █

🏛️ Venue Distribution (Top 5)
================================================================================
  AAAI: 11 (39.3%)
  NeurIPS: 8 (28.6%)
  ...

=== Lifecycle Analysis ===
Stage: growth
Projected ceiling (L): 450
Growth rate (k): 0.65
R-squared: 0.892

=== Researcher Stability ===
Avg Jaccard: 0.234
Stage: mixed
```

## 开发指南

### 运行测试

```bash
pytest tests/ -v
```

### 添加新功能

1. 在 `features/` 下创建新模块
2. 在对应 `__init__.py` 中导出
3. 在 `main.py` 中添加 CLI 命令

## 技术栈

- **数据处理**：Python, Pandas, NumPy
- **统计检验**：SciPy, Mann-Kendall
- **NLP**：NLTK, Gensim (N-gram)
- **网络分析**：NetworkX
- **机器学习**：scikit-learn

## 文档

- [分析框架设计文档](docs/分析框架设计文档.md)
- [深度分析设计方案](docs/深度分析设计方案.md)
- [开源工具调研报告](docs/开源工具调研报告.md)

## License

MIT
