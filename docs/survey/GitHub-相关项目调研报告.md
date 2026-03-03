# 学术界论文趋势分析工具调研报告

## 一、调研概述

本调研旨在识别 GitHub 上与本项目（CCF-A 顶会论文深度分析）功能类似的已开源项目，避免重复造轮子，并从中汲取设计灵感。

**调研范围**：
- 论文趋势分析与可视化
- 主题建模与关键词提取
- 引文网络分析
- 学术知识图谱
- 研究影响力评估

**调研时间**：2025年

---

## 二、GitHub 相关项目汇总

### 2.1 高星项目（★100+）

| 项目 | Stars | 描述 | 技术栈 |
|------|-------|------|--------|
| **AI-Paper-Trends** | ★130 | 使用主题建模分析顶级 AI 会议论文，发现研究热点和趋势 | Python, Topic Modeling |

> 来源：https://github.com/... (AI-Paper-Trends)

### 2.2 中星项目（★10-100）

| 项目 | Stars | 描述 | 技术栈 | 可借鉴点 |
|------|-------|------|--------|----------|
| **ai-research-trend-dashboard** | ★6 | 使用 OpenAlex 实时可视化 30+ AI 研究子领域的趋势，自动年度更新，包含迷你趋势图、排名、热力图 | Streamlit, OpenAlex API | 数据源接入、实时更新机制 |
| **citation-graph-builder** | - | 从 PDF 解析和 API 查询创建和可视化引文网络 | Python, NetworkX | 引文网络构建 |
| **COVID19-Research-Analysis** | ★2 | 清洗、分析和可视化 COVID-19 研究论文元数据，包含 Streamlit 交互式探索 | Streamlit, Plotly | 交互式仪表盘设计 |

### 2.3 低星/新项目（★<10）

| 项目 | 描述 | 技术栈 | 可借鉴点 |
|------|------|--------|----------|
| **nlg-paper-analysis** | 可视化 NLG 领域主要会议论文的近期趋势 | Python | 会议级分析思路 |
| **ai-research-trends-arxiv** | 使用 arXiv 元数据探索和可视化 AI 研究趋势—主题建模、NLP和研究分析 | Python, Topic Modeling | arXiv 数据处理 |
| **arxiv-topic-modeling** | 对 arXiv 摘要进行主题建模 | Gensim LDA | 主题建模实现 |

---

## 三、业界知名工具（不完全在 GitHub）

### 3.1 bibliometrix（R 包）

- **定位**：文献计量学全能工具
- **功能**：
  - 文献计量分析
  - 共词网络
  - 合作网络
  - 主题演化
- **数据源**：Web of Science, Scopus, PubMed
- **评价**：学术界标准工具，但仅支持 R 语言

### 3.2 VOSviewer

- **定位**：文献网络可视化
- **功能**：
  - 引文网络
  - 合作网络
  - 共词网络
- **评价**：Java 桌面应用，可视化能力强但交互性弱

### 3.3 Connected Papers / ResearchRabbit

- **定位**：论文发现工具
- **特点**：基于引文的图探索，非分析工具
- **评价**：产品化程度高，但非开源

### 3.4 Semantic Scholar / OpenAlex

- **定位**：学术搜索引擎和开放数据平台
- **API**：提供论文元数据、引文、作者机构等
- **本项目已使用**：OpenAlex（可选接入）

---

## 四、本项目差异化分析

### 4.1 现有项目共性问题

| 问题 | 描述 |
|------|------|
| **数据源单一** | 多依赖 arXiv，缺乏对 CCF 会议的专门处理 |
| **分析深度浅** | 主要是词频统计，缺少 Mann-Kendall 趋势检验、S曲线拟合等统计方法 |
| **可视化单一** | 缺少交互式 HTML 可视化（PyVis） |
| **无中文支持** | 缺乏对中文术语和国内社区的适配 |

### 4.2 本项目独特价值

| 维度 | 现有项目 | 本项目 |
|------|----------|--------|
| **数据范围** | arXiv 为主 | CCF-A 13个顶会（2015-2025） |
| **分析方法** | 朴素词频统计 | Mann-Kendall 趋势检验、S曲线拟合、PMI知识流动 |
| **领域深度** | 通用主题建模 | 预定义领域+种子扩散法软归属 |
| **可视化** | 静态图表 | PyVis 交互式 HTML + JSON 报告 |
| **GPU 加速** | 罕见 | 支持 BERTopic/SPECTER2（RTX 3070 Ti） |

---

## 五、可复用/可借鉴的实现

### 5.1 可直接参考的开源实现

| 功能模块 | 参考项目 | 借鉴内容 |
|----------|----------|----------|
| **OpenAlex API 接入** | ai-research-trend-dashboard | 数据获取 + 自动更新机制 |
| **Streamlit 仪表盘** | COVID19-Research-Analysis | 交互式 UI 设计 |
| **主题建模** | arxiv-topic-modeling | Gensim LDA 管道 |
| **引文网络** | citation-graph-builder | NetworkX 图构建 |

### 5.2 需要自行实现的核心能力

根据调研，以下能力在现有开源项目中**较少见**，是本项目的差异化核心：

1. **Mann-Kendall 趋势检验**：统计显著的趋势判断（非朴素增长率）
2. **N-gram 短语提取**：Gensim Phrases 处理多词术语
3. **会议相似度矩阵时序演化**：跨会议的词汇 borrowing 分析
4. **知识流动有向图**：PMI 计算领域间词汇借用
5. **S 曲线生命周期拟合**：logistic 模型判断领域成熟度
6. **PyVis 交互式可视化**：独立 HTML 导出能力

---

## 六、调研结论

### 6.1 重复性评估

**低重复风险**：现有 GitHub 项目未能完整覆盖本项目的分析需求。

- 多数项目聚焦于 arXiv 论文，缺乏对 CCF 会议的专业处理
- 统计方法论层面（Mann-Kendall、S曲线）在学术分析工具中罕见
- 知识流动分析（PMI）是本项目的独特创新

### 6.2 推荐策略

1. **借鉴而非复制**：参考 ai-research-trend-dashboard 的 OpenAlex 接入思路，但保持本地 DBLP 数据源
2. **差异化核心**：聚焦统计方法论（Mann-Kendall、S曲线）和知识流动分析
3. **开源协作**：如实现成熟，可考虑开源贡献

### 6.3 后续行动

- 深入研究 ai-research-trend-dashboard 的代码架构
- 评估 Bibliometrix 方法论的可复现性
- 确定 PyVis 可视化的具体实现方案

---

## 附录：参考链接

### GitHub 项目

- AI-Paper-Trends: https://github.com/... (★130)
- ai-research-trend-dashboard: https://github.com/QronG9/ai-research-trend-dashboard (★6)
- citation-graph-builder: https://github.com/FZJ-IEK3-VSA/citation-graph-builder
- COVID19-Research-Analysis: https://github.com/kaylakyle/COVID19-Research-Analysis
- nlg-paper-analysis: https://github.com/jingyng/nlg-paper-analysis
- arxiv-topic-modeling: https://github.com/.../arxiv-topic-modeling

### 业界工具

- Bibliometrix: https://bibliometrix.org/
- VOSviewer: https://www.vosviewer.com/
- Connected Papers: https://www.connectedpapers.com/
- OpenAlex: https://openalex.org/
- Semantic Scholar: https://www.semanticscholar.org/
