#!/usr/bin/env python3
"""
Analysis CLI for CCF conference papers.

Architecture:
- cli: Command-line interface
- core: Data models and loaders
- features: Analysis features (preprocessing, trends, topics, ecosystem, network, deep)
- utils: Logging and output management
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from new hierarchical structure
from analysis.core import PaperDataLoader, load_papers_for_analysis
from analysis.features.preprocessing import TextPreprocessor
from analysis.features.topics import DomainClassifier, TopicModeler, SubtopicAnalyzer, classify_papers_by_domain
from analysis.features.trends import TrendAnalyzer, ComparativeAnalyzer, generate_trend_report
from analysis.features.deep import (
    DeepDomainAnalyzer, format_report, DOMAIN_DEFINITIONS, analyze_vocabulary_turnover,
    LifecycleAnalyzer, ResearcherStabilityAnalyzer
)
from analysis.features.ecosystem import VocabularyTimeline, ConferenceSimilarityMatrix, TechnologyDiffusion
from analysis.features.network import CoauthorNetworkAnalyzer

# Use unified logging system
from analysis.utils import setup_logger, get_logger, OutputManager

# Setup logger
logger = setup_logger('analysis')


def list_conferences(args):
    """List available conferences."""
    loader = PaperDataLoader(args.data_dir)
    conferences = loader.get_available_conferences()

    print("\n=== Available Conferences ===")
    for conf in conferences:
        conf_data = loader.load_conference(conf)
        print(f"  {conf}: {conf_data.paper_count} papers ({conf_data.year_range[0]}-{conf_data.year_range[1]})")

    return conferences


def show_statistics(args):
    """Show overall statistics in table format."""
    loader = PaperDataLoader(args.data_dir)
    stats = loader.get_statistics()

    # Overall summary
    total_papers = stats['total_papers']
    total_with_abstract = stats['papers_with_abstract']
    coverage_rate = total_with_abstract / total_papers * 100 if total_papers > 0 else 0

    print("\n" + "=" * 70)
    print("                     CCF-A 会议论文统计概览")
    print("=" * 70)

    # Summary table
    print(f"\n{'📊 总体统计':<20}")
    print("-" * 50)
    print(f"  {'会议数量:':<18} {stats['total_conferences']}")
    print(f"  {'论文总数:':<18} {total_papers:,}")
    print(f"  {'有摘要论文:':<18} {total_with_abstract:,}")
    print(f"  {'完整率:':<18} {coverage_rate:.1f}%")

    # Conference table
    print(f"\n{'📚 各会议详情':<20}")
    print("-" * 70)
    print(f"  {'会议':<10} {'论文数':>10} {'有摘要':>10} {'完整率':>10} {'年份范围':>15}")
    print("  " + "-" * 66)

    # Sort by paper count
    sorted_confs = sorted(
        stats['conferences'].items(),
        key=lambda x: x[1]['papers'],
        reverse=True
    )

    for conf, conf_stats in sorted_confs:
        papers = conf_stats['papers']
        with_abstract = conf_stats['with_abstract']
        rate = with_abstract / papers * 100 if papers > 0 else 0
        year_range = conf_stats.get('year_range', (0, 0))
        if isinstance(year_range, tuple):
            year_str = f"{year_range[0]}-{year_range[1]}"
        else:
            year_str = str(year_range)

        print(f"  {conf.upper():<10} {papers:>10,} {with_abstract:>10,} {rate:>9.1f}% {year_str:>15}")

    print("  " + "-" * 66)
    total_row = f"  {'总计':<10} {total_papers:>10,} {total_with_abstract:>10,} {coverage_rate:>9.1f}% {'-':>15}"
    print(total_row)
    print("=" * 70)


def analyze_conference(args):
    """Analyze a specific conference."""
    loader = PaperDataLoader(args.data_dir)
    conf_data = loader.load_conference(args.conference, args.years)

    papers = conf_data.papers
    print(f"\n=== {args.conference.upper()} Analysis ===")
    print(f"Total papers: {len(papers)}")
    print(f"Papers with abstract: {conf_data.papers_with_abstract}")
    print(f"Year range: {conf_data.year_range[0]} - {conf_data.year_range[1]}")

    if not papers:
        return

    # Yearly trends
    analyzer = TrendAnalyzer()
    yearly = analyzer.analyze_yearly_trends(papers)

    print("\n=== Yearly Trends ===")
    print(f"{'Year':<8} {'Papers':<10} {'With Abstract':<15}")
    for i, year in enumerate(yearly['years']):
        print(f"{year:<8} {yearly['paper_counts'][i]:<10} {yearly['papers_with_abstract'][i]:<15}")

    # Keyword trends
    if args.keywords:
        keywords = analyzer.analyze_keyword_trends(papers, top_n=20)
        print("\n=== Top Keywords ===")
        for kw, count in list(keywords['keyword_total'].items())[:20]:
            print(f"  {kw}: {count}")

    # Emerging keywords
    if args.emerging:
        keywords = analyzer.analyze_keyword_trends(papers)
        print("\n=== Emerging Keywords ===")
        for kw, growth in keywords['emerging_keywords'][:10]:
            print(f"  {kw}: +{growth*100:.1f}% growth")

    # Domain distribution
    if args.domains:
        domain_papers = classify_papers_by_domain(papers)
        print("\n=== Domain Distribution ===")
        for domain, domain_papers_list in sorted(domain_papers.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {domain}: {len(domain_papers_list)} papers")

    # Author trends
    if args.authors:
        author_data = analyzer.analyze_author_trends(papers)
        print("\n=== Top Authors ===")
        for author, count in author_data['prolific_authors'][:10]:
            print(f"  {author}: {count} papers")

    # Save detailed report
    if args.output:
        output_mgr = OutputManager(Path(args.output))

        report = generate_trend_report(papers, args.conference)

        report_data = {
            'conference': report.conference,
            'year_range': report.year_range,
            'total_papers': report.total_papers,
            'papers_with_abstract': report.papers_with_abstract,
            'yearly_counts': report.yearly_counts,
            'domain_distribution': report.domain_distribution,
            'top_keywords': report.top_keywords
        }

        filepath = output_mgr.save_json(report_data, 'trends', f"{args.conference}_report.json")
        print(f"\nReport saved to: {filepath}")


def analyze_domain(args):
    """Analyze a specific domain."""
    loader = PaperDataLoader(args.data_dir)

    # Load papers
    if args.conferences:
        papers = load_papers_for_analysis(args.conferences, args.years)
    else:
        papers = load_papers_for_analysis(years=args.years)

    # Filter by domain
    classifier = DomainClassifier()

    domain_papers = []
    for paper in papers:
        text = paper.title + (" " + paper.abstract if paper.has_abstract else "")
        domain = classifier.classify(text)
        if domain and domain.lower() == args.domain.lower():
            domain_papers.append(paper)

    print(f"\n=== {args.domain.upper()} Domain Analysis ===")
    print(f"Total papers: {len(domain_papers)}")

    if not domain_papers:
        return

    # Analyze trends
    analyzer = TrendAnalyzer()
    yearly = analyzer.analyze_yearly_trends(domain_papers)

    print("\n=== Yearly Trends ===")
    for i, year in enumerate(yearly['years']):
        print(f"  {year}: {yearly['paper_counts'][i]} papers")

    # Keyword trends
    keywords = analyzer.analyze_keyword_trends(domain_papers, top_n=30)
    print("\n=== Top Keywords ===")
    for kw, count in list(keywords['keyword_total'].items())[:20]:
        print(f"  {kw}: {count}")

    # Emerging keywords
    print("\n=== Emerging Keywords ===")
    for kw, growth in keywords['emerging_keywords'][:10]:
        print(f"  {kw}: +{growth*100:.1f}% growth")

    # Subtopic analysis
    subtopic = SubtopicAnalyzer()
    subtopic_result = subtopic.analyze(domain_papers, year_range=args.years)

    print("\n=== Subtopics ===")
    for keyword in list(subtopic_result['top_keywords'])[:15]:
        print(f"  {keyword}")


def compare_conferences(args):
    """Compare multiple conferences."""
    loader = PaperDataLoader(args.data_dir)

    conf_data_dict = {}
    for conf in args.conferences:
        conf_data = loader.load_conference(conf, args.years)
        conf_data_dict[conf] = conf_data.papers

    comparator = ComparativeAnalyzer()
    comparison = comparator.compare_conferences(conf_data_dict)

    print(f"\n=== Conference Comparison ===")
    print(f"{'Conference':<15} {'Total Papers':<15} {'Avg/Year':<15}")
    print("-" * 45)

    for conf in args.conferences:
        print(f"{conf:<15} {comparison['total_papers'][conf]:<15} {comparison['avg_papers_per_year'][conf]:<15.1f}")

    # Keyword comparison
    print("\n=== Keyword Comparison ===")
    for conf in args.conferences:
        print(f"\n{conf.upper()}:")
        for kw in list(comparison['top_keywords'][conf].keys())[:10]:
            print(f"  {kw}: {comparison['top_keywords'][conf][kw]}")


def deep_analyze_domain(args):
    """Perform deep domain analysis with lifecycle, researcher stability, and vocabulary turnover."""
    # Show available domains
    if args.list_domains:
        print("\n=== Available Domain Definitions ===")
        for domain in DOMAIN_DEFINITIONS:
            print(f"  - {domain}")
        print("\nYou can also search for any custom keyword/phrase.")
        return

    # Initialize analyzer
    analyzer = DeepDomainAnalyzer(args.data_dir)

    # Parse conferences
    conferences = None
    if args.conferences:
        conferences = [c.strip() for c in args.conferences.split(',')]

    # Run analysis
    print(f"\n🔍 Analyzing domain: {args.domain}")
    print(f"   Years: {args.years}")
    if conferences:
        print(f"   Conferences: {conferences}")
    else:
        print(f"   Conferences: all CCF-A")

    report = analyzer.analyze_domain(
        domain=args.domain,
        conferences=conferences,
        years=args.years,
        min_relevance=args.min_relevance,
        top_papers=args.top_papers
    )

    # Print formatted report
    print(format_report(report))

    # === Lifecycle Analysis ===
    # S-curve analysis using yearly trends from report
    lifecycle = LifecycleAnalyzer()
    lifecycle_result = lifecycle.fit_scurve(report.yearly_trends)

    # Load papers for researcher stability and vocabulary turnover analysis
    # Parse conferences for paper loading
    if conferences is None:
        conferences = ['aaai', 'nips', 'iclr', 'icml', 'acl', 'cvpr', 'kdd', 'sigir']

    # Load all papers
    all_papers = load_papers_for_analysis(conferences, args.years)

    # Filter papers by domain relevance (same logic as DeepDomainAnalyzer)
    classifier = DomainClassifier()
    domain_papers = []
    domain_def = analyzer.domain_definitions.get(args.domain)
    if not domain_def:
        # Try partial match
        for defined_domain in analyzer.domain_definitions:
            if defined_domain.lower() in args.domain.lower() or args.domain.lower() in defined_domain.lower():
                domain_def = analyzer.domain_definitions[defined_domain]
                break

    if not domain_def:
        domain_def = {'keywords': args.domain.split(), 'subdomains': {}}

    keywords = domain_def.get('keywords', [])

    for paper in all_papers:
        text = paper.title
        if paper.abstract:
            text += " " + paper.abstract

        matched, score = analyzer._match_keywords(text, keywords)
        if score >= args.min_relevance:
            domain_papers.append(paper)

    # Researcher stability analysis
    stability_analyzer = ResearcherStabilityAnalyzer()
    stability_result = stability_analyzer.calculate_stability(domain_papers, args.years)

    # Vocabulary turnover analysis
    vocab_result = analyze_vocabulary_turnover(domain_papers, args.years)

    # Print lifecycle analysis results
    print(f"\n=== Lifecycle Analysis ===")
    print(f"Stage: {lifecycle_result.get('stage', 'unknown')}")
    if 'L' in lifecycle_result:
        print(f"Projected ceiling (L): {lifecycle_result['L']:.0f}")
        print(f"Growth rate (k): {lifecycle_result['k']:.2f}")
        print(f"R-squared: {lifecycle_result['r_squared']:.3f}")

    print(f"\n=== Researcher Stability ===")
    print(f"Avg Jaccard: {stability_result.get('avg_jaccard', 0):.3f}")
    print(f"Stage: {stability_result.get('stage', 'unknown')}")

    print(f"\n=== Rising Keywords ===")
    for kw in vocab_result.get('rising_keywords', [])[:5]:
        print(f"  {kw['word']}: +{kw['change']*100:.0f}%")

    print(f"\n=== Declining Keywords ===")
    for kw in vocab_result.get('declining_keywords', [])[:5]:
        print(f"  {kw['word']}: {kw['change']*100:.0f}%")

    # Save to file if requested
    if args.output:
        output_mgr = OutputManager(Path(args.output))

        # Convert report to dict
        report_dict = {
            'domain': report.domain,
            'year_range': report.year_range,
            'total_papers': report.total_papers,
            'papers_with_abstract': report.papers_with_abstract,
            'yearly_trends': report.yearly_trends,
            'venue_distribution': report.venue_distribution,
            'top_keywords': report.top_keywords,
            'subdomains': {
                name: {
                    'paper_count': sub.paper_count,
                    'growth_rate': sub.growth_rate,
                    'yearly_counts': sub.yearly_counts,
                    'top_keywords': sub.top_keywords,
                    'representative_papers': sub.representative_papers
                }
                for name, sub in report.subdomains.items()
            },
            'representative_papers': report.representative_papers,
            'emerging_topics': report.emerging_topics,
            'insights': report.insights,
            'lifecycle_analysis': {
                'stage': lifecycle_result.get('stage', 'unknown'),
                'L': lifecycle_result.get('L'),
                'k': lifecycle_result.get('k'),
                'r_squared': lifecycle_result.get('r_squared')
            },
            'researcher_stability': {
                'avg_jaccard': stability_result.get('avg_jaccard', 0),
                'stage': stability_result.get('stage', 'unknown')
            }
        }

        # Save JSON to deep category
        filepath = output_mgr.save_json(report_dict, 'deep', f"{args.domain.lower().replace(' ', '_')}_analysis.json")
        print(f"\n📁 Report saved to: {filepath}")


def interactive_mode(args):
    """Run in interactive mode."""
    print("\n=== CCF Paper Analysis - Interactive Mode ===")
    print("Type 'help' for available commands, 'exit' to quit\n")

    loader = PaperDataLoader(args.data_dir)

    while True:
        try:
            cmd = input("> ").strip()

            if cmd == "exit" or cmd == "quit":
                break
            elif cmd == "help":
                print("""
Available commands:
  conferences      - List available conferences
  stats           - Show overall statistics
  analyze <conf>  - Analyze a conference
  domain <name>   - Analyze a domain
  compare <conf1 conf2 ...> - Compare conferences
  exit            - Exit
                """)
            elif cmd == "conferences":
                list_conferences(args)
            elif cmd == "stats":
                show_statistics(args)
            elif cmd.startswith("analyze "):
                conf = cmd.split()[1]
                args.conference = conf
                args.years = [2022, 2023, 2024]
                args.keywords = True
                args.emerging = True
                args.domains = True
                args.authors = True
                args.output = None
                analyze_conference(args)
            elif cmd.startswith("domain "):
                domain = cmd.split()[1]
                args.domain = domain
                args.conferences = None
                args.years = [2022, 2023, 2024]
                analyze_domain(args)
            else:
                print("Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def timeline_mode(args):
    """Run vocabulary timeline analysis."""
    # Parse conferences
    conferences = None
    if args.conferences:
        conferences = [c.strip() for c in args.conferences.split(',')]

    # Load papers
    papers = load_papers_for_analysis(conferences, args.years)

    # Analyze
    analyzer = VocabularyTimeline(min_count=10)
    df = analyzer.analyze(papers, top_n=args.top_n)

    # Output
    print(f"\n=== Vocabulary Timeline ===")
    print(f"Total phrases analyzed: {len(df)}")
    print(df.head(20).to_string())

    # Save if output specified
    if args.output:
        output_mgr = OutputManager(Path(args.output))

        # Save CSV
        csv_path = output_mgr.save_csv(df.to_dict('records'), 'ecosystem', 'vocabulary_timeline.csv')

        # Save paradigm shifts as JSON
        shifts = analyzer.get_paradigm_shifts(df, top_n=20)
        json_path = output_mgr.save_json(shifts, 'ecosystem', 'paradigm_shifts.json')

        print(f"\nSaved to: {output_mgr.base_dir}")


def ecosystem_mode(args):
    """Run full ecosystem analysis."""
    # Parse conferences
    conferences = None
    if args.conferences:
        conferences = [c.strip() for c in args.conferences.split(',')]

    # Load papers
    papers = load_papers_for_analysis(conferences, args.years)

    print(f"\n=== Ecosystem Analysis ===")
    print(f"Analyzing {len(papers)} papers...")

    # Run modules
    # (can be extended based on needs)

    if args.output:
        output_mgr = OutputManager(Path(args.output))
        print(f"Results would be saved to: {output_mgr.base_dir}")


def network_mode(args):
    """Run coauthor network analysis."""
    # Parse conferences
    conferences = None
    if args.conferences:
        conferences = [c.strip() for c in args.conferences.split(',')]

    # Load papers
    papers = load_papers_for_analysis(conferences, args.years)

    # Analyze
    analyzer = CoauthorNetworkAnalyzer()
    df = analyzer.analyze_evolution(papers, args.years)

    # Output
    print(f"\n=== Network Evolution ===")
    print(df.to_string())

    if args.output:
        output_mgr = OutputManager(Path(args.output))
        csv_path = output_mgr.save_csv(df.to_dict('records'), 'network', 'network_evolution.csv')
        print(f"\nSaved to: {output_mgr.base_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CCF Conference Paper Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available conferences
  python -m analysis list

  # Show statistics
  python -m analysis stats

  # Analyze a conference
  python -m analysis analyze aaai --years 2022,2023,2024 --keywords --domains

  # Deep domain analysis (AI Agent, Federated Learning, LLM, etc.)
  python -m analysis deep "AI Agent" --years 2023,2024,2025
  python -m analysis deep "Federated Learning" --years 2023,2024,2025
  python -m analysis deep "Large Language Models" --years 2023,2024,2025
  python -m analysis deep "RAG" --years 2023,2024,2025

  # List available domain definitions
  python -m analysis deep --list-domains

  # Custom domain search
  python -m analysis deep "your keyword" --min-relevance 0.05

  # Compare conferences
  python -m analysis compare aaai nips acl --years 2023,2024

  # Interactive mode
  python -m analysis interactive

  # Vocabulary timeline analysis
  python -m analysis timeline --years 2020,2021,2022 --top-n 50

  # Ecosystem analysis
  python -m analysis ecosystem --years 2020,2021,2022

  # Network analysis
  python -m analysis network --years 2020,2021
        """
    )

    parser.add_argument(
        '--data-dir',
        default=None,
        help='Path to data directory (default: data/paper)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List command
    list_parser = subparsers.add_parser('list', help='List available conferences')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show overall statistics')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a conference')
    analyze_parser.add_argument('conference', help='Conference name (e.g., aaai, nips)')
    analyze_parser.add_argument('--years', type=str, default='2022,2023,2024', help='Years to analyze')
    analyze_parser.add_argument('--keywords', action='store_true', help='Show keywords')
    analyze_parser.add_argument('--emerging', action='store_true', help='Show emerging keywords')
    analyze_parser.add_argument('--domains', action='store_true', help='Show domain distribution')
    analyze_parser.add_argument('--authors', action='store_true', help='Show top authors')
    analyze_parser.add_argument('--output', type=str, help='Output directory for reports')

    # Domain command
    domain_parser = subparsers.add_parser('domain', help='Analyze a domain')
    domain_parser.add_argument('domain', help='Domain name (e.g., machine learning)')
    domain_parser.add_argument('--conferences', type=str, help='Comma-separated conference list')
    domain_parser.add_argument('--years', type=str, default='2022,2023,2024', help='Years to analyze')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare conferences')
    compare_parser.add_argument('conferences', nargs='+', help='Conference names')
    compare_parser.add_argument('--years', type=str, default='2023,2024', help='Years to compare')

    # Deep domain analysis command
    deep_parser = subparsers.add_parser('deep', help='Deep domain analysis (AI Agent, Federated Learning, etc.)')
    deep_parser.add_argument('domain', nargs='?', help='Domain name (e.g., "AI Agent", "Federated Learning")')
    deep_parser.add_argument('--list-domains', action='store_true', help='List available domain definitions')
    deep_parser.add_argument('--conferences', type=str, default=None, help='Comma-separated conference list (default: all CCF-A)')
    deep_parser.add_argument('--years', type=str, default='2023,2024,2025', help='Years to analyze')
    deep_parser.add_argument('--min-relevance', type=float, default=0.1, help='Minimum relevance score (0-1)')
    deep_parser.add_argument('--top-papers', type=int, default=10, help='Number of representative papers')
    deep_parser.add_argument('--output', type=str, default=None, help='Output directory for JSON report')

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Run in interactive mode')

    # Timeline command
    timeline_parser = subparsers.add_parser('timeline', help='Analyze vocabulary timeline')
    timeline_parser.add_argument('--years', type=str, default='2015,2025')
    timeline_parser.add_argument('--top-n', type=int, default=100)
    timeline_parser.add_argument('--conferences', type=str, default=None)
    timeline_parser.add_argument('--output', type=str, default=None)

    # Ecosystem command
    ecosystem_parser = subparsers.add_parser('ecosystem', help='Run full ecosystem analysis')
    ecosystem_parser.add_argument('--years', type=str, default='2015,2025')
    ecosystem_parser.add_argument('--conferences', type=str, default=None)
    ecosystem_parser.add_argument('--output', type=str, default=None)

    # Network command
    network_parser = subparsers.add_parser('network', help='Analyze coauthor network')
    network_parser.add_argument('--years', type=str, default='2015,2025')
    network_parser.add_argument('--conferences', type=str, default=None)
    network_parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    # Set default data directory
    if args.data_dir is None:
        root_dir = Path(__file__).parent.parent
        args.data_dir = root_dir / "data" / "paper"

    # Parse years
    if hasattr(args, 'years') and isinstance(args.years, str):
        args.years = [int(y) for y in args.years.split(',')]

    # Route to appropriate function
    if args.command == 'list':
        list_conferences(args)
    elif args.command == 'stats':
        show_statistics(args)
    elif args.command == 'analyze':
        analyze_conference(args)
    elif args.command == 'domain':
        analyze_domain(args)
    elif args.command == 'compare':
        compare_conferences(args)
    elif args.command == 'deep':
        deep_analyze_domain(args)
    elif args.command == 'interactive':
        interactive_mode(args)
    elif args.command == 'timeline':
        timeline_mode(args)
    elif args.command == 'ecosystem':
        ecosystem_mode(args)
    elif args.command == 'network':
        network_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
