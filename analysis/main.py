#!/usr/bin/env python3
"""
Analysis CLI for CCF conference papers.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.data_loader import PaperDataLoader, load_papers_for_analysis
from analysis.preprocessing import TextPreprocessor
from analysis.topic_model import DomainClassifier, TopicModeler, SubtopicAnalyzer, classify_papers_by_domain
from analysis.trend_analysis import TrendAnalyzer, ComparativeAnalyzer, generate_trend_report
from analysis.deep_domain import DeepDomainAnalyzer, format_report, DOMAIN_DEFINITIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    """Show overall statistics."""
    loader = PaperDataLoader(args.data_dir)
    stats = loader.get_statistics()

    print("\n=== Overall Statistics ===")
    print(f"Total conferences: {stats['total_conferences']}")
    print(f"Total papers: {stats['total_papers']}")
    print(f"Papers with abstract: {stats['papers_with_abstract']}")
    print(f"Coverage rate: {stats['papers_with_abstract']/stats['total_papers']*100:.1f}%")

    print("\n=== By Conference ===")
    for conf, conf_stats in stats['conferences'].items():
        print(f"  {conf}: {conf_stats['papers']} papers ({conf_stats['with_abstract']} with abstract)")


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
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        report = generate_trend_report(papers, args.conference)

        # Save JSON report
        report_file = output_path / f"{args.conference}_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'conference': report.conference,
                'year_range': report.year_range,
                'total_papers': report.total_papers,
                'papers_with_abstract': report.papers_with_abstract,
                'yearly_counts': report.yearly_counts,
                'domain_distribution': report.domain_distribution,
                'top_keywords': report.top_keywords
            }, f, indent=2)

        print(f"\nReport saved to: {report_file}")


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
    """Perform deep domain analysis."""
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

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

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
            'insights': report.insights
        }

        # Save JSON
        json_file = output_path / f"{args.domain.lower().replace(' ', '_')}_analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        print(f"\n📁 Report saved to: {json_file}")


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
  stats            - Show overall statistics
  analyze <conf>   - Analyze a conference
  domain <name>    - Analyze a domain
  compare <conf1 conf2 ...> - Compare conferences
  exit             - Exit
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
