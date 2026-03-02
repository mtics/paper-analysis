#!/usr/bin/env python3
"""
Main entry point for analysis package.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.main import main

if __name__ == "__main__":
    main()
