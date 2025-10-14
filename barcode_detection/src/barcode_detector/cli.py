"""Command Line Interface for barcode detector."""

import sys
from .main import main

def cli_main():
    """CLI entry point."""
    sys.exit(main())

if __name__ == "__main__":
    cli_main()