"""
Parse all worksheets from the AeroSports Google Sheet into Markdown files,
optionally followed by an LLM-based enrichment pass for RAG/voicebot use.

Usage:
    python run_parsers.py                     # Download if needed, then parse
    python run_parsers.py --force             # Force re-download before parsing
    python run_parsers.py --no-download       # Skip download, use cached .xlsx
    python run_parsers.py --enrich            # Also run LLM enrichment after parsing
    python run_parsers.py --enrich-only       # Skip parsing, only run enrichment
    python run_parsers.py --model qwen3:4b    # Override enrichment model
"""

import argparse
import asyncio
from pathlib import Path

from src.sheet_downloader import download_sheet
from src.enricher import Enricher
from src import (
    LocationInfoParser,
    JumpPricesParser,
    GoKartingParser,
    SpecialProgramsParser,
    AttractionsParser,
    BirthdayPartiesParser,
    GroupBookingsParser,
    AeroCampParser,
    PassesParser,
    PromotionsParser,
    ParkRulesParser,
    FAQsParser,
    VoiceCallScriptsParser,
    ChatbotQuickRepliesParser,
)

SPREADSHEET_ID = "1SGK35TNPDZ8ipEsCe9w1wCmFQNBOTWo1b0ZaWVIa14s"
EXCEL_FILE = f"data/{SPREADSHEET_ID}.xlsx"
OUTPUT_DIR = "output"

PARSERS = [
    ("Location Info", LocationInfoParser),
    ("Jump Prices", JumpPricesParser),
    ("Go Karting", GoKartingParser),
    ("Special Programs", SpecialProgramsParser),
    ("Attractions", AttractionsParser),
    ("Birthday Parties", BirthdayPartiesParser),
    ("Group Bookings", GroupBookingsParser),
    ("Aero Camp", AeroCampParser),
    ("Passes", PassesParser),
    ("Promotions", PromotionsParser),
    ("Park Rules", ParkRulesParser),
    ("FAQs", FAQsParser),
    ("Voice Call Scripts", VoiceCallScriptsParser),
    ("Chatbot Quick Replies", ChatbotQuickRepliesParser),
]


def run(
    force_download: bool = False,
    skip_download: bool = False,
    enrich: bool = False,
    enrich_only: bool = False,
    model: str = "phi4:latest",
    base_url: str = "http://192.168.50.150:11434",
):
    print("=" * 60)
    print("AeroSports RAG — Sheet Parser")
    print("=" * 60)

    if not enrich_only:
        # Download step
        if not skip_download:
            print("\nChecking for sheet updates…")
            path = download_sheet(SPREADSHEET_ID, force=force_download)
            if path:
                print(f"Downloaded → {path}")
            else:
                print("Sheet unchanged, using cached file.")
        else:
            print("\nSkipping download (--no-download).")

        excel_path = Path(EXCEL_FILE)
        if not excel_path.exists():
            print(f"\nERROR: {EXCEL_FILE} not found. Run without --no-download first.")
            return

        # Parse step
        print(f"\nParsing {len(PARSERS)} sheets → {OUTPUT_DIR}/\n")
        created = []
        for name, ParserClass in PARSERS:
            parser = ParserClass(EXCEL_FILE, OUTPUT_DIR)
            result = parser.parse()
            status = "✓" if result["path"] else "–"
            print(f"  {status}  {name:30s}  →  {result['output_file']}")
            if result["path"]:
                created.append(result["output_file"])

        print(f"\nDone. {len(created)} files written to {OUTPUT_DIR}/")

    # Enrichment step
    if enrich or enrich_only:
        enricher = Enricher(
            input_dir=OUTPUT_DIR,
            output_dir=OUTPUT_DIR,
            model=model,
            base_url=base_url,
        )
        asyncio.run(enricher.run_all())

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Parse AeroSports Google Sheet to Markdown"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download the sheet"
    )
    parser.add_argument("--no-download", action="store_true", help="Skip download step")
    parser.add_argument(
        "--enrich", action="store_true", help="Run LLM enrichment after parsing"
    )
    parser.add_argument(
        "--enrich-only",
        action="store_true",
        help="Skip parsing, run only enrichment over existing output/",
    )
    parser.add_argument(
        "--model", default="phi4:latest", help="Ollama model for enrichment"
    )
    parser.add_argument(
        "--base-url", default="http://192.168.50.150:11434", help="Ollama base URL"
    )
    args = parser.parse_args()
    run(
        force_download=args.force,
        skip_download=args.no_download,
        enrich=args.enrich,
        enrich_only=args.enrich_only,
        model=args.model,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
