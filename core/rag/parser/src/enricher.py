"""
Single-pass LLM transformation pipeline for parsed workbook markdowns.

For each workbook markdown in the input directory:
  1. Send it to the LLM with a fresh session (no shared state).
  2. Apply numeric spell-out, table flattening, and symbol expansion.
  3. Save the plain-text output to <output>/enriched/<workbook>.md.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import traceback
import uuid
from pathlib import Path
from typing import Iterable


# ── Load OllamaChat from repo-root `src/ai/localllm.py` ─────────────────────
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]
_LOCALLLM_PATH = _REPO_ROOT / "src" / "ai" / "localllm.py"

if not _LOCALLLM_PATH.exists():
    raise ImportError(f"Cannot find localllm at {_LOCALLLM_PATH}")

_spec = importlib.util.spec_from_file_location("_aero_localllm", _LOCALLLM_PATH)
_localllm = importlib.util.module_from_spec(_spec)
sys.modules["_aero_localllm"] = _localllm
_spec.loader.exec_module(_localllm)  # type: ignore[union-attr]
OllamaChat = _localllm.OllamaChat


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a data enrichment pipeline. Your job is to transform structured business data into clean, spoken-friendly plain text suitable for a voicebot RAG system.

You will receive a document containing structured business data. Apply the following transformations and output plain text only. No markdown headers, no bullet points, no asterisks, no dashes, no hash symbols.

Apply ALL of the following transformations:

1. NUMERIC / PRICE / DATE / TIME SPELL-OUT
   - "$19.90" → "nineteen dollars and ninety cents"
   - "$1,250.99" → "one thousand two hundred and fifty dollars and ninety-nine cents"
   - "90 min" → "ninety minutes"
   - "52 inches" → "fifty-two inches"
   - "2026-04-24" → "April twenty-fourth, twenty twenty-six"
   - Phone numbers, percentages, ages, capacities — all spelled out in words.

2. TABLE FLATTENING
   - Convert every table or bulleted list into natural-language sentences and paragraphs.
   - Every data point must appear in a complete sentence. Example: "The Main Track standalone race costs nineteen dollars and ninety cents, requires drivers to be at least fifty-two inches tall, and gives you up to ten laps on the track."

3. SYMBOL / UNIT / ACRONYM EXPANSION
   - "%" → "percent", "&" → "and", "@" → "at", "#" → "number"
   - "kg" → "kilograms", "ft" → "feet", "hrs" → "hours", "min" → "minutes" (or "minimum" depending on context)

4. PRESERVE TRUTH — Do not invent or assume any numbers, prices, rules, or policies. Keep every price, duration, age range, height requirement, and rule exactly as given.

Output ONLY plain text paragraphs. No markdown syntax of any kind. No preamble. No "Here is the enriched version:" framing."""


TRANSFORM_PROMPT = """\
Transform the data below applying these rules exactly:
1. Spell out all numbers, prices, dates, and times in words.
2. Flatten all tables and lists into complete sentences.
3. Expand all symbols, units, and acronyms.
4. No markdown — no headers, no bullets, no asterisks, no hash symbols.
5. Preserve every fact exactly as given. Do not add or remove information.
Output ONLY the transformed plain text.

----- ORIGINAL DATA -----
{content}
----- END DATA -----"""


# ── Transformer ───────────────────────────────────────────────────────────────


class Enricher:
    """Single-pass LLM transformation over a directory of parsed markdowns."""

    ENRICHED_SUBDIR = "enriched"
    ORIGINAL_SUBDIR = "original"

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        model: str = "qwen3:4b",
        base_url: str = "http://192.168.50.150:11434",
        db_url: str = "sqlite:///enrichment_memory.db",
        user_id: str = "enricher",
        run_id: str | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir
        self.enriched_dir = self.output_dir / self.ENRICHED_SUBDIR
        self.original_dir = self.output_dir / self.ORIGINAL_SUBDIR

        self.model = model
        self.base_url = base_url
        self.db_url = db_url
        self.user_id = user_id
        self.run_id = run_id or uuid.uuid4().hex[:8]

        self._llm = OllamaChat(
            model=model,
            base_url=base_url,
            db_url=db_url,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.2,
        )

    def _discover_workbooks(self) -> list[Path]:
        return [
            p
            for p in sorted(self.input_dir.glob("*.md"))
            if not p.name.startswith("merged_")
        ]

    async def transform_workbook(self, wb: Path) -> Path:
        session_id = f"transform-{self.run_id}-{wb.stem}"
        self._llm.clear_history(session_id, self.user_id)

        content = wb.read_text(encoding="utf-8")
        prompt = TRANSFORM_PROMPT.format(content=content)

        result = await self._llm.achat_with_history(
            prompt, session_id=session_id, user_id=self.user_id
        )

        self.enriched_dir.mkdir(parents=True, exist_ok=True)
        self.original_dir.mkdir(parents=True, exist_ok=True)

        out_path = self.enriched_dir / wb.name
        out_path.write_text(result.strip() + "\n", encoding="utf-8")

        original_copy = self.original_dir / wb.name
        original_copy.write_text(content, encoding="utf-8")

        return out_path

    async def run_all(self) -> dict:
        workbooks = self._discover_workbooks()
        if not workbooks:
            print(f"No workbook markdowns found in {self.input_dir}")
            return {"transformed": []}

        import shutil
        for d in (self.enriched_dir, self.original_dir):
            if d.exists():
                shutil.rmtree(d)

        print("=" * 60)
        print(f"AeroSports RAG — Transform (run_id={self.run_id})")
        print(f"  model      : {self.model}")
        print(f"  input_dir  : {self.input_dir}")
        print(f"  output_dir : {self.output_dir}")
        print(f"  workbooks  : {len(workbooks)}")
        print("=" * 60)

        transformed: list[Path] = []
        failures: list[tuple[str, str]] = []

        for wb in workbooks:
            print(f"  transforming  {wb.name}")
            try:
                out = await self.transform_workbook(wb)
                transformed.append(out)
                print(f"    -> {out}")
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"    !! failed: {exc!r}")
                print(tb)
                failures.append((wb.name, repr(exc)))

        print("\nDone.")
        print(f"  transformed: {len(transformed)} / {len(workbooks)}")
        if failures:
            print(f"  FAILED:      {len(failures)}")
            for name, err in failures:
                print(f"    - {name}: {err}")
        print("=" * 60)

        if failures:
            raise RuntimeError(
                f"Transform failed for {len(failures)}/{len(workbooks)} workbooks: "
                + ", ".join(n for n, _ in failures)
            )

        return {"transformed": [str(p) for p in transformed]}


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run LLM transformation over parsed workbook markdowns."
    )
    ap.add_argument(
        "--input-dir",
        default="output",
        help="Directory containing parsed workbook .md files.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Directory for transformed outputs (defaults to input-dir).",
    )
    ap.add_argument("--model", default="qwen3:4b", help="Ollama model name.")
    ap.add_argument(
        "--base-url", default="http://192.168.50.150:11434", help="Ollama base URL."
    )
    ap.add_argument(
        "--run-id", default=None, help="Custom run id (default: random 8-char hex)."
    )
    args = ap.parse_args()

    enricher = Enricher(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        base_url=args.base_url,
        run_id=args.run_id,
    )
    asyncio.run(enricher.run_all())


if __name__ == "__main__":
    main()
