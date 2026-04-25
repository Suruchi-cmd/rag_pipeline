"""
LLM-based enrichment pipeline for parsed workbook markdowns.

Two-phase process:

  Step 1 — Data Understanding (one shared session):
      Feed every workbook markdown to the LLM using the same `session_id`
      so it accumulates a holistic, cross-workbook view. Final turn asks
      for a consolidated understanding document saved as `data_understanding.md`.

  Step 2 — Per-workbook Enrichment (fresh session each):
      For each workbook spin up an isolated `session_id`, prime it with the
      Step 1 understanding doc, then send the workbook markdown to be
      rewritten into voicebot- and RAG-friendly prose (table flattening,
      number/date/symbol spell-out, acronym expansion, chunk self-containment,
      etc.). Output lands in `<output>/enriched/<workbook>.md`.
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


# ── Load OllamaChat from repo-root `src/ai/localllm.py` without colliding
#    with this package's own `src/` directory.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]  # core/rag/parser/src/enricher.py -> repo root
_LOCALLLM_PATH = _REPO_ROOT / "src" / "ai" / "localllm.py"

if not _LOCALLLM_PATH.exists():
    raise ImportError(f"Cannot find localllm at {_LOCALLLM_PATH}")

_spec = importlib.util.spec_from_file_location("_aero_localllm", _LOCALLLM_PATH)
_localllm = importlib.util.module_from_spec(_spec)
sys.modules["_aero_localllm"] = _localllm
_spec.loader.exec_module(_localllm)  # type: ignore[union-attr]
OllamaChat = _localllm.OllamaChat


# ── Prompts ──────────────────────────────────────────────────────────────────

UNDERSTANDING_SYSTEM_PROMPT = """You are a senior data analyst preparing a knowledge base for a
Retrieval-Augmented Generation (RAG) voicebot that answers customer questions for AeroSports
Scarborough (a family entertainment park).

You will receive a series of workbook markdown documents, one per turn, that together describe
the full business: pricing, attractions, policies, FAQs, scripts, etc. Build and maintain a
holistic understanding across all workbooks in this conversation — note shared schemas,
recurring entities, cross-references, and overlaps.

For each incoming workbook, respond with a short (5-10 line) structured note covering:
  - Purpose of the workbook
  - Key entities / columns / sections
  - Relationships to prior workbooks you have already seen
  - Anything ambiguous or likely to confuse a voicebot

Keep responses concise; depth comes from the final consolidated document."""


CONSOLIDATION_REQUEST = """You have now seen every workbook in the dataset.

Produce a single consolidated UNDERSTANDING DOCUMENT in markdown that will serve as the
grounding context for a downstream enrichment pass. It MUST include:

1. High-level overview of the business and the dataset as a whole.
2. Per-workbook sections (`## <Workbook Name>`) with:
   - One-paragraph summary of its purpose.
   - Key entities, columns, and terminology used.
   - How this workbook relates to the others (shared IDs, overlapping fields,
     references, dependencies).
3. A "Shared Vocabulary" section listing acronyms, abbreviations, codes, units,
   and domain terms a voicebot should expand when speaking to customers
   (e.g., "min" → "minutes" or "min" -> "minimum (depending on context), "$" → "dollars").
4. A "Cross-workbook Relationships" section describing how concepts connect
   across workbooks (e.g., a pass mentioned in Jump Prices also appears in Passes).
5. A "Voicebot Considerations" section flagging tables, cryptic codes, dense
   markdown, ambiguous pronouns, or anything that will need heavy rewriting.

Output ONLY the markdown document. No preamble, no trailing commentary."""


ENRICHMENT_SYSTEM_PROMPT = """You are a friendly, knowledgeable customer service representative at AeroSports Scarborough (a family entertainment park). Your job is to explain business information to customers clearly and helpfully — as if speaking to them face-to-face or over the phone.

You will receive a document containing structured business data. Rewrite it as plain text only. No markdown headers, no bullet points, no asterisks, no dashes, no hash symbols. Use natural paragraphs only.

Structure your response as follows:
- First paragraph: A clear overview of what this section covers — what types or options are available at AeroSports Scarborough and who they suit.
- Subsequent paragraphs: One paragraph per major item or section, explaining it in detail exactly as you would explain it to a customer standing in front of you.

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

4. BE SPECIFIC — never vague. Do not use filler phrases like "exciting opportunity", "enhance your visit", or "perfect for those looking to". Say exactly what it is, what it costs, who it is for, and any requirements or conditions.

5. PRESERVE TRUTH — Do not invent or assume any numbers, prices, rules, or policies. Keep every price, duration, age range, height requirement, and rule exactly as given.

Output ONLY plain text paragraphs. No markdown syntax of any kind. No preamble. No "Here is the explanation:" framing."""


# ── Enricher ─────────────────────────────────────────────────────────────────


class Enricher:
    """Runs the two-phase LLM enrichment over a directory of parsed markdowns."""

    UNDERSTANDING_FILE = "data_understanding.md"
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
        self.understanding_path = self.output_dir / self.UNDERSTANDING_FILE

        self.model = model
        self.base_url = base_url
        self.db_url = db_url
        self.user_id = user_id
        # Run id keeps session_ids unique across reruns so old history
        # does not contaminate a fresh enrichment pass.
        self.run_id = run_id or uuid.uuid4().hex[:8]

        self._understanding_llm = OllamaChat(
            model=model,
            base_url=base_url,
            db_url=db_url,
            system_prompt=UNDERSTANDING_SYSTEM_PROMPT,
            temperature=0.2,
        )
        self._enrichment_llm = OllamaChat(
            model=model,
            base_url=base_url,
            db_url=db_url,
            system_prompt=ENRICHMENT_SYSTEM_PROMPT,
            temperature=0.2,
        )

    # ── Discovery ────────────────────────────────────────────────────────

    def _discover_workbooks(self) -> list[Path]:
        """All .md files in input_dir except our own outputs.

        Only scans input_dir at top level — the `enriched/` and `original/`
        subdirs are intentionally ignored.
        """
        excluded = {self.UNDERSTANDING_FILE}
        files = [
            p
            for p in sorted(self.input_dir.glob("*.md"))
            if p.name not in excluded and not p.name.startswith("merged_")
        ]
        return files

    # ── Step 1: Understanding (shared session) ───────────────────────────

    async def build_understanding(self, workbooks: Iterable[Path]) -> Path:
        session_id = f"understanding-{self.run_id}"
        # Ensure clean slate in case a prior run reused this id.
        self._understanding_llm.clear_history(session_id, self.user_id)

        print(f"\n[Step 1] Understanding phase — session={session_id}")
        for wb in workbooks:
            content = wb.read_text(encoding="utf-8")
            prompt = (
                f"Workbook filename: `{wb.name}`\n\n"
                "Below is the full markdown for this workbook. Ingest it into your running "
                "understanding of the dataset, then reply with the short structured note "
                "described in your instructions.\n\n"
                "----- WORKBOOK MARKDOWN START -----\n"
                f"{content}\n"
                "----- WORKBOOK MARKDOWN END -----"
            )
            print(f"  ingesting  {wb.name} ({len(content):,} chars)")
            reply = await self._understanding_llm.achat_with_history(
                prompt, session_id=session_id, user_id=self.user_id
            )
            print(
                f"    note: {reply.strip().splitlines()[0][:120] if reply.strip() else '(empty)'}"
            )

        print("  consolidating …")
        doc = await self._understanding_llm.achat_with_history(
            CONSOLIDATION_REQUEST, session_id=session_id, user_id=self.user_id
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.understanding_path.write_text(doc.strip() + "\n", encoding="utf-8")
        print(f"  wrote      {self.understanding_path}")
        return self.understanding_path

    # ── Step 2: Enrichment (fresh session per workbook) ──────────────────

    async def enrich_workbook(self, wb: Path, understanding_md: str) -> Path:
        session_id = f"enrich-{self.run_id}-{wb.stem}"
        self._enrichment_llm.clear_history(session_id, self.user_id)

        # Prime the fresh session with the consolidated understanding so
        # the model has cross-workbook context before the rewrite turn.
        priming = (
            "The following is the CONSOLIDATED UNDERSTANDING of the full dataset. "
            "Treat it as ground-truth context. Do NOT rewrite it — just acknowledge "
            "in one sentence that you have read it, then wait for the workbook to enrich.\n\n"
            "----- UNDERSTANDING START -----\n"
            f"{understanding_md}\n"
            "----- UNDERSTANDING END -----"
        )
        await self._enrichment_llm.achat_with_history(
            priming, session_id=session_id, user_id=self.user_id
        )

        original = wb.read_text(encoding="utf-8")
        enrich_prompt = (
            f"Now explain the data in `{wb.name}` to a customer. "
            "Write as a customer service representative speaking directly to the customer. "
            "Plain text paragraphs only — no markdown, no headers, no bullets, no asterisks. "
            "First paragraph: overview of what types or options are available and who they suit. "
            "Then one paragraph per major section or item with full specific details. "
            "Spell out all numbers, prices, dates, and times in words. "
            "Flatten all tables and lists into complete sentences. "
            "Expand all symbols, units, and acronyms. "
            "Be specific — state exact prices, heights, durations, and conditions. "
            "Preserve every fact. Output ONLY the plain text explanation.\n\n"
            "----- ORIGINAL DATA -----\n"
            f"{original}\n"
            "----- END DATA -----"
        )
        enriched = await self._enrichment_llm.achat_with_history(
            enrich_prompt, session_id=session_id, user_id=self.user_id
        )

        self.enriched_dir.mkdir(parents=True, exist_ok=True)
        self.original_dir.mkdir(parents=True, exist_ok=True)

        out_path = self.enriched_dir / wb.name
        out_path.write_text(enriched.strip() + "\n", encoding="utf-8")

        # Mirror the untouched source into a parallel `original/` folder so
        # the two trees line up file-for-file for easy diffing / review.
        original_copy = self.original_dir / wb.name
        original_copy.write_text(original, encoding="utf-8")
        return out_path

    # ── Orchestration ────────────────────────────────────────────────────

    async def run_all(self) -> dict:
        workbooks = self._discover_workbooks()
        if not workbooks:
            print(f"No workbook markdowns found in {self.input_dir}")
            return {"understanding": None, "enriched": []}

        print("=" * 60)
        print(f"AeroSports RAG — Enrichment (run_id={self.run_id})")
        print(f"  model      : {self.model}")
        print(f"  input_dir  : {self.input_dir}")
        print(f"  output_dir : {self.output_dir}")
        print(f"  workbooks  : {len(workbooks)}")
        print("=" * 60)

        understanding_path = await self.build_understanding(workbooks)
        understanding_md = understanding_path.read_text(encoding="utf-8")

        print(f"\n[Step 2] Enrichment phase — {len(workbooks)} workbook(s)")
        enriched_paths: list[Path] = []
        failures: list[tuple[str, str]] = []
        for wb in workbooks:
            print(f"  enriching  {wb.name}")
            try:
                out = await self.enrich_workbook(wb, understanding_md)
                enriched_paths.append(out)
                print(f"    -> {out}")
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"    !! failed: {exc!r}")
                # Emit full traceback so upstream log stream captures the
                # root cause (e.g., LLM 502 HTML payload, timeout, etc.).
                print(tb)
                failures.append((wb.name, repr(exc)))

        print("\nDone.")
        print(f"  understanding: {understanding_path}")
        print(f"  enriched:      {len(enriched_paths)} / {len(workbooks)}")
        if failures:
            print(f"  FAILED:        {len(failures)}")
            for name, err in failures:
                print(f"    - {name}: {err}")
        print("=" * 60)

        # Non-zero exit if any workbook failed so upstream (resync) can
        # report the run as failed instead of silently succeeding.
        if failures:
            raise RuntimeError(
                f"Enrichment failed for {len(failures)}/{len(workbooks)} workbooks: "
                + ", ".join(n for n, _ in failures)
            )

        return {
            "understanding": str(understanding_path),
            "enriched": [str(p) for p in enriched_paths],
        }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run LLM enrichment over parsed workbook markdowns."
    )
    ap.add_argument(
        "--input-dir",
        default="output",
        help="Directory containing parsed workbook .md files.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Directory for enrichment outputs (defaults to input-dir).",
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
