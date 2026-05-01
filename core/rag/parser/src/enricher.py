"""
LLM-based minimal enrichment for parsed workbook markdowns.

For each workbook we run a single fresh LLM session that converts any
markdown tables into natural-language paragraphs / document prose while
keeping every other fact, header, bullet, price, and URL untouched.
Output lands in `<output>/enriched/<workbook>.md`; the source is mirrored
into `<output>/original/<workbook>.md` for easy diffing.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import traceback
import uuid
from pathlib import Path


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


ENRICHMENT_SYSTEM_PROMPT = """You are a technical writer converting structured workbook
markdown into clean prose for a Retrieval-Augmented Generation (RAG) knowledge base for
AeroSports Scarborough (a family entertainment park).

Your job is to (a) add a short page-level summary, and (b) convert markdown tables into
natural-language paragraphs. Everything else stays as-is.

Rules:

1. PAGE SUMMARY
   - Immediately after the top-level `# Title` heading (and before any `##` section),
     insert a short summary paragraph (2-4 sentences) describing what this page covers
     at AeroSports Scarborough. Use only facts that appear in the document — do NOT
     invent details, prices, or policies. Example shape:
       "At AeroSports Scarborough we offer ... This page covers ..."
   - If the document already has a summary paragraph in that position, keep it as-is.

2. TABLE FLATTENING
   - Replace every markdown table with paragraphs or bullets that mention each column
     by name. One row typically becomes one sentence or one bullet.
   - Example row "| Premium Jump Pass | 90 min | $24.90 |" becomes:
     "The Premium Jump Pass lasts 90 min and costs $24.90."

3. PRESERVE EVERYTHING ELSE
   - Keep all section headings (`#`, `##`, `###`) exactly as written.
   - Keep existing bullet lists, key-value bullets (`- **Field:** value`), notes,
     prices, URLs, phone numbers, codes, and any other formatting unchanged.
   - Do NOT spell out numbers, dates, prices, or units. Do NOT expand acronyms.
     Do NOT rephrase non-table prose.
   - Apart from the page summary in rule 1, do NOT add new facts, intros, or commentary.

4. PRESERVE TRUTH
   - Every price, duration, age range, rule, URL, and proper noun must survive
     verbatim. If a cell is empty, omit it from the sentence rather than inventing.

Output ONLY the rewritten markdown document. No preamble, no trailing commentary,
no "Here is the enriched version:" framing."""


class Enricher:
    """Runs minimal LLM enrichment over a directory of parsed markdowns."""

    ENRICHED_SUBDIR = "enriched"
    ORIGINAL_SUBDIR = "original"

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        model: str = "qwen3:4b",
        base_url: str = "https://aerollm.share.zrok.io",
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
            system_prompt=ENRICHMENT_SYSTEM_PROMPT,
            temperature=0.2,
        )

    def _discover_workbooks(self) -> list[Path]:
        return [
            p
            for p in sorted(self.input_dir.glob("*.md"))
            if not p.name.startswith("merged_")
        ]

    async def enrich_workbook(self, wb: Path) -> Path:
        session_id = f"enrich-{self.run_id}-{wb.stem}"
        self._llm.clear_history(session_id, self.user_id)

        original = wb.read_text(encoding="utf-8")
        prompt = (
            f"Rewrite the workbook `{wb.name}` below. Add a short 2-4 sentence "
            "summary paragraph right after the `# Title` heading describing what this "
            "page covers at AeroSports Scarborough (using only facts present in the "
            "document). Convert every markdown table into natural-language paragraphs "
            "(or short bullets where appropriate). Leave all other content unchanged. "
            "Preserve every fact verbatim. Output ONLY the rewritten markdown.\n\n"
            "----- ORIGINAL WORKBOOK MARKDOWN -----\n"
            f"{original}\n"
            "----- END ORIGINAL -----"
        )
        enriched = await self._llm.achat_with_history(
            prompt, session_id=session_id, user_id=self.user_id
        )

        self.enriched_dir.mkdir(parents=True, exist_ok=True)
        self.original_dir.mkdir(parents=True, exist_ok=True)

        out_path = self.enriched_dir / wb.name
        out_path.write_text(enriched.strip() + "\n", encoding="utf-8")

        original_copy = self.original_dir / wb.name
        original_copy.write_text(original, encoding="utf-8")
        return out_path

    async def run_all(self) -> dict:
        workbooks = self._discover_workbooks()
        if not workbooks:
            print(f"No workbook markdowns found in {self.input_dir}")
            return {"enriched": []}

        print("=" * 60)
        print(f"AeroSports RAG — Minimal Enrichment (run_id={self.run_id})")
        print(f"  model      : {self.model}")
        print(f"  input_dir  : {self.input_dir}")
        print(f"  output_dir : {self.output_dir}")
        print(f"  workbooks  : {len(workbooks)}")
        print("=" * 60)

        enriched_paths: list[Path] = []
        failures: list[tuple[str, str]] = []
        for wb in workbooks:
            print(f"  enriching  {wb.name}")
            try:
                out = await self.enrich_workbook(wb)
                enriched_paths.append(out)
                print(f"    -> {out}")
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"    !! failed: {exc!r}")
                print(tb)
                failures.append((wb.name, repr(exc)))

        print("\nDone.")
        print(f"  enriched:      {len(enriched_paths)} / {len(workbooks)}")
        if failures:
            print(f"  FAILED:        {len(failures)}")
            for name, err in failures:
                print(f"    - {name}: {err}")
        print("=" * 60)

        if failures:
            raise RuntimeError(
                f"Enrichment failed for {len(failures)}/{len(workbooks)} workbooks: "
                + ", ".join(n for n, _ in failures)
            )

        return {"enriched": [str(p) for p in enriched_paths]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run minimal LLM enrichment over parsed workbook markdowns."
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
