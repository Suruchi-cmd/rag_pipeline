"""Parser for the 'Aero Camp' sheet.

The sheet has three logical sections separated by embedded header rows:
  1. Camp Programs (schedule & pricing)
  2. Add-Ons (per day extras)
  3. Camp Details (key-value facts)
"""

import pandas as pd
from .base import BaseParser


class AeroCampParser(BaseParser):
    SHEET_NAME = "Aero Camp"
    OUTPUT_FILE = "aero_camp.md"

    def to_markdown(self) -> str:
        df = self.df.reset_index(drop=True)
        lines = ["# Aero Camp", ""]

        program_col = df["program"].astype(str).str.strip()
        addon_idx = df.index[program_col == "ADD-ONS (PER DAY)"]
        detail_idx = df.index[program_col == "CAMP DETAILS"]

        end_programs = addon_idx[0] if len(addon_idx) else len(df)
        end_addons = detail_idx[0] if len(detail_idx) else len(df)

        # ---- Section 1: Camp Programs (rows before ADD-ONS divider) ----
        programs_df = df.loc[: end_programs - 1]
        programs_df = programs_df[programs_df["program"].notna()]

        if not programs_df.empty:
            lines += ["## Camp Programs", ""]
            lines.append("| Program | Schedule | Price | Notes |")
            lines.append("|---------|----------|-------|-------|")
            for _, row in programs_df.iterrows():
                program = self.val(row.get("program", ""))
                schedule = self.val(row.get("schedule", ""))
                price = self.val(row.get("price", ""))
                notes = self.val(row.get("notes", ""))
                if program:
                    lines.append(f"| {program} | {schedule} | {price} | {notes} |")
            lines.append("")

        # ---- Section 2: Add-Ons ----
        if len(addon_idx):
            # skip the divider row (+1) and the embedded column-header row (+2)
            addon_rows = df.loc[addon_idx[0] + 2 : end_addons - 1]
            addon_rows = addon_rows[addon_rows["program"].notna()]

            if not addon_rows.empty:
                lines += ["## Add-Ons (Per Day)", ""]
                for _, row in addon_rows.iterrows():
                    add_on = self.val(row.get("program", ""))
                    price = self.val(row.get("schedule", ""))
                    if add_on and price:
                        lines.append(f"- **{add_on}:** {price}")
                lines.append("")

        # ---- Section 3: Camp Details ----
        if len(detail_idx):
            detail_rows = df.loc[detail_idx[0] + 2 :]  # skip divider + header
            detail_rows = detail_rows[detail_rows["program"].notna()]

            if not detail_rows.empty:
                lines += ["## Camp Details", ""]
                for _, row in detail_rows.iterrows():
                    field = self.val(row.get("program", ""))
                    value = self.val(row.get("schedule", ""))
                    if field and value:
                        lines.append(f"- **{field}:** {value}")
                lines.append("")

        return "\n".join(lines)
