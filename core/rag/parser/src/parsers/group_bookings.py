"""Parser for the 'Group Bookings' sheet."""

import pandas as pd
from .base import BaseParser


class GroupBookingsParser(BaseParser):
    SHEET_NAME = "Group Bookings"
    OUTPUT_FILE = "group_bookings.md"

    def to_markdown(self) -> str:
        lines = ["# Group Bookings", ""]

        # Split rows into blocks separated by empty (NaN field) rows.
        blocks: list[list[pd.Series]] = []
        current: list[pd.Series] = []
        for _, row in self.df.iterrows():
            field = self.val(row.get("field", ""))
            if not field:
                if current:
                    blocks.append(current)
                    current = []
                continue
            current.append(row)
        if current:
            blocks.append(current)

        for i, block in enumerate(blocks):
            first_field = self.val(block[0].get("field", ""))

            if first_field.startswith("---") and first_field.endswith("---"):
                title = first_field.strip("- ").strip().title()
                content_rows = block[1:]
            elif i == 0:
                title = "Group Bookings"
                content_rows = block
            else:
                title = first_field
                content_rows = block

            lines += [f"## {title}", ""]
            for row in content_rows:
                field = self.val(row.get("field", ""))
                value = self.val(row.get("value", ""))
                notes = self.val(row.get("notes", ""))
                if not field or not value:
                    continue
                note_suffix = f" *(Note: {notes})*" if notes else ""
                lines.append(f"- **{field}:** {value}{note_suffix}")
            lines.append("")

        return "\n".join(lines)
