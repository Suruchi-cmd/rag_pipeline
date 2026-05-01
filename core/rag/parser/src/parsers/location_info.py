"""Parser for the 'Location Info' sheet."""

import pandas as pd
from .base import BaseParser


_LINK_KEYWORDS = ("url", "link", "maps", "waiver", "invitation")


def _section_for(field: str) -> str:
    f = field.lower()
    if f.startswith("hours"):
        return "Hours of Operation"
    if any(k in f for k in _LINK_KEYWORDS):
        return "Important Links"
    return "Contact Information"


class LocationInfoParser(BaseParser):
    SHEET_NAME = "Location Info"
    OUTPUT_FILE = "location_info.md"

    def to_markdown(self) -> str:
        lines = ["# AeroSports Scarborough — Location Info", ""]

        sections: dict[str, list[tuple[str, str]]] = {}
        for _, row in self.df.iterrows():
            field = self.val(row.get("field", ""))
            value = self.val(row.get("value", ""))
            if not field or not value:
                continue
            sections.setdefault(_section_for(field), []).append((field, value))

        for section in ("Contact Information", "Hours of Operation", "Important Links"):
            entries = sections.get(section)
            if not entries:
                continue
            lines += [f"## {section}", ""]
            for field, value in entries:
                lines.append(f"- **{field}:** {value}")
            lines.append("")

        return "\n".join(lines)
