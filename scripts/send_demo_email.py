"""Send a demo follow-up email to verify Gmail wiring.

Run after `scripts/setup_email.py` has minted a token. This sends a
realistic-looking follow-up notification using the same code path the
voice router uses, so a passing run here means production sends will work.

    python scripts/send_demo_email.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.email_service import RECIPIENT_EMAIL, send_followup_email  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


DEMO_SUMMARY = (
    "Caller asked about availability for a private birthday party for 22 kids "
    "on Saturday May 17. Caller name: Priya Singh. Wanted a callback to "
    "confirm pricing for the VIP Birthday Package and to ask whether the "
    "Birds Eye Party Arena is bookable that day."
)


def main() -> int:
    print(f"Sending demo follow-up email to {RECIPIENT_EMAIL}…")
    ok = send_followup_email(
        call_sid="DEMO-CALL-20260502-0001",
        phone_number="+14165551234",
        summary=DEMO_SUMMARY,
        flag_reason="callback requested — birthday party booking",
    )
    if not ok:
        print("FAILED — see log above.")
        return 1
    print(f"OK — check the {RECIPIENT_EMAIL} inbox.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
