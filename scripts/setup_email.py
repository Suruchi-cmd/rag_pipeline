"""One-time Gmail OAuth setup + test email.

Run this once on a machine with a browser, signed in as
`events.scb@aerosportsparks.ca`:

    python scripts/setup_email.py

It will:
  1. Open a browser for Google OAuth consent (gmail.send scope).
  2. Cache the refresh token at src/oauths/scarborough_token.json.
  3. Send a test follow-up email to events.scb@aerosportsparks.ca so you
     can confirm the wiring end-to-end.

After this runs successfully once, the token is reused on every send and
no further interactive auth is needed (refresh tokens auto-renew). On a
headless server, run this locally and copy the generated token file to
the server alongside scarborough_credentials.json.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.email_service import (  # noqa: E402
    CREDENTIALS_PATH,
    TOKEN_PATH,
    _get_service,
    send_followup_email,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def main() -> int:
    print(f"Credentials: {CREDENTIALS_PATH}")
    print(f"Token cache: {TOKEN_PATH}")

    if not CREDENTIALS_PATH.exists():
        print(f"\nERROR: OAuth client secrets not found at {CREDENTIALS_PATH}")
        return 1

    print("\n[1/2] Authorising Gmail API (browser will open if no token cached)…")
    _get_service()
    print(f"      Token cached → {TOKEN_PATH}")

    print("\n[2/2] Sending test follow-up email…")
    ok = send_followup_email(
        call_sid="TEST-CALL-SID-0001",
        phone_number="+15555550123",
        summary="This is a test follow-up email sent by scripts/setup_email.py to verify the Gmail wiring.",
        flag_reason="setup self-test",
    )
    if not ok:
        print("      Test send FAILED — check the log above.")
        return 2

    print("      Test send OK — check the events.scb inbox.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
