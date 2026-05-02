"""Gmail-API follow-up email service.

Sends a self-notification from the AeroSports Scarborough events mailbox
(`events.scb@aerosportsparks.ca`) whenever a call ends flagged as
`needs_human`.

First-run setup
---------------
The user token is created via OAuth on first auth and cached at
`src/oauths/scarborough_token.json`. On a headless server, run

    python -m src.email_service bootstrap

once on a machine with a browser to mint the token, then copy the token
file to the server alongside `scarborough_credentials.json`.
"""

from __future__ import annotations

import base64
import logging
from email.mime.text import MIMEText
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

_OAUTH_DIR = Path(__file__).parent / "oauths"
CREDENTIALS_PATH = _OAUTH_DIR / "scarborough_credentials.json"
TOKEN_PATH = _OAUTH_DIR / "scarborough_gmail_token.json"

SENDER_EMAIL = "events.scb@aerosportsparks.ca"
RECIPIENT_EMAIL = "events.scb@aerosportsparks.ca"


def _get_service():
    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            f"Gmail OAuth client secrets not found at {CREDENTIALS_PATH}"
        )

    creds: Credentials | None = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), GMAIL_SCOPES
            )
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _build_raw_message(to: str, sender: str, subject: str, body: str) -> dict:
    msg = MIMEText(body, "plain", "utf-8")
    msg["To"] = to
    msg["From"] = sender
    msg["Subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return {"raw": raw}


def send_followup_email(
    call_sid: str,
    phone_number: str,
    summary: str,
    flag_reason: str | None,
) -> bool:
    """Send a follow-up notification to `RECIPIENT_EMAIL` for a flagged call.

    Returns True on success, False on any failure (errors are logged, not raised,
    so a Gmail outage never breaks call teardown).
    """
    subject = f"[Follow-up] {phone_number or 'unknown caller'} — {flag_reason or 'needs human'}"
    body_lines = [
        "A call was just flagged for follow-up.",
        "",
        f"Caller phone: {phone_number or 'unknown'}",
        f"Twilio Call SID: {call_sid}",
        f"Reason: {flag_reason or '(none)'}",
        "",
        "Summary:",
        summary or "(no summary captured)",
    ]
    body = "\n".join(body_lines)

    try:
        service = _get_service()
        body_payload = _build_raw_message(RECIPIENT_EMAIL, SENDER_EMAIL, subject, body)
        service.users().messages().send(userId="me", body=body_payload).execute()
        logger.info("[%s] Follow-up email sent to %s", call_sid, RECIPIENT_EMAIL)
        return True
    except HttpError as exc:
        logger.error("[%s] Gmail API error sending follow-up: %s", call_sid, exc)
    except Exception as exc:
        logger.error("[%s] Failed to send follow-up email: %s", call_sid, exc)
    return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "bootstrap":
        _get_service()
        print(f"Token cached at {TOKEN_PATH}")
    else:
        print("Usage: python -m src.email_service bootstrap")
