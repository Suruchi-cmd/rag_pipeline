"""
Email handoff alerts for needs_human=True calls.

Reads SMTP creds from env. If unset, the alert is logged only (still useful
for ops tailing logs). Wrapped in try/except so a mailer failure never breaks
a turn or leaks an error to the caller.

Required env vars to actually send mail:
  SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO
"""

from __future__ import annotations

import asyncio
import logging
import os
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)

_SMTP_HOST = os.getenv("SMTP_HOST", "")
_SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
_SMTP_USER = os.getenv("SMTP_USER", "")
_SMTP_PASS = os.getenv("SMTP_PASS", "")
_SMTP_FROM = os.getenv("SMTP_FROM", "")
_SMTP_TO = os.getenv("SMTP_TO", "")  # comma-separated allowed


def _smtp_configured() -> bool:
    return all([_SMTP_HOST, _SMTP_USER, _SMTP_PASS, _SMTP_FROM, _SMTP_TO])


def _send_sync(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = _SMTP_FROM
    msg["To"] = _SMTP_TO
    msg.set_content(body)
    with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=10) as s:
        s.starttls()
        s.login(_SMTP_USER, _SMTP_PASS)
        s.send_message(msg)


async def send_human_handoff_alert(
    call_sid: str,
    phone_number: str,
    summary: str,
    flag_reason: str,
) -> None:
    """Best-effort handoff email. Always returns; never raises."""
    subject = f"[AeroBot] Human follow-up needed — call {call_sid}"
    body = (
        f"Call SID:    {call_sid}\n"
        f"From:        {phone_number}\n"
        f"Summary:     {summary}\n"
        f"Flag reason: {flag_reason or '(none)'}\n"
    )

    if not _smtp_configured():
        logger.warning(
            "[%s] SMTP not configured; handoff alert NOT sent. "
            "Set SMTP_HOST/SMTP_USER/SMTP_PASS/SMTP_FROM/SMTP_TO env vars.\n%s",
            call_sid, body,
        )
        return

    try:
        await asyncio.to_thread(_send_sync, subject, body)
        logger.info("[%s] Handoff alert email sent to %s", call_sid, _SMTP_TO)
    except Exception as exc:
        logger.error("[%s] Handoff alert email FAILED: %s", call_sid, exc)
