"""
Stub mailer module.

Real email sending is deferred. For now, flag alerts are written to a log file
at logs/flag_alerts.log so you can verify end-to-end that flagged calls are
being detected and routed correctly. Replace send_flag_alert with a real
implementation (Gmail SMTP, Resend, etc.) when ready.
"""

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_FLAG_LOG_PATH = os.path.join(_LOG_DIR, "flag_alerts.log")


def send_flag_alert(
    call_id: int,
    phone_number: str,
    summary: str,
    flag_reason: str,
    transcript: str,
) -> None:
    """
    Stub: writes flag alerts to logs/flag_alerts.log instead of sending email.

    Signature matches the real sender so swapping in a real implementation
    later is a single-file change with zero call-site edits.
    """
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        entry = (
            f"\n{'=' * 70}\n"
            f"FLAG ALERT — {timestamp}\n"
            f"Call ID:     {call_id}\n"
            f"From:        {phone_number}\n"
            f"Summary:     {summary}\n"
            f"Flag reason: {flag_reason}\n"
            f"{'-' * 70}\n"
            f"Transcript:\n{transcript}\n"
            f"{'=' * 70}\n"
        )
        with open(_FLAG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.info("Flag alert written to %s for call %s", _FLAG_LOG_PATH, call_id)
    except Exception as exc:
        # Must never crash the call — same contract as a real email sender.
        logger.error("send_flag_alert (stub) failed: %s", exc)