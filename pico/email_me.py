#!/usr/bin/env python3
"""Pico's email helper: send one email to the owner, keep a copy so the next email can continue from it.

    python3 pico/email_me.py "Subject" < body.txt      # send (subject gets the [Pico] prefix)
    python3 pico/email_me.py --last                    # print the last emails sent (read before writing a new one)
    python3 pico/email_me.py --test                    # send a short test email

Settings: ~/.config/pico/gmail.env (GMAIL_USER, GMAIL_APP_PASSWORD, PICO_TO), chmod 600, never committed.
If the settings are missing, the email is only written to pico/emails.log and the script says so.
"""
import smtplib
import sys
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

ENV = Path.home() / ".config/pico/gmail.env"
LOG = Path(__file__).resolve().parent / "emails.log"


def settings():
    cfg = {}
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def log(subject, body, status):
    with open(LOG, "a") as f:
        f.write(f"===== {datetime.now():%Y-%m-%d %H:%M} ({status})\nSubject: [Pico] {subject}\n\n{body}\n\n")


def send(subject, body):
    cfg = settings()
    user, pw, to = cfg.get("GMAIL_USER"), cfg.get("GMAIL_APP_PASSWORD", "").replace(" ", ""), cfg.get("PICO_TO")
    if not (user and pw and to):
        log(subject, body, "NOT SENT: fill in ~/.config/pico/gmail.env")
        print(f"not sent (mail settings missing in {ENV}); saved to {LOG}")
        return 1
    msg = EmailMessage()
    msg["From"] = f"Pico <{user}>"
    msg["To"] = to
    msg["Subject"] = f"[Pico] {subject}"
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as s:
        s.login(user, pw)
        s.send_message(msg)
    log(subject, body, "sent")
    print("sent")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--last":
        text = LOG.read_text() if LOG.exists() else "(no emails yet)"
        print("\n".join(text.splitlines()[-60:]))
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        return send("Test", "Hello from Pico on the lab machine. If you can read this, email works.\n\nPico")
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    return send(sys.argv[1], sys.stdin.read())


if __name__ == "__main__":
    sys.exit(main())
