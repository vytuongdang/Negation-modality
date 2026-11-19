#!/usr/bin/env python3
"""Post a step report to Vy's Telegram topic "Pico Vy Research" (through Vy's Pico bot).

    python3 pico/notify.py "text"                      # a message (long text is split)
    python3 pico/notify.py - < report.txt              # text from stdin
    python3 pico/notify.py "caption" --photo fig.png   # a picture with a caption
    python3 pico/notify.py "caption" --doc REPORT.md   # a file
    python3 pico/notify.py --last                      # the last reports sent (read before writing a new one)

The bot token is read from ~/new/AI_assitant/pico/keys/telegram.txt and the topic's thread id from the bot's
bot_state.json (the topic must exist: Vy sends /setup in the group once). Every report is also appended to
pico/telegram_sent.log, which the chat side of the bot reads to answer Vy's questions.
"""
import json
import sys
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

BOT_DIR = Path.home() / "new" / "AI_assitant" / "pico"
TOPIC = "Pico Vy Research"
LOG = Path(__file__).resolve().parent / "telegram_sent.log"


def target():
    token = (BOT_DIR / "keys" / "telegram.txt").read_text().splitlines()[0].strip()
    state = json.loads((BOT_DIR / "bot_state.json").read_text())
    for chat, names in state.get("topic_names", {}).items():
        for thread, name in names.items():
            if name.lower() == TOPIC.lower():
                return token, chat, thread
    raise SystemExit(f'topic "{TOPIC}" not found: Vy must send /setup in the Telegram group first')


def api(token, method, fields, files=None):
    if files:
        boundary = uuid.uuid4().hex
        body = b""
        for k, v in fields.items():
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, path in files.items():
            body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{Path(path).name}"\r\n'
                     f"Content-Type: application/octet-stream\r\n\r\n").encode() + Path(path).read_bytes() + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/{method}", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    else:
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/{method}",
                                     data=urllib.parse.urlencode(fields).encode())
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    if not out.get("ok"):
        raise RuntimeError(out)


def chunks(text, n=3900):
    while text:
        cut = text.rfind("\n", 0, n) if len(text) > n else len(text)
        cut = cut if cut > 0 else n
        yield text[:cut]
        text = text[cut:].lstrip("\n")


def log(text, status):
    with open(LOG, "a") as f:
        f.write(f"===== {datetime.now():%Y-%m-%d %H:%M} ({status})\n{text}\n\n")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "--last":
        print("\n".join(LOG.read_text().splitlines()[-80:]) if LOG.exists() else "(nothing sent yet)")
        return 0
    text = sys.stdin.read() if args[0] == "-" else args[0]
    photo = args[args.index("--photo") + 1] if "--photo" in args else None
    doc = args[args.index("--doc") + 1] if "--doc" in args else None
    try:
        token, chat, thread = target()
        base = {"chat_id": chat, "message_thread_id": thread}
        if photo or doc:
            api(token, "sendPhoto" if photo else "sendDocument", {**base, "caption": text[:1000]},
                {"photo" if photo else "document": photo or doc})
        else:
            for part in chunks(text.strip()):
                api(token, "sendMessage", {**base, "text": part})
    except (SystemExit, Exception) as e:  # noqa: BLE001  never crash the runner over a notification
        log(text, f"NOT SENT: {e}")
        print(f"not sent: {e}")
        return 1
    log(text + (f"\n[attached: {photo or doc}]" if photo or doc else ""), "sent")
    print("sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
