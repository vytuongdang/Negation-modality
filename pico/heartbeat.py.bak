#!/usr/bin/env python3
"""Pico's heartbeat: a small loop, independent of the runner, that keeps Vy informed on Telegram.

    tmux new -d -s pico-heartbeat "python3 pico/heartbeat.py"     # from the project root

Every CHECK_MIN minutes it looks at the machine; every REPORT_MIN minutes it posts a short status to the
"Pico Vy Research" topic: is the runner alive, what step it is on and the ETA (pico/status.txt, written by the
runner), GPU jobs, free disk. It posts an alert at once (not waiting for the hour) when:
  - free disk on / drops below DISK_WARN_GB (again below DISK_URGENT_GB),
  - the runner's tmux session "pico" is gone (and pico/stop was not asked for),
  - the runner wrote nothing to pico/run.log for STALE_MIN minutes,
  - Claude reports a usage limit in the runner log.
Each alert is sent once per episode and once more when it clears.
"""
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import notify

ROOT = Path(__file__).resolve().parent.parent
PICO = ROOT / "pico"
CHECK_MIN, REPORT_MIN = 10, 60
DISK_WARN_GB, DISK_URGENT_GB = 30, 10
STALE_MIN = 120


def send(text):
    import sys
    sys.argv = ["notify.py", text]
    notify.main()


def tmux_sessions():
    try:
        out = subprocess.run(["tmux", "ls", "-F", "#{session_name}"], capture_output=True, text=True, timeout=10).stdout
        return out.split()
    except Exception:
        return []


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        u, m, t = [x.strip() for x in out.split(",")]
        return f"GPU {u}% · {int(m) / 1024:.1f}/{int(t) / 1024:.0f} GB"
    except Exception:
        return "GPU ?"


def tail(path, n=30):
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def check():
    free_gb = shutil.disk_usage("/").free / 1e9
    sessions = tmux_sessions()
    runner_alive = "pico" in sessions
    log = PICO / "run.log"
    idle_min = (time.time() - log.stat().st_mtime) / 60 if log.exists() else None
    limit = any(re.search(r"usage limit|rate limit|limit reached|resets at", l, re.I) for l in tail(log, 15))
    return {"free_gb": free_gb, "sessions": sessions, "runner_alive": runner_alive,
            "stop_asked": (PICO / "stop").exists(), "idle_min": idle_min, "limit": limit}


def alerts(s):
    out = {}
    if s["free_gb"] < DISK_URGENT_GB:
        out["disk"] = f"🚨 Ổ đĩa sắp đầy: chỉ còn {s['free_gb']:.1f} GB trống! Job có thể hỏng nếu hết chỗ."
    elif s["free_gb"] < DISK_WARN_GB:
        out["disk"] = f"⚠️ Ổ đĩa còn {s['free_gb']:.1f} GB trống (dưới {DISK_WARN_GB} GB)."
    if not s["runner_alive"] and not s["stop_asked"]:
        out["runner"] = ("⚠️ Pico (tmux 'pico') không còn chạy. Bật lại: cd ~/new/cross_multimodal_negation && "
                         "tmux new -d -s pico \"bash pico/runner.sh\"")
    if s["runner_alive"] and s["idle_min"] is not None and s["idle_min"] > STALE_MIN:
        out["stale"] = f"⚠️ Pico không ghi log {s['idle_min']:.0f} phút rồi, có thể đang bị treo."
    if s["limit"]:
        out["limit"] = "⏸️ Claude báo hết quota; Pico tạm nghỉ và tự chạy lại khi quota hồi."
    return out


def report(s):
    status = "\n".join(tail(PICO / "status.txt", 6)) or "(Pico chưa ghi status.txt)"
    jobs = [x for x in s["sessions"] if x.startswith("job-")]
    alive = "✅ Pico đang chạy" if s["runner_alive"] else ("⏹️ Pico đã dừng (theo yêu cầu)" if s["stop_asked"]
                                                            else "❌ Pico không chạy")
    return (f"🫀 {datetime.now():%H:%M} · {alive}\n{status}\n"
            f"Jobs: {', '.join(jobs) or 'không có'} · {gpu()} · Ổ đĩa trống {s['free_gb']:.0f} GB")


def main():
    active, last_report = {}, 0.0
    while True:
        try:
            s = check()
            now_alerts = alerts(s)
            for k, msg in now_alerts.items():
                if active.get(k) != msg.split(":")[0]:
                    send(msg)
                    active[k] = msg.split(":")[0]
            for k in [k for k in active if k not in now_alerts]:
                send({"disk": "✅ Ổ đĩa ổn trở lại.", "runner": "✅ Pico chạy lại rồi.",
                      "stale": "✅ Pico ghi log trở lại.", "limit": "✅ Quota Claude đã hồi, Pico chạy tiếp."}[k])
                del active[k]
            if time.time() - last_report >= REPORT_MIN * 60:
                send(report(s))
                last_report = time.time()
        except Exception as e:  # noqa: BLE001  the heartbeat must never die
            print(f"{datetime.now():%H:%M} heartbeat error: {e}", flush=True)
        time.sleep(CHECK_MIN * 60)


if __name__ == "__main__":
    main()
