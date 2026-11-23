#!/bin/bash
# One-shot restart of the Pico runner (scheduled 2026-11-20 10:00 KST via systemd-run).
export PATH=/home/vyvn/.local/bin:/usr/local/bin:/usr/bin:/bin
cd /home/vyvn/new/cross_multimodal_negation || exit 1
tmux has-session -t '=pico' 2>/dev/null && exit 0
rm -f pico/stop
echo "[2026-11-20 10:00, scheduled restart] Restarted after Vy's overnight stop. The runner stopped at 01:03 while waiting on the usage limit, so NOTES.md was NOT updated and job-h1trial was NOT stopped; it kept running overnight. First: check tmux job-h1trial and the tail of pico/logs/h1_trial.log, update NOTES.md with the exact state, post one short Telegram report, then continue the agreed H1 trial plan. Keep working token-efficiently (see the 00:48 rules in NOTES)." > pico/note.txt
tmux new -d -s pico "bash pico/runner.sh"
