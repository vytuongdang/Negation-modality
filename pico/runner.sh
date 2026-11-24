#!/bin/bash
# Pico runner: keeps Claude Code working on pico/task.md until pico/stop exists, economically.
#   start:   tmux new -d -s pico "bash pico/runner.sh"        (from the project root)
#   watch:   tail -f pico/run.log
#   message: echo "try idea X next" > pico/note.txt            (delivered once, at the next step)
#   stop:    touch pico/stop                                    (after the current step)
# Quota savers:
#   - pause between steps: pico/sleep_seconds if Pico wrote one (30..43200, used once); otherwise 3 h while a
#     GPU job (tmux session job-*) runs, 30 s when Pico is working interactively;
#   - Claude usage limit: back off 15 -> 30 -> 60 min instead of retrying every 30 s (the heartbeat tells Vy);
#   - a fresh session every STEPS_PER_SESSION steps (10; change it in pico/steps_per_session; Pico can ask
#     for a fresh session earlier by creating pico/new_session when its context gets long) (long sessions make every step expensive); Pico's memory is
#     pico/NOTES.md, so nothing is lost.
cd "$(dirname "$0")/.." || exit 1
LOG=pico/run.log
# This machine's settings (not in git): PICO_HOST (server1 | b200), PICO_TOPIC, PICO_NOTES. See pico/hosts/.
[ -f pico/host.env ] && . pico/host.env
# Claude login for unattended runs (B200): ~/.config/pico/claude.env (chmod 600) with
#   export CLAUDE_CODE_OAUTH_TOKEN=...   (from `claude setup-token`)   and   export PATH="$HOME/.local/bin:$PATH"
[ -f ~/.config/pico/claude.env ] && . ~/.config/pico/claude.env
NOTES=${PICO_NOTES:-pico/NOTES.md}
HOSTDOC=""; [ -n "$PICO_HOST" ] && [ -f "pico/hosts/$PICO_HOST.md" ] && HOSTDOC=" and pico/hosts/$PICO_HOST.md (this machine's role and git rules; on this machine your memory file is $NOTES)"
FLAGS="--dangerously-skip-permissions --model claude-opus-5-5 --effort medium"   # Vy: Opus 5.5 medium unless Vy asks otherwise
STEPS_PER_SESSION=10   # default; pico/steps_per_session overrides it (read every step, no restart needed)
LIMIT_RE='usage limit|limit reached|rate limit|resets at|out of extra usage'

run_claude() {   # run one claude -p call, log it, return 2 on a usage limit
  local out; out=$(mktemp)
  claude -p "$@" $FLAGS > "$out" 2>&1
  local rc=$?
  cat "$out" >> $LOG
  if grep -qiE "$LIMIT_RE" "$out" && [ $(wc -c < "$out") -lt 2000 ]; then rm -f "$out"; return 2; fi
  rm -f "$out"; return $rc
}

new_session() {
  SID=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid); echo "$SID" > pico/session_id; STEP=0
  echo "=== $(date) new session $SID" >> $LOG
  run_claude "$(cat pico/task.md)

(You are starting a fresh session: sessions are rotated every $STEPS_PER_SESSION steps to save quota.
Read pico/RESEARCH.md (the study)$HOSTDOC and $NOTES (your memory) first, then continue from where it says.)" --session-id "$SID"
}

backoff=0
if grep -qE '^[0-9a-f-]{36}$' pico/session_id 2>/dev/null; then SID=$(cat pico/session_id); STEP=0; echo "=== $(date) resume $SID" >> $LOG
else new_session; fi
while [ ! -e pico/stop ]; do
  [ -s pico/steps_per_session ] && STEPS_PER_SESSION=$(cat pico/steps_per_session)
  if [ "$STEP" -ge "$STEPS_PER_SESSION" ] || [ -e pico/new_session ]; then rm -f pico/new_session; new_session; continue; fi
  msg="Continue the task from where you are. Read $NOTES and pico/task.md if unsure."
  note=""
  if [ -e pico/note.txt ]; then note=$(cat pico/note.txt); rm pico/note.txt; msg="$msg Note from Vy: $note"; fi
  echo "=== $(date) step $STEP" >> $LOG
  run_claude "$msg" --resume "$SID"; rc=$?
  if [ $rc -eq 2 ]; then
    [ -n "$note" ] && echo "$note" >> pico/note.txt      # keep Vy's note for the next try
    backoff=$(( backoff == 0 ? 900 : (backoff * 2 > 3600 ? 3600 : backoff * 2) ))
    echo "=== $(date) usage limit: waiting $backoff s" >> $LOG
    sleep $backoff; continue
  fi
  backoff=0; STEP=$((STEP + 1))
  if [ -s pico/sleep_seconds ]; then
    pause=$(cat pico/sleep_seconds); rm pico/sleep_seconds
    [[ "$pause" =~ ^[0-9]+$ ]] || pause=30
  elif tmux ls -F '#{session_name}' 2>/dev/null | grep -q '^job-'; then
    pause=10800   # 3 h while a GPU job runs (Pico should set an exact pause from the job ETA)
  else
    pause=30
  fi
  [ "$pause" -lt 30 ] && pause=30; [ "$pause" -gt 43200 ] && pause=43200
  [ "$pause" -gt 60 ] && echo "=== $(date) pause $pause s" >> $LOG
  for ((i = 0; i < pause; i += 10)); do   # wake early for a note from Vy or a stop request
    [ -e pico/note.txt ] || [ -e pico/stop ] && break
    sleep 10
  done
done
echo "=== $(date) stopped (pico/stop found)" >> $LOG
