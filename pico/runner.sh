#!/bin/bash
# Pico runner: keeps one Claude Code conversation working on pico/task.md until pico/stop exists.
#   start:   tmux new -d -s pico "bash pico/runner.sh"        (from the project root)
#   watch:   tail -f pico/run.log
#   message: echo "try idea X next" > pico/note.txt            (delivered once, at the next step)
#   stop:    touch pico/stop                                    (after the current step)
# Pico can ask for a longer pause between steps (e.g. while a GPU job runs) by writing a number of seconds
# (30..3600) into pico/sleep_seconds; it is used once.
cd "$(dirname "$0")/.." || exit 1
LOG=pico/run.log
FLAGS="--dangerously-skip-permissions"
if [ -s pico/session_id ]; then
  SID=$(cat pico/session_id)
  echo "=== $(date) resume session $SID" >> $LOG
  claude -p "You were restarted. Read pico/NOTES.md and pico/task.md, then continue the task." \
    $FLAGS --resume "$SID" >> $LOG 2>&1 || {
      echo "=== resume failed, new session" >> $LOG; rm -f pico/session_id; }
fi
if [ ! -s pico/session_id ]; then
  SID=$(uuidgen); echo "$SID" > pico/session_id
  echo "=== $(date) new session $SID" >> $LOG
  claude -p "$(cat pico/task.md)" $FLAGS --session-id "$SID" >> $LOG 2>&1
fi
while [ ! -e pico/stop ]; do
  msg="Continue the task from where you are. Read pico/NOTES.md and pico/task.md if unsure."
  if [ -e pico/note.txt ]; then
    msg="$msg Note from Vy: $(cat pico/note.txt)"; rm pico/note.txt
  fi
  echo "=== $(date) step" >> $LOG
  claude -p "$msg" $FLAGS --resume "$SID" >> $LOG 2>&1
  pause=30
  if [ -s pico/sleep_seconds ]; then
    pause=$(cat pico/sleep_seconds); rm pico/sleep_seconds
    [[ "$pause" =~ ^[0-9]+$ ]] || pause=30
    [ "$pause" -lt 30 ] && pause=30; [ "$pause" -gt 3600 ] && pause=3600
  fi
  sleep "$pause"
done
echo "=== $(date) stopped (pico/stop found)" >> $LOG
