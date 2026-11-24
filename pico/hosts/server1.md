# Machine: server 1 (test / dev)

Telegram topic: "Pico Vy Research". Memory file: pico/NOTES.md. One GPU, shared with Vy's other work.

Role: write and test code, run small pilots (<= 300 samples), analyses, the paper draft, TTS (CosyVoice
lives only here). Big runs go to B200 (topic "Negation_project_B200"), which has its own runner.

Git (repo shared with B200 through GitHub `origin`, branch main):
- Before you start editing code: `git pull --rebase origin main`.
- Commit small, focused commits; then `git push origin main` so B200 can pull them. If the push is
  rejected, `git pull --rebase` and push again; never force-push.
- Do not edit pico/NOTES_b200.md (B200's memory). Read it (after a pull) to know what B200 is doing.
- Never edit a file that a running job will load later (wrapper scripts instead), on either machine.
- Result files: do not commit outputs of a run tag that B200 also writes (e.g. full runs); pilots here use
  their own tags (pilot*, smoke*).

Dataset on each machine (create_data/dataset/ is NOT in git; each machine has its own copy):
- server 1 = the SUBSET (1,150 units / 1,142 images; pilot200 = 932 questions). All results so far use it.
- B200 = the FULL dataset v2 (built on B200: `subset_units.py --restore`, then the pipeline; TTS also on B200).
  A rebuild re-runs B2/B3, so the same sample_id can mean a different item on the two machines: never mix
  results across machines by sample_id. B200 run tags start with `full` (e.g. scored_full.jsonl).
- Never commit a change to shared config/code that only one machine's dataset needs: use command-line
  options or pico/host.env instead.
