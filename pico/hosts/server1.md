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
