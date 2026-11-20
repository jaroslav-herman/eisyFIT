---
name: session-handoff
description: Use when the user wants to continue this project's work on another machine, asks to "save the session" or share it via GitHub, or wraps up for the day. Writes a handoff note into the repo and pushes it; on a new machine, reads the latest handoff notes to resume context. Never commit raw Vibe session logs or anything from ~/.vibe to git.
---

# Session handoff across machines

## Facts to keep in mind

- Vibe sessions are **local only**: they live under `~/.vibe/logs/session/` on the
  machine that ran them (override with `session_logging.save_dir` in
  `~/.vibe/config.toml`, or `VIBE_HOME`). There is no built-in cloud sync, and a
  session id can only be resumed where its files exist (`vibe --resume <id>` in
  the CLI; in Vibe Desktop the session list shows what is under the local save
  dir). Resume pickers without an id are scoped to the current folder.
- Therefore the **repository is the cross-machine channel**: code, `.eisfit.json`
  projects, project skills, and handoff notes travel via git; transcripts do not.

## Never do this

- Do **not** commit, push, or copy into the repo anything from `~/.vibe/` — session
  logs under `~/.vibe/logs/session/`, `~/.vibe/.env` (API keys), `config.toml`,
  `trusted_folders.toml`. Raw transcripts can contain secrets, full file paths of
  one machine, and data that is not meant for the repo's visibility.
- Do not stage unrelated untracked artifacts when committing a handoff.

## When the user is switching machines / asks to save the session

1. Write a handoff note at `.agents/handoffs/YYYY-MM-DD-HHMM-<slug>.md`
   (create the directory if missing). Content, concise:
   - Objective of the work and current status.
   - What was done, with `path/to/file.py:42` references and commit hashes.
   - Decisions made and why (especially non-obvious ones).
   - Open questions / next steps, in priority order.
   - How to verify: test command (`./.venv/Scripts/python.exe -m unittest discover -s tests`),
     focused test module, or launch command (`python main.py <spectrum>.mpt --cycle 1`).
   - Any uncommitted changes and what they are.
   - Optionally the Vibe session id (the bare uuid string only — never the log file).
2. Sanitize first: no API keys, tokens, or credentials; nothing that must not be
   visible to everyone who can see this repo. If the repo is public and the note
   would be sensitive, ask the user where to put it instead (e.g. a private notes
   repo) before writing.
3. Commit the note (and any finished tracked work it describes) and push to
   `main`, per this repository's convention. Do not stage unrelated files.
4. Tell the user the note path and that the next machine should `git pull` first.

## When a session starts on a (new) machine

1. `git pull` the repo.
2. Read the newest files in `.agents/handoffs/` first (they summarize where the
   previous session stopped), plus the root `AGENTS.md`.
3. Skim `git log --oneline -20` for recent commits.
4. Continue from the handoff's "next steps". Do not try to resume the old session
   id unless its files were explicitly transferred (see below).

## If the user explicitly wants the raw transcript on another machine

Only for the CLI, never by pushing logs to GitHub:

- Point `session_logging.save_dir` in `~/.vibe/config.toml` at a folder synced by
  another tool (OneDrive, Dropbox, Syncthing — not git), or copy the session
  directory `~/.vibe/logs/session/unified/<id>/` to the same relative location on
  the other machine, then `vibe --resume <id>` (not folder-scoped).
- Caveats to state: absolute paths recorded from the old machine will not resolve;
  concurrent access from two machines can corrupt the session; in Vibe Desktop
  there are no CLI flags or slash commands, so the config-file equivalent applies.

## Cloud vs local for this project

- The Tk GUI (`eis_gui.py`, `main.py`) needs a local display — run it on a machine
  you sit at, not in a headless cloud box.
- Headless work — unit tests, fitting/DRT/ML scripts, CI, `vibe -p` runs — can run
  in the cloud; its outputs belong in git (code, `.eisfit.json` projects, ML
  artifacts that are meant to be tracked), so pull them on the local machine before
  opening the GUI.
