---
name: commit
description: Use this skill when the user asks you to commit changes in the current repository. By default, inspect the working tree, group changes by topic, and create one or more git commits that include all current changes unless the user gave specific commit instructions.
---

# Commit

Use this skill when the user wants the current repository changes committed.

## Default behavior

Unless the user gave more specific instructions, commit all current changes in the repository on the **current branch** — do not create a new branch, switch branches, or otherwise move the work elsewhere. The user can always ask to commit on a different branch if they want to.

If the working tree contains unrelated themes, split them into separate commits by topic instead of forcing everything into one commit.

## Workflow

1. Inspect the current state with non-destructive git commands such as `git status --short`, `git diff --stat`, and targeted diffs for touched files.
2. Group files into coherent commit topics based on behavior or purpose.
3. If there is only one coherent topic, create one commit that includes all current changes.
4. If there are multiple topics, stage and commit each topic separately with clear commit messages.
5. Write commit messages with both a concise title and a descriptive comment/body. The title should summarize the resulting change, while the comment/body must briefly describe the specific code changes, affected components, and any relevant behavior or rationale. Do not create title-only commit messages.
6. Do not rewrite, drop, or revert user changes unless the user explicitly asked for that.

## Python files: ruff is mandatory

Every commit containing `.py` files must pass ruff. A `pre-commit` hook (`.githooks/pre-commit`, activated via `core.hooksPath`) checks the **staged** content with `ruff check` and `ruff format --check` and blocks the commit on failure. Before staging `.py` files:

1. Run `ruff check --fix <files>` and `ruff format <files>` on exactly the files you intend to commit — never on the whole repo or on `baselinedir/` (vendored baseline code is excluded from ruff and must never be formatted).
2. Review the resulting diff: `ruff check --fix` can change annotations (`UP`) and drop imports (`F401`), so confirm nothing load-bearing was removed.
3. Stage the fixed files. Never auto-format in the hook direction (the hook is check-only by design, so unstaged edits in `MM` files stay untouched).

If the hook fails, fix the reported issues and re-stage; do not pass `--no-verify` unless the user explicitly asks. Config lives in `ruff.toml` at the repo root.

## Guardrails

- Respect explicit user instructions over this default workflow.
- If the repository contains partially complete or conflicting changes that cannot be safely grouped, explain the blocker instead of making a misleading commit.
- Prefer non-interactive git commands.
- After committing, report the commit hashes and the topic split you used.
