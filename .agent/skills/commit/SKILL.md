---
name: commit
description: Use this skill when the user asks you to commit changes in the current repository. By default, inspect the working tree, group changes by topic, and create one or more git commits that include all current changes unless the user gave specific commit instructions.
---

# Commit

Use this skill when the user wants the current repository changes committed.

## Default behavior

Unless the user gave more specific instructions, commit all current changes in the repository.

If the working tree contains unrelated themes, split them into separate commits by topic instead of forcing everything into one commit.

## Workflow

1. Inspect the current state with non-destructive git commands such as `git status --short`, `git diff --stat`, and targeted diffs for touched files.
2. Group files into coherent commit topics based on behavior or purpose.
3. If there is only one coherent topic, create one commit that includes all current changes.
4. If there are multiple topics, stage and commit each topic separately with clear commit messages.
5. Use concise commit messages that describe the resulting change, not the mechanical edit.
6. Do not rewrite, drop, or revert user changes unless the user explicitly asked for that.

## Guardrails

- Respect explicit user instructions over this default workflow.
- If the repository contains partially complete or conflicting changes that cannot be safely grouped, explain the blocker instead of making a misleading commit.
- Prefer non-interactive git commands.
- After committing, report the commit hashes and the topic split you used.
