## Agent skills

### Issue tracker

Issues are tracked in GitHub at `anthoai97/hibiki-inference`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default canonical triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context layout with root `CONTEXT.md` and `docs/adr/`. See `docs/agents/domain.md`.

### Repository tooling

Run local Git commands through `zsh`. Use the system-installed `gh` CLI for GitHub issues, pull requests, and other GitHub operations.
Run Python, tests, and package tooling in the Conda environment named `hibiki`; for non-interactive commands, use `conda run -n hibiki <command>`.
