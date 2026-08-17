# Issue tracker: GitHub

Issues and specs for this project live in the GitHub repository `anthoai97/hibiki-inference`:

https://github.com/anthoai97/hibiki-inference

Use the `gh` CLI for all operations. Pass `--repo anthoai97/hibiki-inference` when the repository cannot be inferred from the current Git checkout.

## Conventions

- **Create an issue**: `gh issue create --repo anthoai97/hibiki-inference --title "..." --body "..."`
- **Read an issue**: `gh issue view <number> --repo anthoai97/hibiki-inference --comments`, fetching labels and filtering comments with `--json`, `--jq`, or `jq` when needed.
- **List issues**: `gh issue list --repo anthoai97/hibiki-inference --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`, with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --repo anthoai97/hibiki-inference --body "..."`
- **Apply or remove labels**: `gh issue edit <number> --repo anthoai97/hibiki-inference --add-label "..."` or `--remove-label "..."`
- **Close an issue**: `gh issue close <number> --repo anthoai97/hibiki-inference --comment "..."`

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --repo anthoai97/hibiki-inference --comments` and `gh pr diff <number> --repo anthoai97/hibiki-inference`.
- **List external PRs for triage**: list open PRs with their labels, author, association, and comments; retain `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, and `NONE`.
- **Comment, label, or close**: use `gh pr comment`, `gh pr edit`, and `gh pr close` with `--repo anthoai97/hibiki-inference`.

GitHub shares one number space across issues and PRs. Resolve a bare `#42` with `gh pr view 42 --repo anthoai97/hibiki-inference`, falling back to `gh issue view 42 --repo anthoai97/hibiki-inference`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue in `anthoai97/hibiki-inference`.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --repo anthoai97/hibiki-inference --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: an issue labelled `wayfinder:map`, holding the Notes, Decisions-so-far, and Fog body.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue. Where sub-issues are unavailable, add it to a task list in the map and put `Part of #<map>` at the top of the child body. Apply a `wayfinder:<type>` label: `research`, `prototype`, `grilling`, or `task`.
- **Blocking**: use GitHub's native issue dependencies. Where dependencies are unavailable, use a `Blocked by: #<n>, #<n>` line at the top of the child body.
- **Frontier query**: list the map's open children, then discard tickets with open blockers or an assignee. The first remaining ticket in map order wins.
- **Claim**: `gh issue edit <n> --repo anthoai97/hibiki-inference --add-assignee @me`.
- **Resolve**: comment with the answer, close the ticket, and append a context pointer to the map's Decisions-so-far.
