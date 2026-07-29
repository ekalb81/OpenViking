# Memory Plugin Shared Library

This directory contains shared JavaScript modules that are vendored into the
Claude Code, Codex, OpenCode, and pi memory plugins by `sync.mjs`.

## Workspace Peers

`lib/workspace-peer.mjs` derives the default actor peer from the **project root**
containing the current working directory, not from the raw path.

`findProjectRoot` walks up from the working directory to the nearest ancestor
holding a project marker (`.git`, `package.json`, `pyproject.toml`, `Cargo.toml`,
`go.mod`, `pom.xml`, `build.gradle`, `.hg`, `.svn`) and uses that directory. Two
consequences worth knowing:

- A linked git worktree reads `gitdir:` out of its `.git` file and resolves back
  to the main repository, so worktrees share the project's peer instead of each
  getting one of their own.
- Filesystem and drive roots, the bare home directory, and the OS temp root are
  never projects. When no marker is found above the working directory, **no
  workspace peer is derived** and resolution falls through to "no peer".

That last rule is the point: deriving a peer from any directory meant a session
started in `C:\` produced a peer literally named `C--`, and a harness that runs
each session from its own scratch directory minted a brand new peer every run.

`deriveWorkspacePeerId` itself is unchanged and still matches Claude's
project-directory naming: every character outside `A-Z`, `a-z`, and `0-9` becomes
`-`, with no folding or trimming. For example, a project root of
`/Users/x/Dev/OpenViking` becomes `-Users-x-Dev-OpenViking`. Existing projects
therefore keep the peer id they already had; a subdirectory of a project now maps
to the project's peer rather than to one of its own.

Resolution order is:

1. Explicit peer: `OPENVIKING_PEER_ID`, `actor_peer_id` / `peer_id` in
   `ovcli.conf`, or the harness-specific legacy peer config.
2. Project-root-derived peer when `workspacePeer` is not `false` **and** the
   working directory is inside a project.
3. No peer.

Set `OPENVIKING_WORKSPACE_PEER=0` or the harness config `workspacePeer=false`
to disable workspace-derived peers entirely.

## Recall Peer Scope

`lib/recall-core.mjs` defaults to the broad recall mode and does not send a
`peer_scope` field. In that mode, the server can recall global memory, the
current workspace, and other workspace memories; other workspaces are penalized
and rendered later.

When `recallPeerScope` is `actor`, the helper sends `peer_scope:"actor"`. This
is the isolation mode: recall only sees global memory plus the current
workspace. If an older server rejects that field with 400 or 422, `postRecall`
removes `peer_scope` and retries once.

For deployments where one bot serves multiple real people, such as zouk,
vikingbot, or AstrBot, configure an explicit actor peer and use the isolation
mode so one person's memories are not recalled into another person's session.
