# Local deploy (Windows)

Tooling for running OpenViking from this fork on a local Windows machine. The
server runs from a **uv tool install**, not from the repo working tree, so these
scripts exist to keep the two from drifting apart.

Upstream container/Kubernetes deployment is separate — see [helm/openviking](helm/openviking)
and the [Dockerfile](../Dockerfile).

| File | Purpose |
|---|---|
| `deploy.ps1` | Install the package at the current commit, render config, restart the server |
| `check-drift.ps1` | Verify the installed package still matches the deployed commit |
| `ov.conf.template` | Tracked config template; the live `ov.conf` is rendered from it |

## Deploying

```bash
pwsh deploy/deploy.ps1
```

What it does, in order: verify the working tree is clean, resolve the two API
keys, render `~/.openviking/ov.conf` from the template, **stop the server**,
`uv tool install --reinstall .`, start the server, poll `/health`, and stamp the
deployed commit to `~/.openviking/.deployed-commit`.

Useful switches:

```bash
pwsh deploy/deploy.ps1 -DryRun        # show every step, change nothing
pwsh deploy/deploy.ps1 -SkipInstall   # re-render config + restart only
pwsh deploy/deploy.ps1 -AllowDirty    # deploy an uncommitted tree (see below)
```

### The dirty-tree guard

`deploy.ps1` refuses to run when the working tree is dirty, because the commit
stamp is what `check-drift.ps1` compares against — deploying uncommitted code
records a commit that does not describe what is installed. Commit first. Use
`-AllowDirty` only when you accept that the stamp will be wrong.

### API keys

Two different vendors, so two separate keys, and neither is stored in the repo:

| Key | Env var | 1Password switch |
|---|---|---|
| VLM / extraction | `OPENVIKING_VLM_API_KEY` | `-OpRef` |
| Embedding | `OPENVIKING_EMBED_API_KEY` | `-EmbedOpRef` |

Each resolves in the same order: environment variable, then the key already in
the live `ov.conf`, then 1Password when the matching `-OpRef` / `-EmbedOpRef` is
supplied. A normal redeploy needs no key material at all — the existing values
are carried across.

## Verifying

```bash
pwsh deploy/check-drift.ps1
```

Compares every tracked `.py` file in the installed package against the tree of
the commit in the stamp file. Exit codes: `0` in sync, `1` drift detected, `2`
cannot verify. Add `-Detailed` to list the differing files.

Drift means someone copied a file into the install by hand, or a deploy failed
partway. The fix is a clean `deploy.ps1` run, not another hand-copy.

## Config is rendered, not edited

`~/.openviking/ov.conf` is **overwritten from `ov.conf.template` on every
deploy**. Edit the template and commit it; do not hand-edit the live file
expecting the change to survive.

This is also why a deploy re-enables anything the template enables. If you have
temporarily disabled ingest (`ingest.enabled = false`) to quiesce the store, a
deploy restores it — usually what you want, but worth knowing before you deploy
mid-maintenance.

## Restarting drops the extraction queue

Deploying restarts the server, and queued extraction work does not survive a
restart. Before deploying during heavy session activity, drain first:

```bash
curl -s -m 300 -X POST http://127.0.0.1:1933/api/v1/system/wait -H "Content-Type: application/json" -d '{"timeout": 280}'
```

If the queue will not drain because live sessions keep feeding it, set
`ingest.enabled = false` in `ov.conf`, wait for writes to stop, then deploy —
the template render turns ingest back on.

## Dependency pinning

`uv tool install --reinstall .` re-resolves dependencies, so an unpinned
dependency can pull a new major version and break startup on a deploy that
changed no application code. This has already happened once: `mcp` was requested
as `>=1.27.0`, `mcp 2.0.0` removed `mcp.server.fastmcp`, and the server died at
import with `ModuleNotFoundError`. It is now capped at `<2` in `pyproject.toml`.

If the server fails to become healthy after a deploy, read
`~/.openviking/logs/server.log.err` first — a startup `ImportError` or
`ModuleNotFoundError` there almost always means a dependency moved, not that
your change was wrong.

## Server supervision

The server runs under the `OpenViking Server` scheduled task, which invokes
`~/.openviking/run-server.ps1`. That wrapper restarts the server on crash **and
on wedge** — a process that is alive but not answering `/health` for three
consecutive 60s probes is killed and restarted (upstream issue #527). Logs land
in `~/.openviking/logs/server.log`, with startup stderr in `server.log.err`.
