# OpenViking — C4 Architecture

## Level 1 — System Context

```mermaid
flowchart TB
    dev["👤 Developer / End User<br/><i>ov CLI, Web Studio</i>"]
    agent["🤖 AI Coding Agent<br/><i>Claude Code, Codex, Cursor, Trae,<br/>OpenCode, OpenClaw, Pi</i>"]
    app["👤 Application Developer<br/><i>Python / TypeScript / Go SDK</i>"]
    chat["👤 Chat User<br/><i>Feishu, Slack, Telegram, Discord</i>"]

    ov["<b>OpenViking</b><br/>Agent-native Context Database<br/><i>Tiered context (L0/L1/L2) over a<br/>viking:// filesystem tree</i>"]

    llm["LLM / VLM Providers<br/><i>Ark/Doubao, OpenAI, Kimi, GLM,<br/>Gemini, LiteLLM</i>"]
    emb["Embedding & Rerank APIs<br/><i>Volcengine, OpenAI, Cohere, Jina,<br/>Voyage, DashScope, Gemini</i>"]
    vdb["Managed Vector DBs<br/><i>VikingDB, Qdrant, OpenGauss</i>"]
    src["Content Sources<br/><i>Local FS, HTTP, Git, Feishu/Lark,<br/>Web crawler, RSS/Atom</i>"]
    blob["Blob / Cache Backends<br/><i>S3, Redis, SQL, KV</i>"]
    otel["Observability<br/><i>OpenTelemetry, Prometheus</i>"]

    dev --> ov
    agent -- "MCP tool calls" --> ov
    app --> ov
    chat --> ov

    ov -- "abstracts, memory extraction" --> llm
    ov -- "embed / rerank" --> emb
    ov -- "vector read/write" --> vdb
    ov -- "ingest content" --> src
    ov -- "store / cache" --> blob
    ov -- "traces & metrics" --> otel
```

## Level 2 — Containers

```mermaid
flowchart TB
    agent["🤖 AI Agent"]
    user["👤 User"]

    subgraph OV["OpenViking System"]
        cli["<b>ov CLI</b><br/>Rust · crates/ov_cli<br/><i>terminal client + TUI</i>"]
        studio["<b>Web Studio</b><br/>React + Vite · web-studio/<br/><i>browser UI, trajectory viewer</i>"]
        sdk["<b>SDKs</b><br/>sdk/python · sdk/typescript · sdk/go"]
        plug["<b>Agent Plugins</b><br/>examples/*-plugin<br/><i>hooks for Claude Code, Cursor…</i>"]

        server["<b>OpenViking Server</b><br/>Python · FastAPI/Uvicorn · :1933<br/><i>REST routers + MCP endpoint</i><br/>openviking/server, service"]

        workers["<b>Background Workers</b><br/>Python asyncio (in-process)<br/><i>WatchScheduler, semantic/embedding/<br/>commit queues, TaskTracker</i>"]
        ingest["<b>Log-Ingestion Daemon</b><br/>Python · openviking/ingest<br/><i>polls agent session logs</i>"]
        bot["<b>VikingBot</b><br/>Python · bot/vikingbot<br/><i>multi-channel chat agent</i>"]
        caddy["<b>Caddy Proxy</b><br/>optional · TLS/ACME"]

        ragfs[("<b>RAGFS</b><br/>Rust + PyO3 · crates/ragfs<br/><i>agent filesystem engine</i><br/>localfs · s3fs · sqlfs · kvfs")]
        vec[("<b>Vector Index Engine</b><br/>C++ abi3 ext · src/index, src/store<br/><i>local ANN + scalar index</i>")]
    end

    ext_llm["LLM / Embedding / Rerank APIs"]
    ext_vdb["Qdrant · VikingDB · OpenGauss"]
    ext_src["Git · Feishu · Web · Files"]

    user --> cli
    user --> studio
    agent -- "MCP stdio/HTTP" --> server
    agent --> plug
    plug --> server
    sdk --> server
    cli --> server
    studio -- "REST/JSON" --> server
    caddy --> server

    server <--> workers
    ingest --> server
    bot --> server

    server -- "viking:// read/write" --> ragfs
    server -- "vector recall" --> vec
    workers --> ragfs
    workers --> vec

    server --> ext_llm
    workers --> ext_llm
    vec -.-> ext_vdb
    server -.-> ext_vdb
    workers -- "fetch" --> ext_src
```

## Level 3 — Components (OpenViking Server)

```mermaid
flowchart TB
    subgraph transport["Transport — openviking/server"]
        mcp["<b>mcp_endpoint.py</b><br/><i>find, search, recall, read, list,<br/>remember, add_resource, grep, glob,<br/>code_outline, code_search, forget</i>"]
        rest["<b>routers/</b><br/><i>search, filesystem, resources, sessions,<br/>skills, watches, tasks, observer, admin</i>"]
        auth["<b>auth/ · oauth/ · api_keys/</b><br/><i>OAuth 2.1, API keys, OTP → RequestContext</i>"]
    end

    subgraph svc["Business Logic — openviking/service"]
        core["<b>core.py</b> · OpenVikingService"]
        svcs["fs_service · search_service · session_service<br/>resource_service · relation_service · pack_service<br/>task_store · task_tracker"]
    end

    subgraph domain["Domain"]
        retrieve["<b>retrieve/</b><br/>intent_analyzer → hierarchical_retriever<br/>type_quota_recall · memory_lifecycle"]
        session["<b>session/</b><br/>compressor_v2/v3 · memory/extract_loop<br/>memory_updater · merge_op · skill/"]
        parse["<b>parse/</b><br/>accessors/ (local, http, git, feishu, crawler)<br/>parsers/ (pdf, office, code/tree-sitter, media)<br/>parser_router · tree_builder · vlm"]
        models["<b>models/</b><br/>vlm · embedder · rerank backends"]
        resource["<b>resource/</b><br/>watch_manager · watch_scheduler · watch_storage"]
    end

    subgraph storage["Persistence — openviking/storage"]
        vfs["<b>viking_fs.py</b> · VikingFS URI layer<br/><i>read/write/mkdir/abstract/overview/relations</i>"]
        queuefs["<b>queuefs/</b><br/>semantic_queue · embedding_queue<br/>add_resource_processor · session_commit_processor"]
        vdba["<b>vectordb/ · vectordb_adapters/</b><br/>local · qdrant · vikingdb · opengauss · http"]
        txn["<b>transaction/</b> locks + redo_log"]
        obs["<b>observers/</b> filesystem · vikingdb · queue<br/>retrieval · lock — <i>trajectory capture</i>"]
        pack["<b>ovpack/</b> export/backup format"]
    end

    ragfs[("RAGFS (Rust)")]
    engine[("Vector Engine (C++)")]

    mcp --> core
    rest --> core
    auth --> mcp
    auth --> rest
    core --> svcs
    svcs --> retrieve & session & parse & resource
    retrieve --> models
    session --> models
    parse --> models
    svcs --> vfs
    retrieve --> vdba
    parse --> queuefs
    session --> queuefs
    resource --> parse
    queuefs --> vfs
    queuefs --> vdba
    vfs --> txn
    vfs --> ragfs
    vdba --> engine
    obs -.-> vfs
    obs -.-> queuefs
    obs -.-> vdba
    pack --> vfs
```

## Key Flows

```mermaid
flowchart LR
    subgraph read["Read path — MCP query"]
        q["query"] --> ia["intent_analyzer<br/>0–5 typed queries"] --> hr["hierarchical_retriever<br/>recursive directory descent"]
        hr --> vr["vector recall<br/>(URIs + metadata only)"] --> rr["rerank"] --> fetch["VikingFS fetch<br/>L0 → L1 → L2 on demand"]
    end
```

```mermaid
flowchart LR
    subgraph write["Write path — ingestion"]
        i["file / URL / git / Feishu"] --> a["accessor"] --> p["parser_router → parsers"] --> tb["tree_builder"] --> fs["RAGFS"]
        fs --> sq["semantic_queue<br/>bottom-up L0/L1 via LLM/VLM"] --> eq["embedding_queue"] --> vi["vector index"]
    end
```

```mermaid
flowchart LR
    subgraph mem["Session → long-term memory"]
        m["agent messages<br/>+ tool results"] --> c["compressor_v2/v3<br/>keep recent N, archive rest"] --> e["extract_loop<br/>8-category extraction"] --> mu["memory_updater<br/>LLM dedup / merge"] --> w["viking://user/{id}/memories<br/>+ vector index"]
    end
```

```mermaid
flowchart LR
    subgraph watch["Watch / sync"]
        t["WatchScheduler<br/>60s tick, concurrency 4"] --> d["get_due_tasks"] --> o["refresh Feishu OAuth"] --> ar["ResourceService.add_resource"] --> pipe["→ ingestion pipeline"]
    end
```

## Runtime & Infrastructure

### Deployment topology

```mermaid
flowchart TB
    subgraph host["Docker host / K8s node"]
        subgraph c1["container: openviking — ghcr.io/volcengine/openviking"]
            entry["<b>openviking-entrypoint</b> (sh, PID 1)<br/><i>ensure ov.conf → start server → poll /health</i>"]
            srv["<b>openviking-server</b> (uvicorn, :1933)<br/><i>+ RAGFS, C++ vector engine, async workers<br/>all in-process</i>"]
            bbot["<b>vikingbot</b> <i>(optional, --with-bot)</i>"]
            entry --> srv
            entry -.-> bbot
        end
        subgraph c2["container: caddy:2 — legacy ingress"]
            cad["reverse proxy :1934<br/><i>optional :80/:443 ACME TLS</i>"]
        end
        vol[("volume<br/>~/.openviking → /app/.openviking<br/><i>ov.conf, ovcli.conf, workspace</i>")]
    end

    cad --> srv
    srv --- vol
    srv --> ext["external APIs<br/><i>LLM · embedding · rerank · S3 · Qdrant</i>"]
```

**The whole system is one application container.** RAGFS, the C++ vector index engine, and every background worker (WatchScheduler, semantic/embedding/commit queues, TaskTracker) run in-process inside `openviking-server` — they are logical containers in the C4 sense, not separate images. Compose declares no database, cache, or vector-DB service; external stores are reached over the network only when configured.

### Build pipeline

Three-stage [Dockerfile](../Dockerfile), because one wheel bundles four toolchains:

| Stage | Base | Produces |
|---|---|---|
| 1 · `rust-toolchain` | `rust:1.91.1-trixie` | Rust toolchain (ragfs-python's S3 feature set needs ≥ 1.91.1) |
| 2 · `py-builder` | `ghcr.io/astral-sh/uv:python3.13-trixie-slim` | `/app/.venv` — cargo + Node 24 + cmake copied in, `uv sync --no-editable --extra bot --extra gemini` |
| 3 · runtime | `python:3.13-slim-trixie` | Final image: venv + entrypoint only |

`setuptools` is the build backend; `setup.py` drives all three native builds — `build_py` compiles the web-studio SPA (npm/Vite), `build_ext` compiles the C++ vector engine (cmake), and `build_ov_cli_artifact` cargo-builds the `ov` CLI. BuildKit cache mounts cover uv, npm, cargo registry/git/target, and ccache (gcc/g++ routed through `/usr/lib/ccache`).

Runtime OS packages are deliberately thin: `ca-certificates`, `curl`, `git`, `libstdc++6`, `ripgrep` (ripgrep backs the `grep`/`glob` MCP tools; `git` backs the git accessor).

`requires-python = ">=3.10"`, but the shipped image runs 3.13.

### Startup sequence

[docker/openviking-entrypoint.sh](../docker/openviking-entrypoint.sh) is PID 1 and does three things beyond `exec`:

1. **Config gate** — if `/app/.openviking/ov.conf` is missing, it writes one from `OPENVIKING_CONF_CONTENT` if set; otherwise it starts `pending_health_server.py` on :1933 that answers every request with a 503 explaining the fix, then polls every 5s until the file appears. So a misconfigured container is diagnosable over HTTP rather than crash-looping.
2. **Launch** — `openviking-server --host 0.0.0.0 [--with-bot]`. `OPENVIKING_WITH_BOT` defaults to **1** in the image, so Docker runs VikingBot in the same container by default; the Helm chart sets `bot.enabled: false`.
3. **Health barrier & signals** — polls `/health` up to 120×1s, exits non-zero if the server dies or times out, and forwards `INT`/`TERM` to the server PID.

Any other argv (`openviking --help`, `openviking-server init`) is `exec`'d directly, so the image doubles as the CLI.

Health is `GET /health` at every layer — Docker `HEALTHCHECK`, compose healthcheck, and K8s liveness/readiness probes.

### Configuration & secrets

Single volume at `/app/.openviking` mirroring the host `~/.openviking` holds `ov.conf`, `ovcli.conf`, and workspace data. `OPENVIKING_CONFIG_FILE` / `OPENVIKING_CLI_CONFIG_FILE` point at them; `HOME=/app`.

In Helm, `config:` is rendered into a ConfigMap mounted as `ov.conf` — covering `storage.workspace`, `vectordb.backend`, embedding/VLM provider blocks (Volcengine Ark defaults), and `server.*`. Secrets (`root_api_key`, provider API keys) are meant to come through `extraEnv` + `secretKeyRef` rather than the ConfigMap; `root_api_key` is required whenever `host` is `0.0.0.0`.

### Kubernetes

[deploy/helm/openviking](../deploy/helm/openviking): `replicaCount: 1`, `Service` ClusterIP:1933, optional Ingress (cert-manager annotations sketched in comments), requests 500m/1Gi and limits 2 CPU/4Gi, PVC `ReadWriteOnce` 20Gi at `/app/.openviking`.

**Scaling caveat:** `replicaCount: 1` + an RWO PVC + `server.workers: 1` + in-process queues and lock manager mean this is a single-instance deployment. Horizontal scaling isn't something the chart supports as written — the workers and `storage/transaction/` locks assume one process owns the workspace.

## Persistence Summary

| Data | Store |
|---|---|
| L0 abstract / L1 overview / L2 content | RAGFS — `.abstract.md`, `.overview.md`, content files |
| Relations graph | RAGFS — `.relations.json` per directory |
| Vectors + URIs + scalar metadata | Local C++ engine (LevelDB) or Qdrant / VikingDB / OpenGauss / HTTP |
| Memories | `viking://user/{user_id}/memories` |
| Skills | `viking://user/{user_id}/skills` (`SKILL.md`) |
| Resources | `viking://resources/...` |
| Sessions, messages, tool results | RAGFS + vector index |
| Watch tasks (incl. OAuth state) | RAGFS JSON via `watch_storage.py` |
| API keys / OAuth clients / OTP | `server/api_keys/`, `server/oauth/storage.py` |
| Locks / redo log | `storage/transaction/` |
| Export / backup | `.ovpack` archives |
| Caches | Redis / Mooncake / YuanRong; s3fs & sqlfs local caches |
| Config | `ov.conf` (`~/.openviking`) |
