import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import http from "node:http";
import { tmpdir } from "node:os";
import { delimiter, dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { resolveCodexLaunch, trySpawnCodex } from "./codex-launch.mjs";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));

test("Windows Codex launch bypasses the npm POSIX shim", () => {
  const npmBin = String.raw`C:\Users\test\AppData\Roaming\npm`;
  const npmEntryPoint = String.raw`C:\Users\test\AppData\Roaming\npm\node_modules\@openai\codex\bin\codex.js`;
  const launch = resolveCodexLaunch({
    platform: "win32",
    pathValue: `${npmBin};C:\\Windows\\System32`,
    execPath: String.raw`C:\Program Files\nodejs\node.exe`,
    pathExists: (candidate) => candidate === npmEntryPoint,
  });

  assert.deepEqual(launch, {
    command: String.raw`C:\Program Files\nodejs\node.exe`,
    argsPrefix: [npmEntryPoint],
  });
});

test("Codex launch converts a synchronous spawn failure into a fallback signal", () => {
  const failure = Object.assign(new Error("spawn EPERM"), { code: "EPERM" });
  const result = trySpawnCodex(["exec"], { stdio: "pipe" }, {
    resolveLaunch: () => ({ command: "codex", argsPrefix: [] }),
    spawnImpl: () => { throw failure; },
  });

  assert.equal(result.child, null);
  assert.equal(result.error, failure);
});

function readRequestBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf-8");
      try {
        resolve(raw ? JSON.parse(raw) : null);
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

function writeJson(res, value) {
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify(value));
}

function writeStatusJson(res, status, value) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(value));
}

async function withMockOpenViking(handler, fn) {
  const server = http.createServer((req, res) => {
    handler(req, res).catch((err) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "error", error: String(err?.stack || err) }));
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const { port } = server.address();
    return await fn(`http://127.0.0.1:${port}`);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

function runAutoRecall(input, env) {
  return new Promise((resolve, reject) => {
    const cleanEnv = { ...process.env };
    for (const key of Object.keys(cleanEnv)) {
      if (key.startsWith("OPENVIKING_")) delete cleanEnv[key];
    }
    const child = spawn(process.execPath, [join(SCRIPT_DIR, "auto-recall.mjs")], {
      env: { ...cleanEnv, ...env },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`auto-recall exited ${code}: ${stderr}`));
        return;
      }
      resolve({ stdout, stderr });
    });
    child.stdin.end(JSON.stringify(input));
  });
}

async function withFakeCodex(output, fn, { exitCode = 0, stderrOutput = "" } = {}) {
  const binDir = await mkdtemp(join(tmpdir(), "ov-fake-codex-"));
  const executable = join(binDir, "codex");
  const npmEntryPoint = join(binDir, "node_modules", "@openai", "codex", "bin", "codex.js");
  const callLog = join(binDir, "calls.log");
  await writeFile(executable, `#!/bin/sh
output_path=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output-last-message" ]; then
    shift
    output_path="$1"
  fi
  shift
done
cat >/dev/null
printf 'called\\n' >> "$FAKE_CODEX_CALL_LOG"
printf '%s' "$FAKE_CODEX_STDERR" >&2
if [ "$FAKE_CODEX_EXIT_CODE" -ne 0 ]; then
  exit "$FAKE_CODEX_EXIT_CODE"
fi
printf '%s' "$FAKE_CODEX_OUTPUT" > "$output_path"
`);
  await chmod(executable, 0o755);
  await mkdir(dirname(npmEntryPoint), { recursive: true });
  await writeFile(npmEntryPoint, `
const fs = require("node:fs");
let outputPath = "";
for (let index = 0; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--output-last-message") outputPath = process.argv[index + 1] || "";
}
process.stdin.resume();
process.stdin.on("end", () => {
  fs.appendFileSync(process.env.FAKE_CODEX_CALL_LOG, "called\\n");
  process.stderr.write(process.env.FAKE_CODEX_STDERR || "");
  const code = Number(process.env.FAKE_CODEX_EXIT_CODE || 0);
  if (code !== 0) process.exit(code);
  fs.writeFileSync(outputPath, process.env.FAKE_CODEX_OUTPUT || "");
});
`);
  try {
    return await fn({
      callLog,
      env: {
        PATH: `${binDir}${delimiter}${process.env.PATH}`,
        FAKE_CODEX_CALL_LOG: callLog,
        FAKE_CODEX_EXIT_CODE: String(exitCode),
        FAKE_CODEX_OUTPUT: output,
        FAKE_CODEX_STDERR: stderrOutput,
      },
    });
  } finally {
    await rm(binDir, { recursive: true, force: true });
  }
}

async function runEndpointCompressionCase({
  prompt,
  entry,
  rendered,
  compressorOutput,
  exitCode = 0,
  stderrOutput = "",
  extraEnv = {},
  serverStats = { returned: 1 },
}) {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-auto-recall-endpoint-compress-"));
  const debugLogPath = join(stateDir, "debug.log");
  let requestBody = null;
  try {
    return await withFakeCodex(compressorOutput, async ({ callLog, env }) => {
      const result = await withMockOpenViking(async (req, res) => {
        const url = new URL(req.url, "http://127.0.0.1");
        if (req.method === "GET" && url.pathname === "/health") {
          writeJson(res, { status: "ok", result: { ok: true } });
          return;
        }
        if (req.method === "POST" && url.pathname === "/api/v1/search/recall") {
          requestBody = await readRequestBody(req);
          writeJson(res, {
            status: "ok",
            result: { entries: [entry], rendered, stats: serverStats },
          });
          return;
        }
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ status: "error", error: "not found" }));
      }, async (baseUrl) => runAutoRecall(
        { prompt, session_id: "codex:endpoint-compress" },
        {
          ...env,
          OPENVIKING_AUTO_RECALL: "1",
          OPENVIKING_CODEX_STATE_DIR: stateDir,
          OPENVIKING_CONFIG_FILE: join(stateDir, "missing-ov.conf"),
          OPENVIKING_CLI_CONFIG_FILE: join(stateDir, "missing-ovcli.conf"),
          OPENVIKING_CREDENTIAL_SOURCE: "env",
          OPENVIKING_DEBUG: "1",
          OPENVIKING_DEBUG_LOG: debugLogPath,
          OPENVIKING_RECALL_COMPRESS: "1",
          OPENVIKING_RECALL_TIMEOUT_MS: "10000",
          OPENVIKING_MIN_QUERY_LENGTH: "1",
          OPENVIKING_SCORE_THRESHOLD: "0",
          OPENVIKING_TIMEOUT_MS: "5000",
          OPENVIKING_URL: baseUrl,
          ...extraEnv,
        },
      ));
      const compressorCallLog = await readFile(callLog, "utf-8").catch(() => "");
      const debugLog = await readFile(debugLogPath, "utf-8").catch(() => "");
      return {
        output: JSON.parse(result.stdout.trim()),
        compressorCalls: compressorCallLog.trim().split("\n").filter(Boolean).length,
        requestBody,
        debugEntries: debugLog.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line)),
      };
    }, { exitCode, stderrOutput });
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
}

async function runFallbackClassificationCase({
  endpointDelayMs = 0,
  fallbackDelayMs = 0,
  fallbackStatus = 200,
  sessionId = "codex:fallback-transport",
} = {}) {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-auto-recall-fallback-transport-"));
  const debugLogPath = join(stateDir, "debug.log");
  const prompt = "PRIVATE_FALLBACK_TRANSPORT_PROMPT";
  try {
    return await withMockOpenViking(async (req, res) => {
      const url = new URL(req.url, "http://127.0.0.1");
      if (req.method === "GET" && url.pathname === "/health") {
        writeJson(res, { status: "ok", result: { ok: true } });
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/v1/search/recall") {
        await readRequestBody(req);
        if (endpointDelayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, endpointDelayMs));
          if (res.destroyed) return;
        }
        writeStatusJson(res, 500, { status: "error", error: "PRIVATE_ENDPOINT_ERROR_TEXT" });
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/v1/search/search") {
        await readRequestBody(req);
        if (fallbackDelayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, fallbackDelayMs));
          if (res.destroyed) return;
        }
        if (fallbackStatus >= 500) {
          writeStatusJson(res, fallbackStatus, {
            status: "error",
            error: "PRIVATE_FALLBACK_ERROR_TEXT",
          });
        } else {
          writeJson(res, { status: "ok", result: { memories: [], skills: [] } });
        }
        return;
      }
      writeStatusJson(res, 404, { status: "error", error: "not found" });
    }, async (baseUrl) => {
      const result = await runAutoRecall(
        { prompt, session_id: sessionId },
        {
          OPENVIKING_AUTO_RECALL: "1",
          OPENVIKING_CODEX_STATE_DIR: stateDir,
          OPENVIKING_CONFIG_FILE: join(stateDir, "missing-ov.conf"),
          OPENVIKING_CLI_CONFIG_FILE: join(stateDir, "missing-ovcli.conf"),
          OPENVIKING_CREDENTIAL_SOURCE: "env",
          OPENVIKING_DEBUG: "1",
          OPENVIKING_DEBUG_LOG: debugLogPath,
          OPENVIKING_RECALL_COMPRESS: "0",
          OPENVIKING_RECALL_LIMIT: "1",
          OPENVIKING_RECALL_TIMEOUT_MS: "10000",
          OPENVIKING_MIN_QUERY_LENGTH: "1",
          OPENVIKING_SCORE_THRESHOLD: "0",
          OPENVIKING_TIMEOUT_MS: "1000",
          OPENVIKING_URL: baseUrl,
        },
      );
      const debugLog = await readFile(debugLogPath, "utf-8");
      return {
        output: JSON.parse(result.stdout.trim()),
        prompt,
        debugEntries: debugLog.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line)),
      };
    });
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
}

test("auto-recall uses context-aware search with the derived OpenViking session id", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-auto-recall-state-"));
  const debugLogPath = join(stateDir, "debug.log");
  const requests = [];
  const rawPrompt = "RAW_FALLBACK_QUERY_SENTINEL";
  const memoryUri = "viking://user/private/memories/events/fallback-secret.md";
  const memoryAbstract = "RAW_FALLBACK_ABSTRACT_SENTINEL";
  const memoryContent = "RAW_FALLBACK_CONTENT_SENTINEL";
  let hookOutput = null;

  try {
    await withMockOpenViking(async (req, res) => {
      const url = new URL(req.url, "http://127.0.0.1");
      if (req.method === "GET" && url.pathname === "/health") {
        writeJson(res, { status: "ok", result: { ok: true } });
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/v1/search/search") {
        const body = await readRequestBody(req);
        requests.push({ path: url.pathname, body });
        if (body.target_uri === "viking://user/memories") {
          writeJson(res, {
            status: "ok",
            result: {
              memories: [{
                uri: memoryUri,
                level: 2,
                score: 0.9,
                category: "events",
                abstract: memoryAbstract,
              }],
              skills: [],
            },
          });
          return;
        }
        writeJson(res, { status: "ok", result: { memories: [], skills: [] } });
        return;
      }
      if (req.method === "GET" && url.pathname === "/api/v1/content/read") {
        writeJson(res, { status: "ok", result: memoryContent });
        return;
      }
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "error", error: "not found" }));
    }, async (baseUrl) => {
      const result = await runAutoRecall(
        { prompt: rawPrompt, session_id: "codex:123" },
        {
          OPENVIKING_AUTO_RECALL: "1",
          OPENVIKING_CODEX_STATE_DIR: stateDir,
          OPENVIKING_CONFIG_FILE: join(stateDir, "missing-ov.conf"),
          OPENVIKING_CLI_CONFIG_FILE: join(stateDir, "missing-ovcli.conf"),
          OPENVIKING_CREDENTIAL_SOURCE: "env",
          OPENVIKING_DEBUG: "1",
          OPENVIKING_DEBUG_LOG: debugLogPath,
          OPENVIKING_RECALL_COMPRESS: "0",
          OPENVIKING_RECALL_LIMIT: "1",
          OPENVIKING_RECALL_TIMEOUT_MS: "10000",
          OPENVIKING_MIN_QUERY_LENGTH: "1",
          OPENVIKING_SCORE_THRESHOLD: "0",
          OPENVIKING_TIMEOUT_MS: "5000",
          OPENVIKING_URL: baseUrl,
        },
      );

      hookOutput = JSON.parse(result.stdout.trim());
      assert.match(hookOutput.hookSpecificOutput.additionalContext, new RegExp(memoryContent));
    });

    assert.equal(requests.length, 3);
    assert.deepEqual(
      requests.map((request) => [request.body.target_uri, Boolean(request.body.session_id)]).sort(),
      [
        ["viking://user/memories", true],
        ["viking://user/skills", false],
        ["viking://user/skills", true],
      ],
    );
    const debugLog = await readFile(debugLogPath, "utf-8");
    const debugEntries = debugLog.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
    const serializedLog = JSON.stringify(debugEntries);
    for (const secret of [rawPrompt, memoryUri, memoryAbstract, memoryContent]) {
      assert.doesNotMatch(serializedLog, new RegExp(secret.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    }
    assert.doesNotMatch(serializedLog, /"uris"\s*:/);
    assert.doesNotMatch(serializedLog, /"items"\s*:/);
    const outcome = debugEntries.find((entry) => entry.stage === "recall_outcome");
    assert.equal(outcome.data.path, "fallback_search");
    assert.equal(outcome.data.outcome, "emitted");
    assert.deepEqual(outcome.data.server.selected_by_type, { events: 1 });
    assert.deepEqual(outcome.data.server.returned_by_type, {});
    assert.equal(outcome.data.output.basis, "fallback_digest");
    assert.equal(outcome.data.output.chars, hookOutput.hookSpecificOutput.additionalContext.length);
    assert.equal(outcome.data.output.uri_reference_count, 1);
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("auto-recall prefers the server recall endpoint when available", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-auto-recall-endpoint-"));
  const debugLogPath = join(stateDir, "debug.log");
  const requests = [];
  const rawPrompt = "RAW_QUERY_SENTINEL use server recall";
  const eventUri = "viking://user/secret/memories/events/private-launch.md";
  const experienceUri = "viking://user/secret/memories/experiences/private-verification.md";
  const rootUri = "viking://user/secret/memories";
  const eventContent = "RAW_EVENT_CONTENT_SENTINEL";
  const experienceContent = "RAW_EXPERIENCE_CONTENT_SENTINEL";
  const rendered = [
    '<memory_group type="events" count="1">',
    '<memory index="1" type="summary">',
    `  <uri>${eventUri}</uri>`,
    `  <summary>${eventContent}</summary>`,
    '</memory>',
    '</memory_group>',
    '<memory_group type="experiences" count="1">',
    '<memory index="2" type="summary">',
    `  <uri>${experienceUri}</uri>`,
    `  <summary>${experienceContent}</summary>`,
    '</memory>',
    '</memory_group>',
  ].join("\n");
  let hookOutput = null;

  try {
    await withMockOpenViking(async (req, res) => {
      const url = new URL(req.url, "http://127.0.0.1");
      if (req.method === "GET" && url.pathname === "/health") {
        writeJson(res, { status: "ok", result: { ok: true } });
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/v1/search/recall") {
        const body = await readRequestBody(req);
        requests.push({ path: url.pathname, body });
        writeJson(res, {
          status: "ok",
          result: {
            entries: [
              {
                uri: eventUri,
                score: 0.9,
                type: "events",
                mode: "summary",
                rank: 1,
                summary: eventContent,
              },
              {
                uri: experienceUri,
                score: 0.85,
                type: "experiences",
                mode: "summary",
                rank: 2,
                summary: experienceContent,
              },
            ],
            rendered,
            stats: {
              quotas: { events: 2, entities: 2, preferences: 2, experiences: 3 },
              roots: [rootUri],
              searched: { events: 4, entities: 3, preferences: 2, experiences: 5 },
              retrieved_by_type: { events: 3, entities: 2, preferences: 1, experiences: 4 },
              selected_by_type: { events: 1, entities: 0, preferences: 0, experiences: 1 },
              returned_by_type: { events: 1, entities: 0, preferences: 0, experiences: 1 },
              returned_by_mode: { full: 0, summary: 2, uri: 0 },
              excluded_by_type_reason: {
                events: { missing_or_profile_uri: 1, duplicate_content: 0, budget: 1 },
                entities: { missing_or_profile_uri: 0, duplicate_content: 1, budget: 1 },
              },
              origins: { actor_peer: 1, self: 0, other_peer: 1 },
              returned: 2,
              dropped: 3,
              rendered_chars: rendered.length,
              max_chars: 6500,
              min_score: 0,
              peer_scope: "all",
              raw_payload: experienceContent,
            },
          },
        });
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/v1/search/search") {
        requests.push({ path: url.pathname, body: await readRequestBody(req) });
        writeStatusJson(res, 500, { status: "error", error: "should not fallback" });
        return;
      }
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "error", error: "not found" }));
    }, async (baseUrl) => {
      const result = await runAutoRecall(
        { prompt: rawPrompt, session_id: "codex:recall" },
        {
          OPENVIKING_AUTO_RECALL: "1",
          OPENVIKING_CODEX_STATE_DIR: stateDir,
          OPENVIKING_CONFIG_FILE: join(stateDir, "missing-ov.conf"),
          OPENVIKING_CLI_CONFIG_FILE: join(stateDir, "missing-ovcli.conf"),
          OPENVIKING_CREDENTIAL_SOURCE: "env",
          OPENVIKING_DEBUG: "1",
          OPENVIKING_DEBUG_LOG: debugLogPath,
          OPENVIKING_RECALL_COMPRESS: "0",
          OPENVIKING_RECALL_LIMIT: "2",
          OPENVIKING_RECALL_TIMEOUT_MS: "10000",
          OPENVIKING_MIN_QUERY_LENGTH: "1",
          OPENVIKING_SCORE_THRESHOLD: "0",
          OPENVIKING_TIMEOUT_MS: "5000",
          OPENVIKING_URL: baseUrl,
        },
      );

      hookOutput = JSON.parse(result.stdout.trim());
      assert.match(hookOutput.hookSpecificOutput.additionalContext, /OpenViking memory digest/);
      assert.match(hookOutput.hookSpecificOutput.additionalContext, new RegExp(eventContent));
      assert.match(hookOutput.hookSpecificOutput.additionalContext, new RegExp(experienceContent));
    });

    assert.deepEqual(requests.map((request) => request.path), ["/api/v1/search/recall"]);
    assert.equal(requests[0].body.quotas.events, 2);
    assert.equal(requests[0].body.quotas.experiences, 3);
    assert.equal(requests[0].body.max_chars, 6500);

    const debugLog = await readFile(debugLogPath, "utf-8");
    const debugEntries = debugLog.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
    const serializedLog = JSON.stringify(debugEntries);
    for (const secret of [rawPrompt, eventUri, experienceUri, rootUri, eventContent, experienceContent]) {
      assert.doesNotMatch(serializedLog, new RegExp(secret.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    }
    assert.doesNotMatch(serializedLog, /"uris"\s*:/);
    assert.doesNotMatch(serializedLog, /"items"\s*:/);

    const start = debugEntries.find((entry) => entry.stage === "start");
    assert.equal(start.data.query_length, rawPrompt.length);
    assert.equal(Object.hasOwn(start.data, "query"), false);
    assert.equal(Object.hasOwn(start.data, "queryLength"), false);

    const outcome = debugEntries.find((entry) => entry.stage === "recall_outcome");
    assert.equal(outcome.data.schema_version, 1);
    assert.equal(outcome.data.path, "type_quota_endpoint");
    assert.equal(outcome.data.outcome, "emitted");
    assert.deepEqual(outcome.data.server.returned_by_type, { events: 1, experiences: 1 });
    assert.deepEqual(outcome.data.server.returned_by_mode, { summary: 2 });
    assert.deepEqual(outcome.data.server.rank, { known_count: 2, min: 1, max: 2 });
    assert.equal(outcome.data.server.rendered_chars, rendered.length);
    assert.equal(outcome.data.server.stats.root_count, 1);
    assert.deepEqual(outcome.data.server.stats.retrieved_by_type, {
      events: 3,
      entities: 2,
      preferences: 1,
      experiences: 4,
    });
    assert.equal(Object.hasOwn(outcome.data.server.stats, "roots"), false);
    assert.equal(Object.hasOwn(outcome.data.server.stats, "raw_payload"), false);
    assert.equal(outcome.data.output.basis, "server_render");
    assert.equal(outcome.data.output.chars, hookOutput.hookSpecificOutput.additionalContext.length);
    assert.equal(outcome.data.output.uri_reference_count, 2);
    assert.deepEqual(outcome.data.compression, { enabled: false, outcome: "disabled" });
    assert.equal(outcome.data.top_score, 0.9);
    assert.ok(Number.isFinite(outcome.data.latency_ms) && outcome.data.latency_ms >= 0);
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("auto-recall applies the relevance compressor to server recall entries", async () => {
  const result = await runEndpointCompressionCase({
    prompt: "Explain HTTP 429",
    entry: {
      uri: "viking://user/zeus/memories/events/unrelated.md",
      score: 0.42,
      type: "events",
      mode: "summary",
      summary: "Unrelated remembered detail",
    },
    rendered: "<memory_group>Unrelated remembered detail</memory_group>",
    compressorOutput: "NO_RELEVANT_MEMORY",
  });

  assert.deepEqual(result.output, {});
  assert.equal(result.compressorCalls, 1);
  assert.equal(result.requestBody.max_chars, 18000);
});

test("auto-recall reports fallback HTTP 500 failures as fallback_failed", async () => {
  const result = await runFallbackClassificationCase({ fallbackStatus: 500 });

  assert.deepEqual(result.output, {});
  const outcome = result.debugEntries.find((entry) => entry.stage === "recall_outcome");
  assert.equal(outcome.data.path, "fallback_search");
  assert.equal(outcome.data.outcome, "fallback_failed");
  assert.deepEqual(outcome.data.server.transport, {
    endpoint: "failed",
    fallback: { attempted_count: 4, succeeded_count: 0, failed_count: 4 },
  });
  const telemetryText = JSON.stringify(result.debugEntries);
  assert.doesNotMatch(telemetryText, /PRIVATE_FALLBACK_TRANSPORT_PROMPT/);
  assert.doesNotMatch(telemetryText, /PRIVATE_ENDPOINT_ERROR_TEXT|PRIVATE_FALLBACK_ERROR_TEXT/);
  assert.doesNotMatch(telemetryText, /viking:\/\//);
  assert.doesNotMatch(telemetryText, /"uris"\s*:|"items"\s*:/);
});

test("auto-recall reports endpoint timeout plus empty fallback as degraded_no_results", async () => {
  const result = await runFallbackClassificationCase({ endpointDelayMs: 1300 });

  assert.deepEqual(result.output, {});
  const endpointFallback = result.debugEntries.find((entry) => entry.stage === "recall_endpoint_fallback");
  assert.equal(endpointFallback.data.status, 0);
  const outcome = result.debugEntries.find((entry) => entry.stage === "recall_outcome");
  assert.equal(outcome.data.path, "fallback_search");
  assert.equal(outcome.data.outcome, "degraded_no_results");
  assert.deepEqual(outcome.data.server.transport, {
    endpoint: "failed",
    fallback: { attempted_count: 4, succeeded_count: 4, failed_count: 0 },
  });
  const telemetryText = JSON.stringify(result.debugEntries);
  assert.doesNotMatch(telemetryText, /PRIVATE_FALLBACK_TRANSPORT_PROMPT|PRIVATE_ENDPOINT_ERROR_TEXT/);
  assert.doesNotMatch(telemetryText, /viking:\/\//);
  assert.doesNotMatch(telemetryText, /"uris"\s*:|"items"\s*:/);
});

test("auto-recall reports fallback search timeouts as fallback_failed", async () => {
  const result = await runFallbackClassificationCase({ fallbackDelayMs: 1300, sessionId: "" });

  assert.deepEqual(result.output, {});
  const outcome = result.debugEntries.find((entry) => entry.stage === "recall_outcome");
  assert.equal(outcome.data.path, "fallback_search");
  assert.equal(outcome.data.outcome, "fallback_failed");
  assert.deepEqual(outcome.data.server.transport, {
    endpoint: "failed",
    fallback: { attempted_count: 2, succeeded_count: 0, failed_count: 2 },
  });
  const telemetryText = JSON.stringify(result.debugEntries);
  assert.doesNotMatch(telemetryText, /PRIVATE_FALLBACK_TRANSPORT_PROMPT/);
  assert.doesNotMatch(telemetryText, /PRIVATE_ENDPOINT_ERROR_TEXT|PRIVATE_FALLBACK_ERROR_TEXT/);
  assert.doesNotMatch(telemetryText, /viking:\/\//);
  assert.doesNotMatch(telemetryText, /"uris"\s*:|"items"\s*:/);
});

test("compressed endpoint recall logs aggregate telemetry without leaking recall data", async () => {
  const rawPrompt = "RAW_COMPRESSED_QUERY_SENTINEL";
  const entryUri = "viking://user/private/memories/experiences/secret-learning.md";
  const rootUri = "viking://user/private/memories";
  const recalledContent = "RAW_RECALLED_CONTENT_SENTINEL";
  const compressedContent = "COMPRESSED_OUTPUT_SENTINEL";
  const result = await runEndpointCompressionCase({
    prompt: rawPrompt,
    entry: {
      uri: entryUri,
      score: 0.88,
      type: "experiences",
      mode: "summary",
      rank: 3,
      summary: recalledContent,
    },
    rendered: `<memory_group>${recalledContent} ${entryUri}</memory_group>`,
    compressorOutput: `OpenViking memory digest:\n- ${compressedContent} (${entryUri})`,
    serverStats: {
      roots: [rootUri],
      searched: { events: 0, entities: 0, preferences: 0, experiences: 5 },
      retrieved_by_type: { events: 0, entities: 0, preferences: 0, experiences: 4 },
      selected_by_type: { events: 0, entities: 0, preferences: 0, experiences: 3 },
      returned_by_type: { events: 0, entities: 0, preferences: 0, experiences: 1 },
      returned_by_mode: { full: 0, summary: 1, uri: 0 },
      excluded_by_type_reason: {
        experiences: { missing_or_profile_uri: 0, duplicate_content: 1, budget: 1 },
      },
      returned: 1,
      dropped: 2,
      rendered_chars: 123,
      raw_payload: recalledContent,
    },
  });

  const additionalContext = result.output.hookSpecificOutput.additionalContext;
  assert.match(additionalContext, new RegExp(compressedContent));
  assert.match(additionalContext, new RegExp(entryUri.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));

  const serializedLog = JSON.stringify(result.debugEntries);
  for (const secret of [rawPrompt, entryUri, rootUri, recalledContent, compressedContent]) {
    assert.doesNotMatch(serializedLog, new RegExp(secret.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }

  const outcome = result.debugEntries.find((entry) => entry.stage === "recall_outcome");
  assert.equal(outcome.data.schema_version, 1);
  assert.equal(outcome.data.path, "type_quota_endpoint");
  assert.equal(outcome.data.outcome, "emitted");
  assert.deepEqual(outcome.data.server.returned_by_type, { experiences: 1 });
  assert.deepEqual(outcome.data.server.returned_by_mode, { summary: 1 });
  assert.deepEqual(outcome.data.server.rank, { known_count: 1, min: 3, max: 3 });
  assert.equal(outcome.data.server.stats.root_count, 1);
  assert.deepEqual(outcome.data.server.stats.excluded_by_type_reason, {
    experiences: { missing_or_profile_uri: 0, duplicate_content: 1, budget: 1 },
  });
  assert.equal(outcome.data.output.basis, "compressed_digest");
  assert.equal(outcome.data.output.chars, additionalContext.length);
  assert.equal(outcome.data.output.uri_reference_count, 1);
  assert.deepEqual(Object.keys(outcome.data.output).sort(), ["basis", "chars", "uri_reference_count"]);
  assert.deepEqual(outcome.data.compression, { enabled: true, outcome: "compressed" });
  assert.equal(outcome.data.top_score, 0.88);
  assert.ok(Number.isFinite(outcome.data.latency_ms) && outcome.data.latency_ms >= 0);
  assert.doesNotMatch(
    JSON.stringify(outcome.data),
    /surviv|emitted_count|item_count|"uris"\s*:|"items"\s*:/i,
  );
});

test("auto-recall falls back to a bounded deterministic digest when endpoint compression fails", async () => {
  const stderrSecret = "RAW_COMPRESSOR_STDERR_SENTINEL viking://user/private/from-stderr.md";
  const result = await runEndpointCompressionCase({
    prompt: "Which editor do I prefer?",
    entry: {
      uri: "viking://user/zeus/memories/preferences/editor.md",
      score: 0.91,
      type: "preferences",
      mode: "summary",
      summary: "Use Vim",
    },
    rendered: "<memory_group>Use Vim</memory_group>",
    compressorOutput: "",
    exitCode: 1,
    stderrOutput: stderrSecret,
  });

  assert.match(result.output.hookSpecificOutput.additionalContext, /Use Vim/);
  assert.doesNotMatch(result.output.hookSpecificOutput.additionalContext, /<memory_group>/);
  assert.equal(result.compressorCalls, 1);
  const serializedLog = JSON.stringify(result.debugEntries);
  assert.doesNotMatch(serializedLog, /RAW_COMPRESSOR_STDERR_SENTINEL/);
  assert.doesNotMatch(serializedLog, /viking:\/\/user\/private\/from-stderr\.md/);
  const compressExit = result.debugEntries.find((entry) => entry.stage === "compress_exit");
  assert.equal(compressExit.data.exit_code, 1);
  assert.equal(compressExit.data.stderr_chars, stderrSecret.length);
  const outcome = result.debugEntries.find((entry) => entry.stage === "recall_outcome");
  assert.equal(outcome.data.output.basis, "fallback_digest");
  assert.deepEqual(outcome.data.compression, { enabled: true, outcome: "runtime_failed" });
});

test("auto-recall preserves recalled memory when compressor spawn throws synchronously", async () => {
  const preloadDir = await mkdtemp(join(tmpdir(), "ov-sync-spawn-failure-"));
  const preloadPath = join(preloadDir, "throw-codex-spawn.cjs");
  await writeFile(preloadPath, `
const childProcess = require("node:child_process");
const { syncBuiltinESMExports } = require("node:module");
const originalSpawn = childProcess.spawn;
childProcess.spawn = function patchedSpawn(command, ...args) {
  const argv = Array.isArray(args[0]) ? args[0] : [];
  const launchesCodexEntryPoint = argv.some((arg) => {
    const value = String(arg).replaceAll("\\\\", "/");
    return value.includes("/node_modules/@openai/codex/bin/codex.js");
  });
  if (command === "codex" || launchesCodexEntryPoint) {
    throw Object.assign(new Error("spawn EPERM"), { code: "EPERM" });
  }
  return originalSpawn.call(this, command, ...args);
};
syncBuiltinESMExports();
`);

  try {
    const result = await runEndpointCompressionCase({
      prompt: "Which editor do I prefer?",
      entry: {
        uri: "viking://user/zeus/memories/preferences/editor.md",
        score: 0.91,
        type: "preferences",
        mode: "summary",
        summary: "Use Vim",
      },
      rendered: "<memory_group>Use Vim</memory_group>",
      compressorOutput: "unused",
      extraEnv: { NODE_OPTIONS: `--require=${preloadPath}` },
    });

    assert.match(result.output.hookSpecificOutput.additionalContext, /Use Vim/);
    assert.doesNotMatch(result.output.hookSpecificOutput.additionalContext, /<memory_group>/);
    assert.equal(result.compressorCalls, 0);
  } finally {
    await rm(preloadDir, { recursive: true, force: true });
  }
});

test("auto-recall expands configured user in memory search target", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-auto-recall-user-target-"));
  const requests = [];

  try {
    await withMockOpenViking(async (req, res) => {
      const url = new URL(req.url, "http://127.0.0.1");
      if (req.method === "GET" && url.pathname === "/health") {
        writeJson(res, { status: "ok", result: { ok: true } });
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/v1/search/search") {
        const body = await readRequestBody(req);
        requests.push({ path: url.pathname, body });
        if (body.target_uri === "viking://user/zeus/memories") {
          writeJson(res, {
            status: "ok",
            result: {
              memories: [{
                uri: "viking://user/zeus/memories/entities/project/example.md",
                level: 2,
                score: 0.9,
                category: "entities",
                abstract: "configured user memory",
              }],
              skills: [],
            },
          });
          return;
        }
        writeJson(res, { status: "ok", result: { memories: [], skills: [] } });
        return;
      }
      if (req.method === "GET" && url.pathname === "/api/v1/content/read") {
        writeJson(res, { status: "ok", result: "configured user recalled detail" });
        return;
      }
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "error", error: "not found" }));
    }, async (baseUrl) => {
      const result = await runAutoRecall(
        { prompt: "please use configured user memory", session_id: "codex:456" },
        {
          OPENVIKING_AUTO_RECALL: "1",
          OPENVIKING_CODEX_STATE_DIR: stateDir,
          OPENVIKING_CONFIG_FILE: join(stateDir, "missing-ov.conf"),
          OPENVIKING_CLI_CONFIG_FILE: join(stateDir, "missing-ovcli.conf"),
          OPENVIKING_CREDENTIAL_SOURCE: "env",
          OPENVIKING_USER: "zeus",
          OPENVIKING_RECALL_COMPRESS: "0",
          OPENVIKING_RECALL_LIMIT: "1",
          OPENVIKING_RECALL_TIMEOUT_MS: "10000",
          OPENVIKING_MIN_QUERY_LENGTH: "1",
          OPENVIKING_SCORE_THRESHOLD: "0",
          OPENVIKING_TIMEOUT_MS: "5000",
          OPENVIKING_URL: baseUrl,
        },
      );

      const output = JSON.parse(result.stdout.trim());
      assert.match(
        output.hookSpecificOutput.additionalContext,
        /configured user recalled detail/,
      );
    });

    // Memory and skill searches run in parallel; arrival order is not guaranteed.
    const memorySearch = requests.find(
      (request) => request.body.target_uri === "viking://user/zeus/memories",
    );
    assert.ok(memorySearch, "expected a memories search request");
    assert.equal(memorySearch.body.session_id, "cx-codex_456");
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("auto-recall preserves explicit default user memory target", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "ov-auto-recall-default-user-"));
  const requests = [];

  try {
    await withMockOpenViking(async (req, res) => {
      const url = new URL(req.url, "http://127.0.0.1");
      if (req.method === "GET" && url.pathname === "/health") {
        writeJson(res, { status: "ok", result: { ok: true } });
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/v1/search/search") {
        const body = await readRequestBody(req);
        requests.push({ path: url.pathname, body });
        if (body.target_uri === "viking://user/default/memories") {
          writeJson(res, {
            status: "ok",
            result: {
              memories: [{
                uri: "viking://user/default/memories/preferences/default-food.md",
                level: 2,
                score: 0.9,
                category: "preferences",
                abstract: "explicit default user memory",
              }],
              skills: [],
            },
          });
          return;
        }
        writeJson(res, { status: "ok", result: { memories: [], skills: [] } });
        return;
      }
      if (req.method === "GET" && url.pathname === "/api/v1/content/read") {
        writeJson(res, { status: "ok", result: "explicit default user recalled detail" });
        return;
      }
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: "error", error: "not found" }));
    }, async (baseUrl) => {
      const result = await runAutoRecall(
        { prompt: "please use default user memory", session_id: "codex:789" },
        {
          OPENVIKING_AUTO_RECALL: "1",
          OPENVIKING_CODEX_STATE_DIR: stateDir,
          OPENVIKING_CONFIG_FILE: join(stateDir, "missing-ov.conf"),
          OPENVIKING_CLI_CONFIG_FILE: join(stateDir, "missing-ovcli.conf"),
          OPENVIKING_CREDENTIAL_SOURCE: "env",
          OPENVIKING_USER: "default",
          OPENVIKING_RECALL_COMPRESS: "0",
          OPENVIKING_RECALL_LIMIT: "1",
          OPENVIKING_RECALL_TIMEOUT_MS: "10000",
          OPENVIKING_MIN_QUERY_LENGTH: "1",
          OPENVIKING_SCORE_THRESHOLD: "0",
          OPENVIKING_TIMEOUT_MS: "5000",
          OPENVIKING_URL: baseUrl,
        },
      );

      const output = JSON.parse(result.stdout.trim());
      assert.match(
        output.hookSpecificOutput.additionalContext,
        /explicit default user recalled detail/,
      );
    });

    // Memory and skill searches run in parallel; arrival order is not guaranteed.
    const memorySearch = requests.find(
      (request) => request.body.target_uri === "viking://user/default/memories",
    );
    assert.ok(memorySearch, "expected a memories search request");
    assert.equal(memorySearch.body.session_id, "cx-codex_789");
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});
