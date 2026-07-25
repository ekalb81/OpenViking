import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import http from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const LIB_DIR = dirname(fileURLToPath(import.meta.url));
const AUTO_RECALL = join(LIB_DIR, "..", "auto-recall.mjs");

function readRequestBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      try {
        const raw = Buffer.concat(chunks).toString("utf-8");
        resolve(raw ? JSON.parse(raw) : null);
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

function writeJson(res, value, status = 200) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(value));
}

async function withMockOpenViking(handler, fn) {
  const server = http.createServer((req, res) => {
    Promise.resolve(handler(req, res)).catch((err) => {
      writeJson(res, { status: "error", error: String(err?.stack || err) }, 500);
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
    const child = spawn(process.execPath, [AUTO_RECALL], {
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

function hookEnv(homeDir, baseUrl, debugLog) {
  return {
    OPENVIKING_MEMORY_ENABLED: "1",
    OPENVIKING_AUTO_RECALL: "1",
    OPENVIKING_CONFIG_FILE: join(homeDir, "missing-ov.conf"),
    OPENVIKING_CLI_CONFIG_FILE: join(homeDir, "missing-ovcli.conf"),
    OPENVIKING_CREDENTIAL_SOURCE: "env",
    OPENVIKING_DEBUG: "1",
    OPENVIKING_DEBUG_LOG: debugLog,
    OPENVIKING_HOME: homeDir,
    OPENVIKING_MIN_QUERY_LENGTH: "1",
    OPENVIKING_RECALL_LIMIT: "2",
    OPENVIKING_RECALL_MAX_CONTENT_CHARS: "500",
    OPENVIKING_RECALL_TOKEN_BUDGET: "2000",
    OPENVIKING_SCORE_THRESHOLD: "0",
    OPENVIKING_TIMEOUT_MS: "5000",
    OPENVIKING_URL: baseUrl,
  };
}

async function readTelemetry(homeDir, debugLog) {
  const state = JSON.parse(await readFile(join(homeDir, "state", "last-recall.json"), "utf-8"));
  const debugLines = (await readFile(debugLog, "utf-8"))
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  return { state, debugLines };
}

async function runFallbackClassificationCase({
  endpointDelayMs = 0,
  fallbackDelayMs = 0,
  fallbackStatus = 200,
} = {}) {
  const homeDir = await mkdtemp(join(tmpdir(), "ov-claude-recall-fallback-"));
  const debugLog = join(homeDir, "logs", "claude-hooks.jsonl");
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
        writeJson(res, { status: "error", error: "PRIVATE_ENDPOINT_ERROR_TEXT" }, 500);
        return;
      }
      if (req.method === "GET" && url.pathname === "/api/v1/system/status") {
        writeJson(res, { status: "ok", result: { user: "default" } });
        return;
      }
      if (req.method === "GET" && url.pathname === "/api/v1/fs/ls") {
        writeJson(res, { status: "ok", result: [] });
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/v1/search/find") {
        await readRequestBody(req);
        if (fallbackDelayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, fallbackDelayMs));
          if (res.destroyed) return;
        }
        if (fallbackStatus >= 500) {
          writeJson(res, { status: "error", error: "PRIVATE_FALLBACK_ERROR_TEXT" }, fallbackStatus);
        } else {
          writeJson(res, { status: "ok", result: { memories: [], skills: [] } });
        }
        return;
      }
      writeJson(res, { status: "error", error: "not found" }, 404);
    }, async (baseUrl) => {
      const result = await runAutoRecall(
        { prompt, session_id: "claude:fallback-transport", cwd: homeDir },
        {
          ...hookEnv(homeDir, baseUrl, debugLog),
          OPENVIKING_TIMEOUT_MS: "1000",
        },
      );
      return {
        output: JSON.parse(result.stdout.trim()),
        prompt,
        ...(await readTelemetry(homeDir, debugLog)),
      };
    });
  } finally {
    await rm(homeDir, { recursive: true, force: true });
  }
}

test("endpoint recall records accurate aggregate emission telemetry without prompt or memory payloads", async () => {
  const homeDir = await mkdtemp(join(tmpdir(), "ov-claude-recall-telemetry-"));
  const debugLog = join(homeDir, "logs", "claude-hooks.jsonl");
  const prompt = "RAW_PROMPT_SECRET verify the operation";
  const lessonBody = "LESSON_BODY_SECRET verify before declaring closure";
  const lessonUri = "viking://user/default/memories/experiences/private-lesson.md";
  let requestedQuery = "";

  try {
    await withMockOpenViking(async (req, res) => {
      const url = new URL(req.url, "http://127.0.0.1");
      if (req.method === "GET" && url.pathname === "/health") {
        writeJson(res, { status: "ok", result: { ok: true } });
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/v1/search/recall") {
        const body = await readRequestBody(req);
        requestedQuery = body.query;
        const rendered = [
          '<memory_group type="experiences" count="2">',
          `<memory type="full"><uri>${lessonUri}</uri><content>${lessonBody}</content></memory>`,
          '<memory type="uri"><uri>viking://user/default/memories/experiences/private-hint.md</uri></memory>',
          "</memory_group>",
          '<memory_group type="events" count="1"><memory type="summary">PRIVATE_EVENT_SUMMARY</memory></memory_group>',
        ].join("\n");
        writeJson(res, {
          status: "ok",
          result: {
            entries: [
              { uri: lessonUri, score: 0.82, type: "experiences", mode: "full", content: lessonBody },
              {
                uri: "viking://user/default/memories/events/private-event.md",
                score: 0.91,
                type: "events",
                mode: "summary",
                summary: "PRIVATE_EVENT_SUMMARY",
              },
              {
                uri: "viking://user/default/memories/experiences/private-hint.md",
                score: 0.63,
                type: "experiences",
                mode: "uri",
              },
            ],
            rendered,
            stats: {
              searched: {
                events: 4,
                entities: 2,
                preferences: 0,
                experiences: 3,
                "viking://private/search": 500,
              },
              retrieved_by_type: {
                events: "4",
                entities: 2.9,
                preferences: -1,
                experiences: 3,
                "viking://private/type": 99,
              },
              selected_by_type: { events: 1, entities: "0", preferences: 0, experiences: 2 },
              returned_by_type: { events: 1, entities: 0, preferences: 0, experiences: "2" },
              returned_by_mode: { full: 1, summary: "1", uri: 1, private_mode: 99 },
              excluded_by_type_reason: {
                events: {
                  missing_or_profile_uri: "1",
                  duplicate_content: 0,
                  budget: 0,
                  "viking://private/reason": 99,
                },
                entities: { missing_or_profile_uri: 0, duplicate_content: 2.8, budget: -4 },
                preferences: { missing_or_profile_uri: 0, duplicate_content: 0, budget: 0 },
                experiences: { missing_or_profile_uri: 0, duplicate_content: 0, budget: "2" },
                "viking://private/type": { budget: 99 },
              },
              returned: 3,
              dropped: 2,
              max_chars: 1000,
              rendered_chars: String(rendered.length),
              roots: ["viking://user/default/memories"],
            },
          },
        });
        return;
      }
      writeJson(res, { status: "error", error: "not found" }, 404);
    }, async (baseUrl) => {
      const result = await runAutoRecall(
        { prompt, session_id: "claude:telemetry", cwd: homeDir },
        hookEnv(homeDir, baseUrl, debugLog),
      );
      const output = JSON.parse(result.stdout.trim());
      const additionalContext = output.hookSpecificOutput.additionalContext;
      assert.match(additionalContext, /LESSON_BODY_SECRET/);
      assert.equal(requestedQuery, prompt, "the private prompt still reaches the recall API");

      const { state, debugLines } = await readTelemetry(homeDir, debugLog);
      assert.equal(state.reason, "ok");
      assert.equal(state.count, 3);
      assert.equal(state.content_items, 2);
      assert.equal(state.hint_items, 1);
      assert.deepEqual(state.mode_counts, { full: 1, summary: 1, uri: 1, other: 0 });
      assert.deepEqual(state.type_counts, {
        events: 1,
        entities: 0,
        preferences: 0,
        experiences: 2,
        other: 0,
      });
      assert.equal(state.top_score, 0.91);
      assert.deepEqual(state.server_stats, {
        searched_count: 9,
        returned_count: 3,
        dropped_count: 2,
        max_chars: 1000,
        retrieved_by_type: { events: 4, entities: 2, preferences: 0, experiences: 3 },
        selected_by_type: { events: 1, entities: 0, preferences: 0, experiences: 2 },
        returned_by_type: { events: 1, entities: 0, preferences: 0, experiences: 2 },
        returned_by_mode: { full: 1, summary: 1, uri: 1 },
        excluded_by_type_reason: {
          events: { missing_or_profile_uri: 1, duplicate_content: 0, budget: 0 },
          entities: { missing_or_profile_uri: 0, duplicate_content: 2, budget: 0 },
          preferences: { missing_or_profile_uri: 0, duplicate_content: 0, budget: 0 },
          experiences: { missing_or_profile_uri: 0, duplicate_content: 0, budget: 2 },
        },
        rendered_chars: additionalContext.length
          - "<openviking-context>\n".length
          - "Relevant memory from OpenViking. Use the recall/read MCP tools to expand URIs.\n".length
          - "\n</openviking-context>".length,
      });
      assert.equal(state.output_stats.emitted, true);
      assert.equal(state.output_stats.item_count, 3);
      assert.equal(state.output_stats.chars, additionalContext.length);
      assert.equal(state.output_stats.estimated_tokens, Math.ceil(additionalContext.length / 4));
      assert.equal(state.tokens_used, state.output_stats.estimated_tokens);

      const start = debugLines.find((entry) => entry.stage === "start");
      assert.equal(start.data.query_length, prompt.length);
      assert.equal(Object.hasOwn(start.data, "query"), false);
      assert.equal(Object.hasOwn(start.data, "queryLength"), false);

      const telemetryText = JSON.stringify({ state, debugLines });
      assert.doesNotMatch(telemetryText, /RAW_PROMPT_SECRET/);
      assert.doesNotMatch(telemetryText, /LESSON_BODY_SECRET|PRIVATE_EVENT_SUMMARY/);
      assert.doesNotMatch(telemetryText, /viking:\/\//);
      assert.doesNotMatch(telemetryText, /"utilized"|"applied"|"used_by_agent"/);
    });
  } finally {
    await rm(homeDir, { recursive: true, force: true });
  }
});

test("endpoint entries are not counted as emitted when the server renders no output", async () => {
  const homeDir = await mkdtemp(join(tmpdir(), "ov-claude-recall-no-output-"));
  const debugLog = join(homeDir, "logs", "claude-hooks.jsonl");
  const prompt = "PRIVATE_EMPTY_OUTPUT_PROMPT";

  try {
    await withMockOpenViking(async (req, res) => {
      const url = new URL(req.url, "http://127.0.0.1");
      if (req.method === "GET" && url.pathname === "/health") {
        writeJson(res, { status: "ok", result: { ok: true } });
        return;
      }
      if (req.method === "POST" && url.pathname === "/api/v1/search/recall") {
        await readRequestBody(req);
        writeJson(res, {
          status: "ok",
          result: {
            entries: [{
              uri: "viking://user/default/memories/experiences/not-emitted.md",
              score: 0.99,
              type: "experiences",
              mode: "full",
              content: "PRIVATE_NOT_EMITTED_CONTENT",
            }],
            rendered: "",
            stats: {
              searched: { events: 0, entities: 0, preferences: 0, experiences: 1 },
              returned: 1,
              dropped: 0,
              max_chars: 1000,
            },
          },
        });
        return;
      }
      writeJson(res, { status: "error", error: "not found" }, 404);
    }, async (baseUrl) => {
      const result = await runAutoRecall(
        { prompt, session_id: "claude:no-output", cwd: homeDir },
        hookEnv(homeDir, baseUrl, debugLog),
      );
      assert.deepEqual(JSON.parse(result.stdout.trim()), { decision: "approve" });

      const { state, debugLines } = await readTelemetry(homeDir, debugLog);
      assert.equal(state.reason, "no_results");
      assert.equal(state.count, 0);
      assert.equal(state.content_items, 0);
      assert.equal(state.hint_items, 0);
      assert.deepEqual(state.mode_counts, { full: 0, summary: 0, uri: 0, other: 0 });
      assert.equal(state.top_score, 0);
      assert.equal(state.server_stats.returned_count, 1);
      assert.equal(state.output_stats.emitted, false);
      assert.equal(state.output_stats.item_count, 0);
      assert.equal(state.output_stats.chars, 0);

      const telemetryText = JSON.stringify({ state, debugLines });
      assert.doesNotMatch(telemetryText, /PRIVATE_EMPTY_OUTPUT_PROMPT|PRIVATE_NOT_EMITTED_CONTENT/);
      assert.doesNotMatch(telemetryText, /viking:\/\//);
    });
  } finally {
    await rm(homeDir, { recursive: true, force: true });
  }
});

test("fallback HTTP 500 failures are reported as fallback_failed, not no_results", async () => {
  const result = await runFallbackClassificationCase({ fallbackStatus: 500 });

  assert.deepEqual(result.output, { decision: "approve" });
  assert.equal(result.state.reason, "fallback_failed");
  assert.equal(result.state.endpoint_transport, "failed");
  assert.deepEqual(result.state.fallback_transport, {
    attempted_count: 2,
    succeeded_count: 0,
    failed_count: 2,
  });
  const telemetryText = JSON.stringify({ state: result.state, debugLines: result.debugLines });
  assert.doesNotMatch(telemetryText, /PRIVATE_FALLBACK_TRANSPORT_PROMPT/);
  assert.doesNotMatch(telemetryText, /PRIVATE_ENDPOINT_ERROR_TEXT|PRIVATE_FALLBACK_ERROR_TEXT/);
  assert.doesNotMatch(telemetryText, /viking:\/\//);
});

test("endpoint timeout plus successful empty fallback is reported as degraded_no_results", async () => {
  const result = await runFallbackClassificationCase({ endpointDelayMs: 1300 });

  assert.deepEqual(result.output, { decision: "approve" });
  const endpointFallback = result.debugLines.find((entry) => entry.stage === "recall_endpoint_fallback");
  assert.equal(endpointFallback.data.status, 0);
  assert.equal(result.state.reason, "degraded_no_results");
  assert.equal(result.state.endpoint_transport, "failed");
  assert.deepEqual(result.state.fallback_transport, {
    attempted_count: 2,
    succeeded_count: 2,
    failed_count: 0,
  });
  const telemetryText = JSON.stringify({ state: result.state, debugLines: result.debugLines });
  assert.doesNotMatch(telemetryText, /PRIVATE_FALLBACK_TRANSPORT_PROMPT|PRIVATE_ENDPOINT_ERROR_TEXT/);
  assert.doesNotMatch(telemetryText, /viking:\/\//);
});

test("fallback search timeouts are reported as fallback_failed", async () => {
  const result = await runFallbackClassificationCase({ fallbackDelayMs: 1300 });

  assert.deepEqual(result.output, { decision: "approve" });
  assert.equal(result.state.reason, "fallback_failed");
  assert.equal(result.state.endpoint_transport, "failed");
  assert.deepEqual(result.state.fallback_transport, {
    attempted_count: 2,
    succeeded_count: 0,
    failed_count: 2,
  });
  const telemetryText = JSON.stringify({ state: result.state, debugLines: result.debugLines });
  assert.doesNotMatch(telemetryText, /PRIVATE_FALLBACK_TRANSPORT_PROMPT/);
  assert.doesNotMatch(telemetryText, /PRIVATE_ENDPOINT_ERROR_TEXT|PRIVATE_FALLBACK_ERROR_TEXT/);
  assert.doesNotMatch(telemetryText, /viking:\/\//);
});
