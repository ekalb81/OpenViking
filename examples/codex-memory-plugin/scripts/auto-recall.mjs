#!/usr/bin/env node

/**
 * Auto-Recall Hook Script for Codex.
 *
 * Triggered by UserPromptSubmit hook.
 * Reads `prompt` from stdin → searches OpenViking → returns recalled memories
 * via `hookSpecificOutput.additionalContext` so Codex injects them into the turn.
 *
 * Codex output schema (codex-rs/hooks/schema/generated/user-prompt-submit.command.output.schema.json):
 *   { hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: "<text>" } }
 * — `decision: "approve"` is NOT a codex thing; only `decision: "block"` is. So a no-op
 * is just `{}`.
 */

import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "./config.mjs";
import { trySpawnCodex } from "./codex-launch.mjs";
import { createLogger } from "./debug-log.mjs";
import {
  buildCodexExecArgs,
  fallbackRecallCompressorProfile,
  loadCachedRecallCompressorProfile,
  markRecallCompressorRuntimeFailed,
} from "./recall-compressor-profile.mjs";
import { deriveOvSessionId } from "./session-state.mjs";
import { postRecall } from "./shared/recall-core.mjs";
import { resolveEffectivePeerId } from "./shared/workspace-peer.mjs";

const cfg = loadConfig();
const { log, logError } = createLogger("auto-recall");
const effectivePeer = resolveEffectivePeerId({ cfg, cwd: process.cwd() });

let emitted = false;
let activeCompressor = null;
let compressionInFlight = false;
let recallDeadline = null;
let recallOutcomeLogged = false;
let observedQueryLength = 0;
let activeRecallPath = "none";
const hookStartedAt = Date.now();
const DEFAULT_FINAL_RECALL_CHARS = 6500;
const TELEMETRY_SCHEMA_VERSION = 1;
const MEMORY_TYPES = ["events", "entities", "preferences", "experiences"];
const MEMORY_MODES = ["full", "summary", "uri"];
const MEMORY_ORIGINS = ["actor_peer", "self", "other_peer"];
const EXCLUSION_REASONS = ["missing_or_profile_uri", "duplicate_content", "budget"];
const FALLBACK_CATEGORIES = [...MEMORY_TYPES, "cases", "trajectories", "skills"];

function output(obj, exitAfter = false) {
  if (emitted) return;
  emitted = true;
  if (recallDeadline) clearTimeout(recallDeadline);
  const line = JSON.stringify(obj) + "\n";
  if (exitAfter) {
    process.stdout.write(line, () => process.exit(0));
    return;
  }
  process.stdout.write(line);
}

function wrapRecallContext(additionalContext) {
  const body = sanitizeInjectedText(additionalContext).trim();
  if (!body) return "";
  return [
    '<openviking-context source="auto-recall" format="digest">',
    body,
    "</openviking-context>",
  ].join("\n");
}

function emit(additionalContext) {
  if (!additionalContext) {
    output({});
    return;
  }
  const wrappedContext = wrapRecallContext(additionalContext);
  if (!wrappedContext) {
    output({});
    return;
  }
  output({
    hookSpecificOutput: {
      hookEventName: "UserPromptSubmit",
      additionalContext: wrappedContext,
    },
  });
}

recallDeadline = setTimeout(() => {
  logError("recall_timeout", `timed out after ${cfg.recallTimeoutMs}ms`);
  logRecallOutcome({
    path: activeRecallPath,
    outcome: "timeout",
    compressionOutcome: cfg.recallCompress
      ? (compressionInFlight ? "timeout" : "not_attempted")
      : "disabled",
  });
  try {
    activeCompressor?.kill("SIGKILL");
  } catch { /* best effort */ }
  output({}, true);
}, cfg.recallTimeoutMs);
recallDeadline.unref?.();

async function fetchJSON(path, init = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), cfg.timeoutMs);
  try {
    const headers = { "Content-Type": "application/json" };
    if (cfg.apiKey) {
      headers["Authorization"] = `Bearer ${cfg.apiKey}`;
      headers["X-API-Key"] = cfg.apiKey;
    }
    if (cfg.sendIdentityHeaders && cfg.account) headers["X-OpenViking-Account"] = cfg.account;
    if (cfg.sendIdentityHeaders && cfg.user) headers["X-OpenViking-User"] = cfg.user;
    if (effectivePeer.peerId) headers["X-OpenViking-Actor-Peer"] = effectivePeer.peerId;
    const res = await fetch(`${cfg.baseUrl}${path}`, { ...init, headers, signal: controller.signal });
    const body = await res.json().catch(() => null);
    if (!body) return { ok: false, status: res.status };
    if (!res.ok || body.status === "error") {
      return { ok: false, status: res.status, error: body.error || body };
    }
    return { ok: true, result: body.result ?? body };
  } catch {
    return { ok: false, status: 0 };
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

function clampScore(v) {
  if (typeof v !== "number" || Number.isNaN(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

function countKnownValues(items, allowed, valueFor) {
  const counts = {};
  for (const item of items || []) {
    const raw = String(valueFor(item) || "").toLowerCase();
    const key = allowed.includes(raw) ? raw : "other";
    counts[key] = (counts[key] || 0) + 1;
  }
  return counts;
}

function numericMap(value, allowedKeys) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const result = {};
  for (const key of allowedKeys) {
    const number = Number(value[key]);
    if (Number.isFinite(number)) result[key] = number;
  }
  return result;
}

function sanitizeServerStats(value) {
  const stats = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const safe = {};
  for (const key of ["returned", "dropped", "rendered_chars", "max_chars", "min_score"]) {
    const number = Number(stats[key]);
    if (Number.isFinite(number)) safe[key] = number;
  }
  if (stats.peer_scope === "actor" || stats.peer_scope === "all") {
    safe.peer_scope = stats.peer_scope;
  }
  if (Array.isArray(stats.roots)) safe.root_count = stats.roots.length;

  const quotas = numericMap(stats.quotas, MEMORY_TYPES);
  const searched = numericMap(stats.searched, MEMORY_TYPES);
  const retrievedByType = numericMap(stats.retrieved_by_type, MEMORY_TYPES);
  const selectedByType = numericMap(stats.selected_by_type, MEMORY_TYPES);
  const returnedByType = numericMap(stats.returned_by_type, MEMORY_TYPES);
  const returnedByMode = numericMap(stats.returned_by_mode, MEMORY_MODES);
  const origins = numericMap(stats.origins, MEMORY_ORIGINS);
  const penalties = numericMap(stats.other_peer_penalties, MEMORY_TYPES);
  const excludedByTypeReason = {};
  for (const type of MEMORY_TYPES) {
    const reasons = numericMap(stats.excluded_by_type_reason?.[type], EXCLUSION_REASONS);
    if (Object.keys(reasons).length > 0) excludedByTypeReason[type] = reasons;
  }
  if (Object.keys(quotas).length > 0) safe.quotas = quotas;
  if (Object.keys(searched).length > 0) safe.searched = searched;
  if (Object.keys(retrievedByType).length > 0) safe.retrieved_by_type = retrievedByType;
  if (Object.keys(selectedByType).length > 0) safe.selected_by_type = selectedByType;
  if (Object.keys(returnedByType).length > 0) safe.returned_by_type = returnedByType;
  if (Object.keys(returnedByMode).length > 0) safe.returned_by_mode = returnedByMode;
  if (Object.keys(excludedByTypeReason).length > 0) safe.excluded_by_type_reason = excludedByTypeReason;
  if (Object.keys(origins).length > 0) safe.origins = origins;
  if (Object.keys(penalties).length > 0) safe.other_peer_penalties = penalties;
  return safe;
}

function summarizeRanks(entries) {
  const ranks = (entries || [])
    .map((entry) => Number(entry?.rank))
    .filter((rank) => Number.isFinite(rank) && rank > 0);
  return {
    known_count: ranks.length,
    min: ranks.length > 0 ? Math.min(...ranks) : null,
    max: ranks.length > 0 ? Math.max(...ranks) : null,
  };
}

function endpointServerTelemetry(entries, stats, rendered) {
  return {
    stats: sanitizeServerStats(stats),
    returned_by_type: countKnownValues(entries, MEMORY_TYPES, (entry) => entry?.type),
    returned_by_mode: countKnownValues(entries, MEMORY_MODES, (entry) => entry?.mode),
    rank: summarizeRanks(entries),
    rendered_chars: String(rendered || "").length,
  };
}

function fallbackServerTelemetry(raw, eligible, picked, transport = null) {
  const telemetry = {
    stats: {
      retrieved: Array.isArray(raw) ? raw.length : 0,
      eligible: Array.isArray(eligible) ? eligible.length : 0,
      selected: Array.isArray(picked) ? picked.length : 0,
    },
    selected_by_type: countKnownValues(picked, FALLBACK_CATEGORIES, (entry) => entry?.category),
    returned_by_type: {},
    returned_by_mode: {},
    rank: { known_count: 0, min: null, max: null },
    rendered_chars: 0,
  };
  if (transport) {
    telemetry.transport = {
      endpoint: "failed",
      fallback: {
        attempted_count: Math.max(0, Math.floor(Number(transport.attempted_count) || 0)),
        succeeded_count: Math.max(0, Math.floor(Number(transport.succeeded_count) || 0)),
        failed_count: Math.max(0, Math.floor(Number(transport.failed_count) || 0)),
      },
    };
  }
  return telemetry;
}

function countUriReferences(value) {
  return (String(value || "").match(/\bviking:\/\/[^\s<>"']+/gi) || []).length;
}

function logRecallOutcome({
  path = activeRecallPath,
  outcome,
  server = {},
  outputBasis = "none",
  outputText = "",
  compressionOutcome = cfg.recallCompress ? "not_attempted" : "disabled",
  topScore = 0,
} = {}) {
  if (recallOutcomeLogged) return;
  recallOutcomeLogged = true;
  const emittedContext = outputText ? wrapRecallContext(outputText) : "";
  log("recall_outcome", {
    schema_version: TELEMETRY_SCHEMA_VERSION,
    path,
    outcome: outcome || "error",
    query_length: observedQueryLength,
    server,
    output: {
      basis: outputBasis,
      chars: emittedContext.length,
      uri_reference_count: countUriReferences(emittedContext),
    },
    compression: {
      enabled: cfg.recallCompress,
      outcome: compressionOutcome,
    },
    top_score: clampScore(Number(topScore)),
    latency_ms: Math.max(0, Date.now() - hookStartedAt),
  });
}

const PREFERENCE_QUERY_RE = /prefer|preference|favorite|favourite|like|偏好|喜欢|爱好|更倾向/i;
const TEMPORAL_QUERY_RE = /when|what time|date|day|month|year|yesterday|today|tomorrow|last|next|什么时候|何时|哪天|几月|几年|昨天|今天|明天/i;
const QUERY_TOKEN_RE = /[a-z0-9一-龥]{2,}/gi;
const STOPWORDS = new Set([
  "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "did", "does",
  "is", "are", "was", "were", "the", "and", "for", "with", "from", "that", "this", "your", "you",
]);

function buildQueryProfile(query) {
  const text = query.trim();
  const allTokens = text.toLowerCase().match(QUERY_TOKEN_RE) || [];
  const tokens = allTokens.filter((t) => !STOPWORDS.has(t));
  return {
    tokens,
    wantsPreference: PREFERENCE_QUERY_RE.test(text),
    wantsTemporal: TEMPORAL_QUERY_RE.test(text),
  };
}

function lexicalOverlapBoost(tokens, text) {
  if (tokens.length === 0 || !text) return 0;
  const haystack = ` ${text.toLowerCase()} `;
  let matched = 0;
  for (const token of tokens.slice(0, 8)) {
    if (haystack.includes(token)) matched += 1;
  }
  return Math.min(0.2, (matched / Math.min(tokens.length, 4)) * 0.2);
}

function getRankingBreakdown(item, profile) {
  const base = clampScore(item.score);
  const abstract = (item.abstract || item.overview || "").trim();
  const cat = (item.category || "").toLowerCase();
  const uri = item.uri.toLowerCase();
  const leafBoost = (item.level === 2 || uri.endsWith(".md")) ? 0.12 : 0;
  const eventBoost = profile.wantsTemporal && (cat === "events" || uri.includes("/events/")) ? 0.1 : 0;
  const prefBoost = profile.wantsPreference && (cat === "preferences" || uri.includes("/preferences/")) ? 0.08 : 0;
  const overlapBoost = lexicalOverlapBoost(profile.tokens, `${item.uri} ${abstract}`);
  return {
    baseScore: base,
    leafBoost,
    eventBoost,
    prefBoost,
    overlapBoost,
    finalScore: base + leafBoost + eventBoost + prefBoost + overlapBoost,
  };
}

function rankForInjection(item, profile) {
  return getRankingBreakdown(item, profile).finalScore;
}

function dedupeByAbstract(items) {
  const seen = new Set();
  return items.filter((item) => {
    const key = (item.abstract || item.overview || "").trim().toLowerCase() || item.uri;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function pickMemories(items, limit, queryText) {
  if (items.length === 0 || limit <= 0) return [];
  const profile = buildQueryProfile(queryText);
  const sorted = [...items].sort((a, b) => rankForInjection(b, profile) - rankForInjection(a, profile));
  const deduped = dedupeByAbstract(sorted);
  const leaves = deduped.filter((m) => m.level === 2 || m.uri.endsWith(".md"));
  if (leaves.length >= limit) return leaves.slice(0, limit);
  const picked = [...leaves];
  const used = new Set(picked.map((m) => m.uri));
  for (const item of deduped) {
    if (picked.length >= limit) break;
    if (used.has(item.uri)) continue;
    picked.push(item);
  }
  return picked;
}

function postProcess(items, limit, threshold) {
  const seen = new Set();
  const sorted = [...items].sort((a, b) => clampScore(b.score) - clampScore(a.score));
  const result = [];
  for (const item of sorted) {
    if (item.level !== 2) continue;
    if (clampScore(item.score) < threshold) continue;
    const cat = (item.category || "").toLowerCase() || "unknown";
    const abs = (item.abstract || item.overview || "").trim().toLowerCase();
    const key = abs ? `${cat}:${abs}` : `uri:${item.uri}`;
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(item);
    if (result.length >= limit) break;
  }
  return result;
}

async function searchScope(query, targetUri, limit, bucket = "memories", sessionId = null) {
  const body = { query, target_uri: targetUri, limit, score_threshold: 0 };
  if (sessionId) body.session_id = sessionId;
  const result = await fetchJSON("/api/v1/search/search", {
    method: "POST",
    body: JSON.stringify(body),
  });
  if (!result.ok) return { ok: false, items: [] };
  return {
    ok: true,
    items: Array.isArray(result.result?.[bucket]) ? result.result[bucket] : [],
  };
}

// Candidate target URIs for a bucket, most-specific first. In trusted mode a
// user's memories live under viking://user/<user>/<bucket>; in api_key mode the
// server canonicalizes the generic viking://user/<bucket> to the authenticated
// user. Trying the user-scoped path first with the generic path as a fallback
// recalls correctly in both modes. De-duped so a missing/blank user never
// doubles the request count.
function userScopedTargets(kind) {
  const suffix = kind.replace(/^\/+/, "");
  const targets = [`viking://user/${suffix}`];
  if (cfg.user) {
    targets.unshift(`viking://user/${cfg.user}/${suffix}`);
  }
  return [...new Set(targets)];
}

// Two-phase, short-circuiting search over the candidate targets:
//   1) a session-scoped pass (uses OpenViking's session-aware planner);
//   2) only if the entire session pass is empty, a single session-independent
//      pass (the planner can legitimately decide that no extra context is
//      needed for this session, but auto-recall still needs a memory lookup).
// Each phase stops at the first non-empty target, so a warm user costs one
// request and the worst case is bounded by (targets x 2) — instead of running
// a per-target session+fallback for every target.
async function searchBucket(query, targetUris, limit, bucket, sessionId = null) {
  const transport = { attempted_count: 0, succeeded_count: 0, failed_count: 0 };
  const search = async (targetUri, activeSessionId) => {
    let result;
    try {
      result = await searchScope(query, targetUri, limit, bucket, activeSessionId);
    } catch {
      result = { ok: false, items: [] };
    }
    transport.attempted_count += 1;
    transport[result.ok ? "succeeded_count" : "failed_count"] += 1;
    return result.items;
  };
  for (const targetUri of targetUris) {
    const items = await search(targetUri, sessionId);
    if (items.length > 0) return { items, transport };
  }
  if (!sessionId) return { items: [], transport };
  for (const targetUri of targetUris) {
    const items = await search(targetUri, null);
    if (items.length > 0) return { items, transport };
  }
  return { items: [], transport };
}

async function searchAll(query, limit, sessionId = null) {
  const [memoryResult, skillResult] = await Promise.all([
    searchBucket(query, userScopedTargets("memories"), limit, "memories", sessionId),
    searchBucket(query, userScopedTargets("skills"), limit, "skills", sessionId),
  ]);
  const userMems = memoryResult.items;
  const userSkills = skillResult.items;
  const transport = {
    attempted_count: memoryResult.transport.attempted_count + skillResult.transport.attempted_count,
    succeeded_count: memoryResult.transport.succeeded_count + skillResult.transport.succeeded_count,
    failed_count: memoryResult.transport.failed_count + skillResult.transport.failed_count,
  };
  log("search_complete", { scope: "user", rawCount: userMems.length, topScores: userMems.slice(0, 3).map((m) => m.score) });
  log("search_complete", { scope: "skills", rawCount: userSkills.length, topScores: userSkills.slice(0, 3).map((m) => m.score) });
  const all = [...userMems, ...userSkills];
  const seen = new Set();
  const items = all.filter((m) => {
    if (seen.has(m.uri)) return false;
    seen.add(m.uri);
    return true;
  });
  return { items, transport };
}

function resolveRecallSessionId(codexSessionId) {
  if (!codexSessionId) return null;
  // Derive directly: the OV session id is deterministic (cx-<safe-id>), so
  // recall does not need to read plugin state. This keeps the recall hook
  // crash-free even if the state file is corrupt/missing, and stays in sync
  // with capture, which now also derives cx-* unconditionally.
  return deriveOvSessionId(codexSessionId);
}

async function readMemoryContent(uri) {
  try {
    const result = await fetchJSON(`/api/v1/content/read?uri=${encodeURIComponent(uri)}`);
    if (result.ok && typeof result.result === "string" && result.result.trim()) return result.result.trim();
  } catch { /* fallback */ }
  return null;
}

async function recallViaTypeQuotaEndpoint(query) {
  const body = {
    query,
    quotas: {
      events: Math.max(cfg.recallLimit, 1),
      entities: Math.max(cfg.recallLimit, 1),
      preferences: Math.max(1, Math.min(cfg.recallLimit, 3)),
      experiences: cfg.recallExperiences,
    },
    max_chars: cfg.recallCompress
      ? cfg.recallCompressMaxInputChars
      : DEFAULT_FINAL_RECALL_CHARS,
    min_score: cfg.scoreThreshold,
    render: true,
  };
  if (cfg.recallPeerScope === "actor") body.peer_scope = "actor";
  let result;
  try {
    result = await postRecall(fetchJSON, body, { actorPeerId: effectivePeer.peerId, log });
  } catch {
    result = { ok: false, status: 0 };
  }
  if (!result.ok) {
    log("recall_endpoint_fallback", { status: result.status || 0 });
    return null;
  }
  const rendered = String(result.result?.rendered || "").trim();
  const entries = Array.isArray(result.result?.entries) ? result.result.entries : [];
  const items = entries
    .map((entry) => ({
      uri: String(entry?.uri || "").trim(),
      category: String(entry?.type || "memory").trim() || "memory",
      score: clampScore(entry?.score),
      text: String(
        entry?.content || entry?.summary || entry?.abstract || entry?.uri || "",
      ).trim(),
    }))
    .filter((entry) => entry.uri && entry.text);
  const context = rendered
    ? [
        "OpenViking memory digest:",
        rendered,
        "",
        "More detail: use the OpenViking MCP recall/read/search tools with cited viking:// URIs if needed.",
      ].join("\n")
    : "";
  return {
    context,
    items,
    server: endpointServerTelemetry(entries, result.result?.stats, rendered),
    topScore: entries.reduce((best, entry) => Math.max(best, clampScore(entry?.score)), 0),
  };
}

function truncateText(text, maxChars) {
  const value = String(text || "").trim();
  if (value.length <= maxChars) return value;
  return `${value.slice(0, Math.max(0, maxChars - 20)).trimEnd()}\n[truncated]`;
}

function sanitizeInjectedText(text) {
  return String(text || "")
    .replace(/<\/?relevant-memor(?:y|ies)\b[^>]*>/gi, "legacy memory wrapper")
    .replace(/<\/?openviking-context\b[^>]*>/gi, "openviking context marker");
}

function isNoRelevantMemory(text) {
  const value = String(text || "")
    .trim()
    .replace(/^openviking memory digest:\s*/i, "")
    .trim();
  return !value || /^NO_RELEVANT_MEMORY\.?$/i.test(value) || /^no (?:directly )?relevant memor(?:y|ies)\.?$/i.test(value);
}

function hasDigestSignal(text) {
  const body = String(text || "").replace(/^openviking memory digest:\s*/i, "").trim();
  return /(^|\n)\s*[-*]\s+\S/.test(body) || /\bviking:\/\//i.test(body);
}

function appendMcpRetrievalHint(text) {
  const value = String(text || "").trim();
  if (!/\bviking:\/\//i.test(value) || /OpenViking MCP/i.test(value)) return value;
  return `${value}\n\nMore detail: use the OpenViking MCP read/search tools with the cited viking:// URI if needed.`;
}

function fallbackDigest(items) {
  const lines = items.slice(0, cfg.recallCompressMaxBullets).map((item) => {
    const text = sanitizeInjectedText(truncateText(item.text, 260)).replace(/\s+/g, " ");
    return `- [${item.category || "memory"}] ${text} (${item.uri})`;
  });
  return lines.length > 0 ? appendMcpRetrievalHint(`OpenViking memory digest:\n${lines.join("\n")}`) : "";
}

function normalizeCompressedContext(text) {
  let value = String(text || "").trim();
  if (!value) return "";
  value = value.replace(/^```(?:text|markdown)?\s*/i, "").replace(/\s*```$/i, "").trim();
  value = sanitizeInjectedText(value);
  if (isNoRelevantMemory(value)) return "";
  if (!value.toLowerCase().startsWith("openviking memory digest:")) {
    value = `OpenViking memory digest:\n${value}`;
  }
  if (!hasDigestSignal(value)) return "";
  return truncateText(appendMcpRetrievalHint(value), 4000);
}

async function getRecallCompressorProfile() {
  const cached = await loadCachedRecallCompressorProfile(cfg);
  if (cached) return cached;
  const fallback = fallbackRecallCompressorProfile(cfg);
  log("compress_profile_cache_miss", fallback);
  return fallback;
}

async function runCodexCompressor(prompt, profile) {
  const tmp = await mkdtemp(join(tmpdir(), "ov-recall-compress-"));
  const outputPath = join(tmp, "last-message.txt");
  const args = buildCodexExecArgs(profile, outputPath);

  try {
    return await new Promise((resolve) => {
      const env = {
        ...process.env,
        OPENVIKING_AUTO_RECALL: "0",
        OPENVIKING_AUTO_CAPTURE: "0",
        OPENVIKING_RECALL_COMPRESS: "0",
      };
      let child = null;
      let timer = null;
      let done = false;
      let timedOut = false;
      let stderr = "";
      const finish = (value, { runtimeFailed = false } = {}) => {
        if (done) return;
        done = true;
        if (activeCompressor === child) activeCompressor = null;
        clearTimeout(timer);
        if (runtimeFailed) {
          // Mark the profile as runtime_failed so subsequent UPS calls in
          // this same codex session skip compress (avoids burning
          // ~recallCompressTimeoutMs per turn on a guaranteed-to-fail
          // spawn). Next SessionStart's cache-first detect treats this
          // marker as a cache miss and re-resolves against the current
          // catalogue, so a transient failure self-recovers across codex
          // restarts. Best-effort write; failure is non-fatal.
          markRecallCompressorRuntimeFailed(cfg, { failedModel: profile.model || "" })
            .catch(() => {});
        }
        resolve(value);
      };
      const launch = trySpawnCodex(args, { env, stdio: ["pipe", "ignore", "pipe"] });
      if (launch.error) {
        logError("compress_spawn", launch.error);
        finish(null, { runtimeFailed: true });
        return;
      }
      child = launch.child;
      activeCompressor = child;
      timer = setTimeout(() => {
        timedOut = true;
        logError("compress_timeout", `timed out after ${cfg.recallCompressTimeoutMs}ms`);
        try {
          child.kill("SIGKILL");
        } catch { /* best effort */ }
      }, cfg.recallCompressTimeoutMs);

      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
        if (stderr.length > 4000) stderr = stderr.slice(-4000);
      });
      child.on("error", (err) => {
        logError("compress_spawn", err);
        finish(null, { runtimeFailed: true });
      });
      child.on("close", async (code) => {
        if (timedOut) {
          finish(null, { runtimeFailed: true });
          return;
        }
        if (code !== 0) {
          log("compress_exit", {
            exit_code: code,
            stderr_chars: stderr.length,
            profile,
          });
          finish(null, { runtimeFailed: true });
          return;
        }
        try {
          finish(await readFile(outputPath, "utf-8"));
        } catch (err) {
          logError("compress_read", err);
          finish(null, { runtimeFailed: true });
        }
      });
      child.stdin.end(prompt);
    });
  } finally {
    await rm(tmp, { recursive: true, force: true }).catch(() => {});
  }
}

async function compressMemoryContext(userPrompt, items) {
  if (!cfg.recallCompress) return { context: null, outcome: "disabled" };
  const profile = await getRecallCompressorProfile();
  if (!profile.enabled) {
    log("compress_skip", { reason: "profile disabled", profile });
    return {
      context: null,
      outcome: profile.source === "runtime_failed" ? "runtime_failed" : "profile_disabled",
    };
  }
  const perItemChars = Math.max(500, Math.floor(cfg.recallCompressMaxInputChars / Math.max(1, items.length)));
  const payload = {
    user_prompt: userPrompt,
    max_bullets: cfg.recallCompressMaxBullets,
    memories: items.map((item) => ({
      uri: item.uri,
      category: item.category || "memory",
      score: item.score,
      text: truncateText(item.text, perItemChars),
    })),
  };
  const prompt = `You are a memory relevance compressor for a Codex UserPromptSubmit hook.

Task:
- Keep only memories directly useful for answering the user's current prompt.
- Drop stale, generic, duplicate, merely adjacent, or operationally unrelated memories.
- Compress to at most ${cfg.recallCompressMaxBullets} short bullets.
- Preserve concrete facts, dates, paths, repo names, commands, and user preferences.
- Include the source viking:// URI when the agent may need to inspect more detail.
- If the answer needs detail beyond the bullet, say to use OpenViking MCP read/search with the cited viking:// URI if needed.
- Do not include XML/HTML wrappers.
- Do not mention that you filtered memories.
- Output either "OpenViking memory digest:" followed by useful bullets, or exactly: NO_RELEVANT_MEMORY.
- If no memory is directly useful, output exactly: NO_RELEVANT_MEMORY.

Input JSON:
${JSON.stringify(payload, null, 2)}
`;
  compressionInFlight = true;
  let raw;
  try {
    raw = await runCodexCompressor(prompt, profile);
  } finally {
    compressionInFlight = false;
  }
  if (raw === null) return { context: null, outcome: "runtime_failed" };
  const compressed = normalizeCompressedContext(raw);
  log("compressed", { input_count: items.length, chars: compressed.length, profile });
  return {
    context: compressed,
    outcome: compressed ? "compressed" : "filtered_all",
  };
}

async function main() {
  if (!cfg.autoRecall) {
    log("skip", { stage: "init", reason: "autoRecall disabled" });
    logRecallOutcome({ path: "none", outcome: "disabled", compressionOutcome: "not_attempted" });
    emit();
    return;
  }

  let input;
  try {
    const chunks = [];
    for await (const chunk of process.stdin) chunks.push(chunk);
    input = JSON.parse(Buffer.concat(chunks).toString());
  } catch {
    log("skip", { stage: "stdin_parse", reason: "invalid input" });
    logRecallOutcome({ path: "none", outcome: "bad_stdin", compressionOutcome: "not_attempted" });
    emit();
    return;
  }

  const userPrompt = (input.prompt || "").trim();
  observedQueryLength = userPrompt.length;
  const codexSessionId = typeof input.session_id === "string" ? input.session_id.trim() : "";
  const recallSessionId = resolveRecallSessionId(codexSessionId);
  log("start", {
    codexSessionId: codexSessionId || null,
    recallSessionId,
    query_length: observedQueryLength,
    config: {
      recallLimit: cfg.recallLimit,
      recallExperiences: cfg.recallExperiences,
      scoreThreshold: cfg.scoreThreshold,
      peerSource: effectivePeer.source,
      recallPeerScope: cfg.recallPeerScope,
    },
  });

  if (!userPrompt || userPrompt.length < cfg.minQueryLength) {
    log("skip", { stage: "query_check", reason: "query too short or empty" });
    logRecallOutcome({ path: "none", outcome: "short_query", compressionOutcome: "not_attempted" });
    emit();
    return;
  }

  const health = await fetchJSON("/health");
  if (!health.ok) {
    logError("health_check", "server unreachable or unhealthy");
    logRecallOutcome({ path: "none", outcome: "offline", compressionOutcome: "not_attempted" });
    emit();
    return;
  }

  activeRecallPath = "type_quota_endpoint";
  const endpointRecall = await recallViaTypeQuotaEndpoint(userPrompt);
  if (endpointRecall !== null) {
    if (!endpointRecall.context && endpointRecall.items.length === 0) {
      log("skip", { stage: "recall_endpoint", reason: "no results" });
      logRecallOutcome({
        outcome: "no_results",
        server: endpointRecall.server,
        compressionOutcome: cfg.recallCompress ? "not_attempted" : "disabled",
        topScore: endpointRecall.topScore,
      });
      emit();
      return;
    }
    const compression = endpointRecall.items.length > 0
      ? await compressMemoryContext(userPrompt, endpointRecall.items)
      : { context: null, outcome: cfg.recallCompress ? "not_attempted" : "disabled" };
    const endpointFallback = cfg.recallCompress && endpointRecall.items.length > 0
      ? fallbackDigest(endpointRecall.items)
      : endpointRecall.context;
    const memoryContext = compression.context === null
      ? endpointFallback
      : compression.context;
    const outputBasis = compression.context !== null
      ? "compressed_digest"
      : (cfg.recallCompress && endpointRecall.items.length > 0 ? "fallback_digest" : "server_render");
    if (!memoryContext) {
      log("skip", { stage: "recall_endpoint", reason: "compressor found no relevant memory" });
      logRecallOutcome({
        outcome: compression.outcome === "filtered_all" ? "filtered_all" : "no_results",
        server: endpointRecall.server,
        outputBasis,
        compressionOutcome: compression.outcome,
        topScore: endpointRecall.topScore,
      });
      emit();
      return;
    }
    log("recall_endpoint", {
      chars: memoryContext.length,
      compressed: compression.context !== null,
    });
    logRecallOutcome({
      outcome: "emitted",
      server: endpointRecall.server,
      outputBasis,
      outputText: memoryContext,
      compressionOutcome: compression.outcome,
      topScore: endpointRecall.topScore,
    });
    emit(memoryContext);
    return;
  }

  activeRecallPath = "fallback_search";
  const candidateLimit = Math.max(cfg.recallLimit * 4, 20);
  const fallbackRecall = await searchAll(userPrompt, candidateLimit, recallSessionId);
  const allMemories = fallbackRecall.items;
  if (allMemories.length === 0) {
    const outcome = fallbackRecall.transport.succeeded_count === 0
      ? "fallback_failed"
      : "degraded_no_results";
    log("skip", { stage: "search", reason: outcome, transport: fallbackRecall.transport });
    logRecallOutcome({
      outcome,
      server: fallbackServerTelemetry([], [], [], fallbackRecall.transport),
      compressionOutcome: cfg.recallCompress ? "not_attempted" : "disabled",
    });
    emit();
    return;
  }

  const processed = postProcess(allMemories, candidateLimit, cfg.scoreThreshold);
  log("post_process", { beforeCount: allMemories.length, afterCount: processed.length });

  const profile = buildQueryProfile(userPrompt);
  const ranked = [...processed]
    .map((item) => ({ item, breakdown: getRankingBreakdown(item, profile) }))
    .sort((a, b) => b.breakdown.finalScore - a.breakdown.finalScore);

  if (cfg.logRankingDetails) {
    for (const [index, entry] of ranked.entries()) {
      const category = String(entry.item.category || "").toLowerCase();
      log("ranking_detail", {
        rank: index + 1,
        category: MEMORY_TYPES.includes(category) ? category : "other",
        ...entry.breakdown,
      });
    }
  } else {
    log("ranking_summary", {
      candidateCount: processed.length,
      topScores: ranked.slice(0, 5).map((entry) => entry.breakdown.finalScore),
    });
  }

  const memories = pickMemories(processed, cfg.recallLimit, userPrompt);
  if (memories.length === 0) {
    log("skip", { stage: "pick", reason: "no memories survived ranking" });
    logRecallOutcome({
      outcome: "filtered_all",
      server: fallbackServerTelemetry(allMemories, processed, [], fallbackRecall.transport),
      compressionOutcome: cfg.recallCompress ? "not_attempted" : "disabled",
      topScore: ranked[0]?.item?.score || 0,
    });
    emit();
    return;
  }

  log("picked", {
    pickedCount: memories.length,
    topScore: memories.reduce((best, item) => Math.max(best, clampScore(item.score)), 0),
  });

  const memoryItems = await Promise.all(
    memories.map(async (item) => {
      let text = (item.abstract || item.overview || item.uri).trim();
      if (item.level === 2) {
        const content = await readMemoryContent(item.uri);
        if (content) text = content;
      }
      return {
        uri: item.uri,
        category: item.category || "memory",
        score: clampScore(item.score),
        text,
      };
    }),
  );

  const compression = await compressMemoryContext(userPrompt, memoryItems);
  const memoryContext = compression.context === null ? fallbackDigest(memoryItems) : compression.context;
  const outputBasis = compression.context === null ? "fallback_digest" : "compressed_digest";
  logRecallOutcome({
    outcome: memoryContext ? "emitted" : "filtered_all",
    server: fallbackServerTelemetry(allMemories, processed, memories, fallbackRecall.transport),
    outputBasis,
    outputText: memoryContext,
    compressionOutcome: compression.outcome,
    topScore: memories.reduce((best, item) => Math.max(best, clampScore(item.score)), 0),
  });

  emit(memoryContext);
}

main().catch((err) => {
  logError("uncaught", err);
  logRecallOutcome({
    outcome: "error",
    compressionOutcome: cfg.recallCompress ? "not_attempted" : "disabled",
  });
  emit();
});
