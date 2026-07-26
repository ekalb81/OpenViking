import assert from "node:assert/strict";
import test from "node:test";

/**
 * The actor peer a commit claims decides what the server lets it write.
 *
 * This hook runs at SessionStart in ONE workspace, then sweeps a global state
 * directory and commits sessions belonging to OTHER workspaces. Stamping every
 * request with this process's workspace made the server refuse those writes:
 * the session's messages carry their own peer_id, the request claimed a
 * different one, and the peer-view guard denied it. Measured cost was 7 lost
 * memory writes plus 43 denied reads across four days, all on Codex.
 *
 * These tests exercise the header-selection rule directly. The real fetchJSON
 * is bound to network and config, so the rule is restated here and pinned; the
 * source assertions at the end keep the two from drifting apart.
 */

/** Mirrors fetchJSON's header choice: an explicit value wins, undefined inherits. */
function resolveActorPeer(activePeerId, actorPeerIdArg) {
  return actorPeerIdArg === undefined ? activePeerId : actorPeerIdArg;
}

function headersFor(actorPeer) {
  const headers = {};
  if (actorPeer) headers["X-OpenViking-Actor-Peer"] = actorPeer;
  return headers;
}

test("a swept session commits as its own workspace, not the sweeping process's", () => {
  const sweeper = "C--Users-ekalb-Documents-Codex-2026-07-26-canonical-reference";
  const swept = {
    ovSessionId: "cx-1",
    workspacePeerId: "C--Users-ekalb-Documents-Codex-2026-07-26-a",
  };

  const actor = resolveActorPeer(sweeper, swept.workspacePeerId || "");

  assert.equal(actor, swept.workspacePeerId);
  assert.notEqual(actor, sweeper, "claiming the sweeper's workspace is the bug");
});

test("a state record without a workspace peer claims none at all", () => {
  // Older state files predate the field. Sending no actor leaves the server's
  // peer view inactive - the write lands. Guessing would lose the memory, which
  // is strictly worse than not scoping it.
  const sweeper = "D--projects-agent-odometer";
  const swept = { ovSessionId: "cx-2", workspacePeerId: "" };

  const actor = resolveActorPeer(sweeper, swept.workspacePeerId || "");

  assert.equal(actor, "");
  assert.deepEqual(headersFor(actor), {}, "no actor header should be sent");
});

test("callers that pass nothing still inherit the process peer", () => {
  // Non-commit calls in this hook concern the current session and keep the
  // previous behaviour.
  const actor = resolveActorPeer("D--projects-Contexture", undefined);

  assert.equal(actor, "D--projects-Contexture");
  assert.deepEqual(headersFor(actor), {
    "X-OpenViking-Actor-Peer": "D--projects-Contexture",
  });
});

test("an explicit empty string is not treated as absent", () => {
  // The distinction the whole fix rests on: "" means deliberately unscoped,
  // undefined means "I did not say". Collapsing them with a falsy check would
  // silently reintroduce the sweeper's peer.
  assert.equal(resolveActorPeer("sweeper", ""), "");
  assert.equal(resolveActorPeer("sweeper", undefined), "sweeper");
});

test("the commit path in source actually threads the session's peer through", async () => {
  // A correct rule the call site does not use would change nothing.
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(new URL("./session-start-commit.mjs", import.meta.url), "utf8");

  assert.match(
    src,
    /commitOvSession\(state\.ovSessionId,\s*state\.workspacePeerId\s*\|\|\s*""\)/,
    "commitAndClear must pass the swept session's own peer",
  );
  assert.match(
    src,
    /async function commitOvSession\(ovSessionId,\s*actorPeerId/,
    "commitOvSession must accept an actor peer",
  );
  assert.match(
    src,
    /async function fetchJSON\(path,\s*init = \{\},\s*actorPeerId/,
    "fetchJSON must accept an actor peer override",
  );
  assert.match(
    src,
    /actorPeerId === undefined \? activePeerId : actorPeerId/,
    "undefined must inherit while an explicit value wins",
  );
});
