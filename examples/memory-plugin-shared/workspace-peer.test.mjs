import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import {
  deriveWorkspacePeerId,
  findProjectRoot,
  resolveEffectivePeerId,
} from "./lib/workspace-peer.mjs";

const HOME = path.resolve("/home/u");
const TMP = path.resolve("/tmp");

/** Fake filesystem: maps an absolute path to "dir" or to file contents. */
function io(entries = {}) {
  const norm = (p) => path.resolve(p);
  const map = new Map(Object.entries(entries).map(([k, v]) => [norm(k), v]));
  return {
    existsSync: (p) => map.has(norm(p)),
    statSync: (p) => {
      const v = map.get(norm(p));
      if (v === undefined) throw new Error("ENOENT");
      return { isDirectory: () => v === "dir" };
    },
    readFileSync: (p) => {
      const v = map.get(norm(p));
      if (v === undefined || v === "dir") throw new Error("EISDIR");
      return v;
    },
    homedir: () => HOME,
    tmpdir: () => TMP,
  };
}

const repo = (root) => io({ [path.join(root, ".git")]: "dir" });

test("deriveWorkspacePeerId keeps Claude project directory naming", () => {
  assert.equal(deriveWorkspacePeerId("/Users/x/Dev/OpenViking"), "-Users-x-Dev-OpenViking");
  assert.equal(deriveWorkspacePeerId("abc.DEF_123@x-y"), "abc-DEF-123-x-y");
  assert.equal(deriveWorkspacePeerId(""), "");
  assert.equal(deriveWorkspacePeerId(null), "");
});

test("a real project keeps the peer id it already had", () => {
  const root = path.resolve("/d/projects/OpenViking");
  assert.deepEqual(
    resolveEffectivePeerId({ cfg: {}, cwd: root, io: repo(root) }),
    { peerId: deriveWorkspacePeerId(root), source: "workspace" },
  );
});

test("a subdirectory maps to the project root, not its own peer", () => {
  const root = path.resolve("/d/projects/OpenViking");
  const deep = path.join(root, "openviking", "session", "memory");
  assert.equal(findProjectRoot(deep, repo(root)), root);
  assert.equal(
    resolveEffectivePeerId({ cfg: {}, cwd: deep, io: repo(root) }).peerId,
    deriveWorkspacePeerId(root),
  );
});

test("a git worktree resolves to its main repository", () => {
  const main = path.resolve("/d/projects/OpenViking");
  const wt = path.join(main, ".claude", "worktrees", "feature-x");
  const gitdir = path.join(main, ".git", "worktrees", "feature-x");
  assert.equal(
    findProjectRoot(wt, io({ [path.join(wt, ".git")]: `gitdir: ${gitdir}\n` })),
    main,
  );
});

test("filesystem and drive roots never become a peer", () => {
  const root = path.parse(path.resolve("/")).root;
  assert.equal(findProjectRoot(root, io()), "");
  assert.deepEqual(
    resolveEffectivePeerId({ cfg: {}, cwd: root, io: io() }),
    { peerId: "", source: "none" },
  );
});

test("a per-session scratch directory does not mint a peer", () => {
  // The regression this guards: every Codex session ran from its own title
  // directory, so each run created a brand new peer identity.
  const titleDir = path.join(HOME, "Documents", "Codex", "2026-07-27-perform-an-audit-of");
  assert.equal(findProjectRoot(titleDir, io()), "");
  assert.deepEqual(
    resolveEffectivePeerId({ cfg: {}, cwd: titleDir, io: io() }),
    { peerId: "", source: "none" },
  );
});

test("the bare home directory is not a project even when it is a repo", () => {
  assert.equal(findProjectRoot(HOME, repo(HOME)), "");
});

test("the temp root is not a project", () => {
  assert.equal(findProjectRoot(TMP, io()), "");
  assert.equal(findProjectRoot(path.join(TMP, "crash-7232026"), io()), "");
});

test("non-git project markers still count", () => {
  const root = path.resolve("/srv/app");
  for (const marker of ["package.json", "pyproject.toml", "Cargo.toml", "go.mod"]) {
    assert.equal(
      findProjectRoot(path.join(root, "src"), io({ [path.join(root, marker)]: "x" })),
      root,
      marker,
    );
  }
});

test("explicit peer still wins over workspace", () => {
  const root = path.resolve("/d/projects/OpenViking");
  assert.deepEqual(
    resolveEffectivePeerId({ cfg: { peerId: " configured " }, cwd: root, io: repo(root) }),
    { peerId: "configured", source: "explicit" },
  );
});

test("workspace peer can still be disabled", () => {
  const root = path.resolve("/d/projects/OpenViking");
  assert.deepEqual(
    resolveEffectivePeerId({ cfg: { workspacePeer: false }, cwd: root, io: repo(root) }),
    { peerId: "", source: "none" },
  );
  assert.deepEqual(
    resolveEffectivePeerId({ cfg: {}, cwd: "", io: io() }),
    { peerId: "", source: "none" },
  );
});
