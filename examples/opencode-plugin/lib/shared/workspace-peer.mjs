// GENERATED FROM examples/memory-plugin-shared/lib. DO NOT EDIT.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

/**
 * Files/directories that mark a real project root. `.git` is checked first so a
 * git worktree can be resolved back to its main repository.
 */
const PROJECT_MARKERS = [
  ".git",
  "package.json",
  "pyproject.toml",
  "Cargo.toml",
  "go.mod",
  "pom.xml",
  "build.gradle",
  "build.gradle.kts",
  ".hg",
  ".svn",
];

const DEFAULT_IO = {
  existsSync: (p) => fs.existsSync(p),
  statSync: (p) => fs.statSync(p),
  readFileSync: (p, enc) => fs.readFileSync(p, enc),
  homedir: () => os.homedir(),
  tmpdir: () => os.tmpdir(),
};

/**
 * Peer id for a workspace directory. Mirrors Claude's project-directory naming.
 * Deliberately unchanged: real projects keep the peer id they already have.
 */
export function deriveWorkspacePeerId(cwd) {
  return String(cwd || "").replace(/[^A-Za-z0-9]/g, "-");
}

function samePath(a, b) {
  if (!a || !b) return false;
  const norm = (v) => path.resolve(v).replace(/[\\/]+$/, "").toLowerCase();
  return norm(a) === norm(b);
}

/**
 * True for directories that must never become a peer on their own: filesystem and
 * drive roots, the bare home directory, and the OS temp root. A session started in
 * one of these is not "a project" -- deriving a peer from it is what produced ids
 * like "C--" and buried real memories under a junk identity.
 */
function isRootLike(dir, io) {
  if (!dir) return true;
  if (path.parse(dir).root === dir) return true;
  let home = "";
  let tmp = "";
  try {
    home = io.homedir();
  } catch {
    home = "";
  }
  try {
    tmp = io.tmpdir();
  } catch {
    tmp = "";
  }
  return samePath(dir, home) || samePath(dir, tmp);
}

/**
 * Given a directory containing `.git`, return the project root.
 *
 * In a linked worktree `.git` is a file holding `gitdir: <main>/.git/worktrees/<name>`,
 * so the worktree resolves to the main repository and shares its peer instead of
 * minting a separate identity per worktree.
 */
function resolveGitRoot(dir, io) {
  const dotGit = path.join(dir, ".git");
  let stat;
  try {
    stat = io.statSync(dotGit);
  } catch {
    return dir;
  }
  if (stat.isDirectory()) return dir;

  let text = "";
  try {
    text = String(io.readFileSync(dotGit, "utf-8"));
  } catch {
    return dir;
  }
  const match = /^gitdir:\s*(.+)$/m.exec(text);
  if (!match) return dir;

  const gitDir = match[1].trim().split("/").join(path.sep);
  const marker = `${path.sep}.git${path.sep}worktrees${path.sep}`;
  const idx = gitDir.toLowerCase().indexOf(marker.toLowerCase());
  if (idx > 0) return gitDir.slice(0, idx);
  return dir;
}

/**
 * Walk up from `cwd` to the nearest project root, or "" when there is none.
 *
 * Returning "" is the point: it is what stops a per-session scratch directory
 * (for example ~/Documents/Codex/<session-title>) from becoming a brand new peer
 * on every run.
 */
export function findProjectRoot(cwd, io = DEFAULT_IO) {
  const start = String(cwd || "").trim();
  if (!start) return "";

  let dir;
  try {
    dir = path.resolve(start);
  } catch {
    return "";
  }

  const seen = new Set();
  while (dir && !seen.has(dir)) {
    seen.add(dir);
    if (!isRootLike(dir, io)) {
      for (const marker of PROJECT_MARKERS) {
        let present = false;
        try {
          present = io.existsSync(path.join(dir, marker));
        } catch {
          present = false;
        }
        if (!present) continue;
        const root = marker === ".git" ? resolveGitRoot(dir, io) : dir;
        return isRootLike(root, io) ? "" : root;
      }
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return "";
}

export function resolveEffectivePeerId({ cfg = {}, cwd = "", io = DEFAULT_IO } = {}) {
  const explicit = String(cfg.peerId || "").trim();
  if (explicit) return { peerId: explicit, source: "explicit" };

  if (cfg.workspacePeer !== false) {
    const root = findProjectRoot(cwd, io);
    if (root) {
      const peerId = deriveWorkspacePeerId(root);
      if (peerId) return { peerId, source: "workspace" };
    }
  }

  return { peerId: "", source: "none" };
}
