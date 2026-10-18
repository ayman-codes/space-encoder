/**
 * Vendors OpenCode skills from public GitHub repositories.
 *
 * Each entry maps a skill id to a repository and the directory that contains
 * its `SKILL.md`. Every file below that directory is downloaded so supporting
 * references, scripts, and templates stay available to the skill.
 */

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const skills = [
  {
    name: "find-skills",
    repo: "vercel-labs/skills",
    dir: "skills/find-skills",
  },
  {
    name: "web-search-deepresearch",
    repo: "zapabob/skills",
    dir: "registry/skills/web-search-deepresearch/variants/cursor-staging",
  },
  {
    name: "auto-deep-research-guide",
    repo: "wentorai/research-plugins",
    dir: "skills/research/deep-research/auto-deep-research-guide",
  },
  {
    name: "browser-use",
    repo: "browser-use/browser-use",
    dir: "skills/browser-use",
  },
  {
    name: "agent-browser",
    repo: "vercel-labs/agent-browser",
    dir: "skills/agent-browser",
  },
  {
    name: "tavily-search",
    repo: "tavily-ai/skills",
    dir: "skills/tavily-search",
  },
];

const MAX_FILE_BYTES = 512 * 1024;
const OUT_ROOT = path.resolve(".opencode", "skills");
const HEADERS = {
  "User-Agent": "eos-skill-vendor",
  Accept: "application/vnd.github+json",
};

async function listBlobs(repo, dir) {
  const response = await fetch(
    `https://api.github.com/repos/${repo}/git/trees/HEAD?recursive=1`,
    { headers: HEADERS },
  );
  if (!response.ok) {
    throw new Error(`${repo}: tree request failed with HTTP ${response.status}`);
  }
  const payload = await response.json();
  return (payload.tree ?? []).filter(
    (entry) =>
      entry.type === "blob" &&
      entry.path.startsWith(`${dir}/`) &&
      (entry.size ?? 0) <= MAX_FILE_BYTES,
  );
}

async function download(repo, blobPath) {
  const response = await fetch(
    `https://raw.githubusercontent.com/${repo}/HEAD/${blobPath}`,
    { headers: HEADERS },
  );
  if (!response.ok) {
    throw new Error(`${repo}/${blobPath}: HTTP ${response.status}`);
  }
  return Buffer.from(await response.arrayBuffer());
}

let totalFiles = 0;
for (const skill of skills) {
  const blobs = await listBlobs(skill.repo, skill.dir);
  if (blobs.length === 0) {
    throw new Error(`${skill.name}: no files found under ${skill.dir}`);
  }
  for (const blob of blobs) {
    const relative = blob.path.slice(skill.dir.length + 1);
    const target = path.join(OUT_ROOT, skill.name, ...relative.split("/"));
    const body = await download(skill.repo, blob.path);
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, body);
    totalFiles += 1;
  }
  process.stdout.write(`${skill.name}: ${blobs.length} files\n`);
}
process.stdout.write(`vendored ${totalFiles} files into ${OUT_ROOT}\n`);
