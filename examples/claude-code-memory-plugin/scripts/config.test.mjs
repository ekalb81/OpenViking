import assert from "node:assert/strict";
import { join } from "node:path";
import { tmpdir } from "node:os";
import test from "node:test";

import { loadConfig } from "./config.mjs";

const ENV_KEYS = [
  "OPENVIKING_CLI_CONFIG_FILE",
  "OPENVIKING_CONFIG_FILE",
  "OPENVIKING_MEMORY_ENABLED",
  "OPENVIKING_RECALL_EXPERIENCES",
];

async function withIsolatedConfig(action) {
  const saved = Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]]));
  const missing = join(tmpdir(), `openviking-missing-config-${process.pid}.json`);
  process.env.OPENVIKING_CLI_CONFIG_FILE = missing;
  process.env.OPENVIKING_CONFIG_FILE = missing;
  process.env.OPENVIKING_MEMORY_ENABLED = "1";
  delete process.env.OPENVIKING_RECALL_EXPERIENCES;
  try {
    await action();
  } finally {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

test("Claude recall experience quota defaults to three and honors bounded overrides", async () => {
  await withIsolatedConfig(() => {
    assert.equal(loadConfig().recallExperiences, 3);

    process.env.OPENVIKING_RECALL_EXPERIENCES = "2.9";
    assert.equal(loadConfig().recallExperiences, 2);

    process.env.OPENVIKING_RECALL_EXPERIENCES = "0";
    assert.equal(loadConfig().recallExperiences, 0);

    process.env.OPENVIKING_RECALL_EXPERIENCES = "-4";
    assert.equal(loadConfig().recallExperiences, 0);
  });
});
