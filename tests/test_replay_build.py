"""Replay output preflight and input replacement with mocked generation."""

import json
import pathlib
import tempfile
import types
import unittest
from unittest.mock import patch

from trlx import TrlxError, replay_build


class ReplayOutput(unittest.TestCase):
    # Real prompt reading and output publication surround a deterministic sampler.
    def setUp(self):
        self.directory = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent)))
        self.prompts = self.directory / "prompts.jsonl"
        self.prompts.write_text('{"prompt":"question"}\n')
        self.args = types.SimpleNamespace(
            prompts=str(self.prompts), out=str(self.prompts), model="base", endpoint=None,
            max_tokens=4, timeout=None, retries=None, concurrency=None, api_key=None,
            force=False, no_staging=False,
        )
        self.sample = self.enterContext(patch("trlx.replay_build._from_local", return_value=["answer"]))

    # Existing output must be refused before local or paid endpoint generation.
    def test_existing_output_refuses_before_generation(self):
        with self.assertRaisesRegex(TrlxError, "--force"):
            replay_build.run(self.args)
        self.sample.assert_not_called()
        self.assertEqual(json.loads(self.prompts.read_text()), {"prompt": "question"})

    # Input replacement retains all prompt content in the completed replay rows.
    def test_force_replaces_prompts_after_reading(self):
        self.args.force = True
        replay_build.run(self.args)
        self.assertEqual(self.sample.call_args.args[1], ["question"])
        self.assertEqual(json.loads(self.prompts.read_text())["messages"], [
            {"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"},
        ])

    # Direct publication changes failure preservation, not replacement authorization.
    def test_direct_mode_requires_force(self):
        self.args.no_staging = True
        with self.assertRaisesRegex(TrlxError, "--force"):
            replay_build.run(self.args)
        self.args.force = True
        replay_build.run(self.args)
        self.assertEqual(json.loads(self.prompts.read_text())["messages"][-1]["content"], "answer")

    # An output alias is replaced without modifying its former input target.
    def test_symlink_output_preserves_prompts(self):
        output = self.directory / "replay.jsonl"
        output.symlink_to(self.prompts)
        self.args.out, self.args.force = str(output), True
        replay_build.run(self.args)
        self.assertFalse(output.is_symlink())
        self.assertEqual(json.loads(self.prompts.read_text()), {"prompt": "question"})

    # Status output uses the endpoint's sanitized address, never userinfo or query values.
    def test_endpoint_status_omits_url_credentials(self):
        self.args.endpoint = "https://user:secret@example.test/v1?token=private"
        self.args.timeout, self.args.retries, self.args.concurrency = 1, 0, 1
        with patch("trlx.replay_build.Endpoint.complete_many", return_value=["answer"]), patch("builtins.print") as output:
            replay_build._from_endpoint(self.args, ["question"])
        line = output.call_args.args[0]
        self.assertIn("example.test/v1/chat/completions", line)
        for secret in ("user", "secret", "private", "token="):
            self.assertNotIn(secret, line)
