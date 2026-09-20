"""Filesystem replacement contracts, using only repository-local scratch artifacts."""

import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from dataset import io


class OutputPublication(unittest.TestCase):
    # Each case owns its scratch tree; real operator files are never destinations.
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent)
        self.addCleanup(self.scratch.cleanup)
        self.root = pathlib.Path(self.scratch.name)

    # Prepare a small existing output for refusal and preservation assertions.
    def existing(self, name="data.jsonl"):
        path = self.root / name
        path.write_text('{"old": true}\n', encoding="utf-8")
        return path

    # Every supported format must use the destination extension, not the staging name.
    def test_format_round_trips(self):
        for suffix in io.FORMATS:
            with self.subTest(suffix=suffix):
                path = self.root / ("output" + suffix)
                io.write_rows(path, [{"value": "hello"}])
                self.assertEqual(io.read_rows(path), [{"value": "hello"}])

    # Direct writing alone is never replacement authorization.
    def test_existing_output_requires_force_in_both_modes(self):
        path = self.existing()
        old = path.read_bytes()
        for direct in (False, True):
            with self.subTest(direct=direct), self.assertRaisesRegex(io.DatasetError, "Use --force"):
                io.write_rows(path, [{"new": True}], no_staging=direct)
            self.assertEqual(path.read_bytes(), old)

    # Inputs are consumed before publication and may then be replaced intentionally.
    def test_force_replaces_input_in_both_modes(self):
        for direct in (False, True):
            path = self.existing(str(direct) + ".jsonl")
            rows = io.read_rows(path)
            io.write_rows(path, rows + [{"new": True}], [path], force=True, no_staging=direct)
            self.assertEqual(io.read_rows(path), [{"old": True}, {"new": True}])

    # Failed serialization must not truncate a staged destination, including CSV schema errors.
    def test_preparation_failure_preserves_original(self):
        for suffix, rows in ((".jsonl", [{"v": 1}, {"v": {1, 2}}]),
                             (".csv", [{"v": [1, 2]}]),
                             (".parquet", [{"v": 1}, {"v": {"nested": 1}}])):
            path = self.existing("broken" + suffix)
            old = path.read_bytes()
            with self.subTest(suffix=suffix), self.assertRaises(io.DatasetError):
                io.write_rows(path, rows, force=True)
            self.assertEqual(path.read_bytes(), old)
        self.assertFalse(list(self.root.glob(".*.stage-*")))

    # No-staging must allocate no staged output and must acknowledge the failed direct write.
    def test_direct_failure_reports_loss_without_staging(self):
        path = self.existing()
        with patch.object(io.tempfile, "mkdtemp", side_effect=AssertionError("staging used")):
            with self.assertRaisesRegex(io.DatasetError, "original may be lost"):
                io.write_rows(path, [{"new": True}, {"bad": {1, 2}}], force=True, no_staging=True)
        self.assertIn('"new": true', path.read_text())
        self.assertNotIn('"old"', path.read_text())

    # Both live and dangling output links are replaced without modifying their former targets.
    def test_symlink_replacement_in_both_modes(self):
        for direct in (False, True):
            for dangling in (False, True):
                with self.subTest(direct=direct, dangling=dangling):
                    target = self.root / f"target-{direct}-{dangling}.jsonl"
                    if not dangling:
                        target.write_text("original")
                    link = self.root / f"link-{direct}-{dangling}.jsonl"
                    link.symlink_to(target)
                    with self.assertRaisesRegex(io.DatasetError, "symlink itself"):
                        io.write_rows(link, [], no_staging=direct)
                    io.write_rows(link, [{"new": 1}], force=True, no_staging=direct)
                    self.assertFalse(link.is_symlink())
                    self.assertEqual(io.read_rows(link), [{"new": 1}])
                    self.assertEqual(target.read_text() if target.exists() else None,
                                     None if dangling else "original")

    # Replacing one hardlink must leave other names for the original inode unchanged.
    def test_hardlink_replacement_preserves_other_name(self):
        for direct in (False, True):
            target = self.existing(f"original-{direct}.jsonl")
            alias = self.root / f"alias-{direct}.jsonl"
            os.link(target, alias)
            io.write_rows(alias, [{"new": True}], force=True, no_staging=direct)
            self.assertEqual(io.read_rows(target), [{"old": True}])
            self.assertFalse(target.samefile(alias))

    # Healing's explicit exception retains the link while publishing to its resolved target.
    def test_follow_symlink_repairs_target(self):
        for direct in (False, True):
            target = self.existing(f"target-{direct}.jsonl")
            link = self.root / f"repair-{direct}.jsonl"
            link.symlink_to(target)
            io.write_text(link, "repaired", force=True, no_staging=direct, follow_symlinks=True)
            self.assertTrue(link.is_symlink())
            self.assertEqual(target.read_text(), "repaired")

    # File/directory type changes are authorized replacement, not a reason to ignore force.
    def test_file_replaces_directory_and_its_unrelated_contents(self):
        for direct in (False, True):
            path = self.root / f"directory-{direct}.jsonl"
            path.mkdir()
            (path / "unrelated").write_text("old")
            io.write_rows(path, [{"new": True}], force=True, no_staging=direct)
            self.assertEqual(io.read_rows(path), [{"new": True}])

    # A merged directory may replace a file, directory, or link without following links.
    def test_directory_replacement(self):
        for direct in (False, True):
            for kind in ("file", "directory", "symlink"):
                target = self.root / f"output-{direct}-{kind}"
                foreign = self.root / f"foreign-{direct}-{kind}"
                foreign.mkdir()
                (foreign / "keep").write_text("retained")
                if kind == "directory":
                    target.mkdir()
                    (target / "old").write_text("old")
                elif kind == "symlink":
                    target.symlink_to(foreign, target_is_directory=True)
                else:
                    target.write_text("old")
                io.publish_output(target, lambda p: (p / "new").write_text("new"),
                                  force=True, no_staging=direct, directory=True)
                self.assertEqual([p.name for p in target.iterdir()], ["new"])
                self.assertFalse(target.is_symlink())
                self.assertEqual((foreign / "keep").read_text(), "retained")

    # All inputs inside the replacement tree remain readable during staged production.
    def test_directory_preparation_before_ancestor_replacement(self):
        target = self.root / "models"
        target.mkdir()
        (target / "base").write_text("weights")

        # The staged writer can finish source reads before publication deletes the tree.
        def produce(prepared):
            self.assertNotIn(target, prepared.parents)
            (prepared / "merged").write_text((target / "base").read_text())

        io.publish_output(target, produce, force=True, directory=True)
        self.assertEqual([p.name for p in target.iterdir()], ["merged"])

    # Permissions belong to the regular destination, independent of temporary-file defaults.
    def test_regular_file_permissions_survive(self):
        path = self.existing()
        path.chmod(0o640)
        for direct in (False, True):
            io.write_text(path, "new", force=True, no_staging=direct)
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    # A competing creator after preparation must not be overwritten without force.
    def test_creation_race_is_refused(self):
        target = self.root / "new.txt"

        # Simulate a second process creating the output while our data is prepared.
        def produce(prepared):
            prepared.write_text("ours")
            target.write_text("theirs")

        with self.assertRaisesRegex(io.DatasetError, "Use --force"):
            io.publish_output(target, produce)
        self.assertEqual(target.read_text(), "theirs")

    # A failed second serialization must leave both original split outputs untouched.
    def test_split_prepares_both_before_publication(self):
        first, second = self.existing("first.jsonl"), self.existing("second.jsonl")
        with self.assertRaisesRegex(io.DatasetError, "no destination completed"):
            io.write_many_rows([(first, [{"new": 1}]), (second, [{"bad": {1, 2}}])], force=True)
        self.assertEqual(io.read_rows(first), [{"old": True}])
        self.assertEqual(io.read_rows(second), [{"old": True}])

    # Publication is sequential, so the error must name the output already replaced.
    def test_split_reports_partial_publication(self):
        first, second = self.existing("first.jsonl"), self.existing("second.jsonl")
        actual = pathlib.Path.replace

        # Fail only the second output publication, after the first one succeeded.
        def replace(source, destination):
            if destination == second:
                raise OSError("injected second publication failure")
            return actual(source, destination)

        with patch.object(pathlib.Path, "replace", replace), self.assertRaisesRegex(io.DatasetError, "completed destinations:.*first.jsonl"):
            io.write_many_rows([(first, [{"new": 1}]), (second, [{"new": 2}])], force=True)
        self.assertEqual(io.read_rows(first), [{"new": 1}])
        self.assertEqual(io.read_rows(second), [{"old": True}])

    # Failed directory installation restores the original directory without stale new files.
    def test_directory_publication_failure_restores_original(self):
        target = self.root / "model"
        target.mkdir()
        (target / "old").write_text("old")
        actual = pathlib.Path.replace

        # Fail the install rename but allow the restoration rename to succeed.
        def replace(source, destination):
            if source.name == "result":
                raise OSError("injected installation failure")
            return actual(source, destination)

        with patch.object(pathlib.Path, "replace", replace), self.assertRaisesRegex(io.DatasetError, "original output restored"):
            io.publish_output(target, lambda p: (p / "new").write_text("new"), force=True, directory=True)
        self.assertEqual([p.name for p in target.iterdir()], ["old"])
        self.assertFalse(list(self.root.glob(".*.stage-*")))

    # When restoration also fails, retain both recoverable copies and identify their paths.
    def test_directory_failed_restore_retains_recovery_copies(self):
        target = self.root / "model"
        target.mkdir()
        (target / "old").write_text("old")
        with patch.object(pathlib.Path, "replace", side_effect=OSError("rename unavailable")):
            with self.assertRaisesRegex(io.DatasetError, "Original output is at.*previous; prepared output is at"):
                io.publish_output(target, lambda p: (p / "new").write_text("new"), force=True, directory=True)
        folder, = self.root.glob(".*.stage-*")
        self.assertEqual((folder / "previous" / "old").read_text(), "old")
        self.assertEqual((folder / "result" / "new").read_text(), "new")

    # Programmer exceptions remain distinct and staged scratch is still cleaned.
    def test_producer_bug_is_not_disguised(self):
        path = self.existing()
        with self.assertRaisesRegex(AssertionError, "producer bug"):
            io.publish_output(path, lambda p: self.fail("producer bug"), force=True)
        self.assertEqual(io.read_rows(path), [{"old": True}])
        self.assertFalse(list(self.root.glob(".*.stage-*")))

    # Syntax guidance must give a usable shell command even when the source contains spaces.
    def test_json_error_has_quoted_heal_guidance(self):
        path = self.root / "broken input.jsonl"
        path.write_text("{bad}\n")
        with self.assertRaisesRegex(io.DatasetError, "invalid JSON.*dataset heal '.*broken input.jsonl'"):
            io.read_rows(path)

    # Parent '..' is evaluated after symlink traversal, exactly as ordinary filesystem access.
    def test_parent_symlink_dotdot_selects_correct_destination(self):
        nested = self.root / "other" / "nested"
        nested.mkdir(parents=True)
        link = self.root / "link"
        link.symlink_to(nested, target_is_directory=True)
        unintended = self.existing("data.jsonl")
        intended = self.root / "other" / "data.jsonl"
        intended.write_text("intended old")
        io.write_text(link / ".." / "data.jsonl", "new", force=True)
        self.assertEqual(intended.read_text(), "new")
        self.assertEqual(io.read_rows(unintended), [{"old": True}])

    # Reapply read-only permission bits after writing; replacement needs writable parent only.
    def test_direct_replacement_of_read_only_file(self):
        path = self.existing()
        path.chmod(0o444)

        # Check mode explicitly so this regression also works when tests run as root.
        def write(destination):
            self.assertTrue(destination.stat().st_mode & 0o200)
            destination.write_text("new")

        io.publish_output(path, write, force=True, no_staging=True)
        self.assertEqual(path.read_text(), "new")
        self.assertEqual(path.stat().st_mode & 0o777, 0o444)

    # The final race window must not move a late directory into a disposable backup.
    def test_directory_creation_race_after_final_validation(self):
        actual = io.validate_output
        for directory in (False, True):
            target = self.root / f"race-{directory}"
            calls = 0

            # Create the competing directory after the second validation has returned.
            def validate(*args, **kwargs):
                nonlocal calls
                result = actual(*args, **kwargs)
                calls += 1
                if calls == 2:
                    target.mkdir()
                    (target / "theirs").write_text("retained")
                return result

            # Produce either kind of output without using any existing destination.
            def produce(prepared):
                (prepared / "ours" if directory else prepared).write_text("ours")

            with patch.object(io, "validate_output", validate), self.assertRaisesRegex(io.DatasetError, "Use --force"):
                io.publish_output(target, produce, directory=directory)
            self.assertEqual((target / "theirs").read_text(), "retained")

    # An old-copy cleanup error reports a published replacement, never claims installation failed.
    def test_old_copy_cleanup_failure_reports_published_output(self):
        target = self.root / "model"
        target.mkdir()
        (target / "old").write_text("old")
        with patch.object(io, "_remove_output", side_effect=OSError("cleanup denied")):
            with self.assertRaisesRegex(io.DatasetError, "new output published, but old-output cleanup failed") as caught:
                io.publish_output(target, lambda p: (p / "new").write_text("new"), force=True, directory=True)
        self.assertNotIn("could not be completed or published", str(caught.exception))
        self.assertEqual((target / "new").read_text(), "new")
        folder, = self.root.glob(".*.stage-*")
        self.assertEqual((folder / "previous" / "old").read_text(), "old")

    # A second failure during scratch cleanup cannot hide a partial split's primary failure.
    def test_split_cleanup_preserves_primary_failure_and_completed_paths(self):
        first, second = self.existing("first.jsonl"), self.existing("second.jsonl")
        actual = io._publish_prepared

        # Install first and deliberately fail before the second installation.
        def publish(target, *args):
            if target == second:
                raise OSError("second installation failed")
            return actual(target, *args)

        with patch.object(io, "_publish_prepared", publish), \
             patch.object(io, "_clean_stage", side_effect=OSError("cleanup denied")):
            with self.assertRaises(io.DatasetError) as caught:
                io.write_many_rows([(first, [{"new": 1}]), (second, [{"new": 2}])], force=True)
        message = str(caught.exception)
        self.assertIn("second installation failed", message)
        self.assertIn("completed destinations: " + str(first), message)
        self.assertIn("cleanup denied", message)

    # Alias-inspection failures are filesystem input errors, even before any serialization.
    def test_split_alias_inspection_failure_is_contextual(self):
        first, second = self.existing("first.jsonl"), self.existing("second.jsonl")
        with patch.object(pathlib.Path, "samefile", side_effect=OSError("stat denied")):
            with self.assertRaisesRegex(io.DatasetError, "stat denied.*no destination completed"):
                io.write_many_rows([(first, []), (second, [])], force=True)

    # Canonicalizing a final '..' keeps direct deletion from trying to remove that directory entry.
    def test_directory_dotdot_output_replaces_named_parent(self):
        target = self.root / "parent"
        child = target / "child"
        child.mkdir(parents=True)
        io.publish_output(child / "..", lambda p: (p / "new").write_text("new"),
                          force=True, no_staging=True, directory=True)
        self.assertEqual([p.name for p in target.iterdir()], ["new"])

    # Repair text can contain decoded escaped surrogate characters; report encoding failures.
    def test_text_encoding_failure_preserves_staged_original(self):
        path = self.existing()
        with self.assertRaisesRegex(io.DatasetError, "cannot encode output as UTF-8"):
            io.write_text(path, "\ud800", force=True)
        self.assertEqual(io.read_rows(path), [{"old": True}])
