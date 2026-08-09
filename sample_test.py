"""Sample tests to verify CI-related changes work locally.

Run:
    python sample_test.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VALIDATOR_PATH = REPO_ROOT / ".github" / "scripts" / "validate_pr_description.py"


def load_validator_module():
    spec = importlib.util.spec_from_file_location(
        "validate_pr_description", VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load validator from {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load_validator_module()

VALID_PR_BODY = """\
## PR-Description

Added CI workflow and PR template validation.

## Test-Plan

Ran `python sample_test.py` locally and opened a test PR.
"""

TEMPLATE_ONLY_BODY = """\
## PR-Description

<!-- Describe what this PR changes and why -->

## Test-Plan

<!-- Describe how you tested these changes -->
"""


def run_validator(pr_body: str) -> int:
    env = os.environ.copy()
    env["PR_BODY"] = pr_body
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode


class TestCIFiles(unittest.TestCase):
    def test_workflow_file_exists(self) -> None:
        workflow = REPO_ROOT / ".github" / "workflows" / "ci.yaml"
        self.assertTrue(workflow.is_file(), "Missing .github/workflows/ci.yaml")

    def test_pr_template_exists(self) -> None:
        template = REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md"
        self.assertTrue(template.is_file(), "Missing .github/PULL_REQUEST_TEMPLATE.md")

    def test_validator_script_exists(self) -> None:
        self.assertTrue(
            VALIDATOR_PATH.is_file(),
            "Missing .github/scripts/validate_pr_description.py",
        )

    def test_pr_template_has_required_headings(self) -> None:
        template_text = (REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("## PR-Description", template_text)
        self.assertIn("## Test-Plan", template_text)


class TestPRDescriptionValidator(unittest.TestCase):
    def test_valid_pr_body_passes(self) -> None:
        self.assertIsNotNone(validator.section_content(VALID_PR_BODY, "PR-Description"))
        self.assertTrue(validator.meaningful("Added CI workflow."))
        self.assertEqual(run_validator(VALID_PR_BODY), 0)

    def test_empty_body_fails(self) -> None:
        self.assertEqual(run_validator(""), 1)

    def test_missing_sections_fails(self) -> None:
        body = "## PR-Description\n\nOnly one section filled in.\n"
        self.assertEqual(run_validator(body), 1)

    def test_template_comments_only_fails(self) -> None:
        self.assertEqual(run_validator(TEMPLATE_ONLY_BODY), 1)

    def test_html_comments_are_ignored_for_content_check(self) -> None:
        body = """\
## PR-Description

<!-- placeholder -->

Real description here.

## Test-Plan

Manual testing done.
"""
        self.assertEqual(run_validator(body), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
