"""Validate that a PR body contains required PR-Description and Test-Plan sections."""

import os
import re
import sys

TEMPLATE = """\
PR description must use this template:

## PR-Description
<describe your changes>

## Test-Plan
<describe how you tested>
"""


def meaningful(text: str) -> bool:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return bool(text.strip())


def section_content(body: str, name: str) -> str | None:
    match = re.search(
        rf"^##\s*{re.escape(name)}\s*\r?\n(.*?)(?=^##\s|\Z)",
        body,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    return match.group(1) if match else None


def main() -> int:
    body = os.environ.get("PR_BODY") or ""
    errors: list[str] = []

    pr_description = section_content(body, "PR-Description")
    test_plan_match = re.search(
        r"^##\s*Test-Plan\s*\r?\n(.*)",
        body,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    test_plan = test_plan_match.group(1) if test_plan_match else None

    if pr_description is None:
        errors.append("Missing required section: ## PR-Description")
    elif not meaningful(pr_description):
        errors.append("## PR-Description section is empty")

    if test_plan is None:
        errors.append("Missing required section: ## Test-Plan")
    elif not meaningful(test_plan):
        errors.append("## Test-Plan section is empty")

    if errors:
        print(TEMPLATE)
        print()
        for error in errors:
            print(f"- {error}")
        return 1

    print("PR description template check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
