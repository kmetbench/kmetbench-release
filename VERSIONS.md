# Version Tracker

Tracks the current public version and links for each artifact.

| Artifact | Version / Revision | URL | Updated |
| --- | --- | --- | --- |
| Paper | ACL Findings 2026 / OpenReview | https://openreview.net/forum?id=1Gn5pKek8k | 2026-04-16 |
| Dataset | v1.0.0 | https://huggingface.co/datasets/soyeonbot/K-MetBench | 2026-04-16 |
| Code | v1.0.0 | https://github.com/kmetbench/kmetbench-release | 2026-04-16 |
| Leaderboard | v1.0.0 | https://kmetbench.github.io/ | 2026-04-16 |

## Version Update Guide

Follow these steps when updating an artifact:

1. Update the `Version / Revision` and `Updated` columns in the table above.
2. Add a dated changelog entry to the `Updates` section in `README.md`.
3. For code changes, also bump the version in `pyproject.toml`.
4. Create a git tag for the public code snapshot when appropriate, for example `git tag v0.2.0`.
