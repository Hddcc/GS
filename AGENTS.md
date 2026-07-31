# Repository Working Rules

1. Every code, configuration, dataset-protocol, training, evaluation, result, innovation-direction, or conclusion change must update `项目进展记录.md` in the same working session and commit.
2. Do not present a probe or admission-test result as a formal paper result. Record the research status explicitly.
3. Keep `main` as the stable reproducible branch. Develop each new idea in a dedicated `experiment/<topic>` branch and merge only after the documented admission criteria pass.
4. Never commit datasets, checkpoints, generated results, credentials, caches, or server-transfer archives. Preserve them outside Git and record their paths and checksums in Markdown.
5. Do not delete user-owned data or experiment artifacts. Only generated temporary files may be removed after their exact paths are verified.

