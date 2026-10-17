# CPU deployment smoke measurements — 2026-10-17

Target: one shared Ryzen 9 9950X vCPU, 1963 MiB RAM, no GPU. Tests used a 0.8 CPU
quota, 1100 MiB container RAM limit and one inference thread. Application deployment
uses a 1400 MiB ceiling to leave room for the bot. No model process remains after a job.

| Test | End-to-end seconds | Child peak RSS, MiB |
|---|---:|---:|
| Whisper Turbo Q5, portable CPU build, forced RU | 76.45 | 730.5 |
| Whisper Turbo Q5, target-native build, forced RU | 68.03 | 730.3 |
| Bonsai 2 2B PQ2_0, short Russian summary | 12.07 | 1061.2 |

Whisper input: 10.64-second synthetic Russian speech with Telegram, VPS, Docker,
SQLite, API, GitHub Actions and “не удаляй базу данных”. Both runs preserved the
terms and negation. Bonsai returned five relevant points and preserved the negation;
wording was sometimes awkward. These examples do not establish general accuracy.

These are single cold-process smoke runs, not a statistically controlled benchmark.
Host CPU steal/load and file cache vary. Whisper uses up to 30-second windows, so
short-clip RTF should not be linearly extrapolated to long recordings. Quantized
weight size is not runtime peak RSS. This configuration favors low idle memory and
rare requests; it does not provide fast interactive Whisper on one vCPU.

Runtime versions/checksums are in Dockerfile and scripts/models.json. Reproduce with
`python -m scripts.smoke stt --audio FILE --language ru` or `python -m scripts.smoke llm`
inside the built CPU image, mounting scripts and models. For actual application
verification also test RU/EN auto-detection and the Telegram reply/button flow.
External API providers need separate credentials/live tests; local contract tests
alone do not establish compatibility with an arbitrary provider.
