# Security Policy

## Supported versions

`setqca` is pre-1.0. Only the most recent release receives fixes.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report them through one of the following channels:

1. **GitHub Security Advisories** (preferred) — open a report at
   [Security → Report a vulnerability](https://github.com/DiogoRibeiro7/setqca-python/security/advisories/new).
2. **Email** — <dfr@esmad.ipp.pt> with `[SECURITY]` in the subject line.

Please include:

- the type of issue and its impact;
- the affected source files and version or commit;
- step-by-step reproduction instructions;
- a proof of concept, if you have one.

### What to expect

| Stage                        | Target        |
| ---------------------------- | ------------- |
| Acknowledgement of the report| 48 hours      |
| Assessment and next steps    | 5 working days|
| Fix or status update         | 30 days       |

We will credit you for the discovery unless you ask to remain anonymous.

## Threat model

`setqca` is a scientific computation library. It performs no network access,
executes no user-supplied code, spawns no subprocesses, and writes no files. Its
inputs are numeric arrays and `pandas` DataFrames supplied by the calling
program.

**In scope:**

- arbitrary code execution reachable through the public API;
- vulnerabilities introduced by a direct dependency (`numpy`, `pandas`);
- supply-chain integrity of published artefacts on PyPI;
- unsafe deserialisation or resource-exhaustion bugs reachable from the API.

**Out of scope:**

- long runtimes on deliberately adversarial inputs. Exact Boolean minimisation
  is worst-case exponential in the number of conditions by construction; this is
  a documented mathematical property, not a defect. Use `max_solutions` and a
  sensible number of conditions to bound the work.
- Numerical disagreements with the reference R `QCA` implementation. These are
  correctness issues — please open a
  [parity report](https://github.com/DiogoRibeiro7/setqca-python/issues/new?template=parity_report.yml)
  instead.

## Release integrity

Releases are published to PyPI from a tagged commit through GitHub Actions using
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/). No
long-lived API token exists for this project, and no maintainer uploads
artefacts from a personal machine. Every published artefact is built from a
commit whose test, type and lint gates passed in CI.

## Automated scanning

- **Dependabot** — weekly dependency and monthly action updates.
- **pip-audit** — dependency vulnerability scanning on every push and weekly on
  a schedule.
- **CodeQL** — static analysis with the `security-and-quality` query suite.

## Contact

- Email: <dfr@esmad.ipp.pt>
- GitHub: [@DiogoRibeiro7](https://github.com/DiogoRibeiro7)
