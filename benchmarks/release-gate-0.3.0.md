# Release gate — laya-apple 0.3.0

- generated: 2026-09-23T19:24:26+0800
- git revision: `3236789ac005adca8caf39240b995e962607db88` (dirty)
- python: 3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]
- platform: {"soc": "Apple M4 Max", "macos": "26.6.2", "macos_build": "25G83", "coremltools": "9.0"}
- quick mode: False
- soak seconds: 0
- **result: PASS**

| step | status | duration (s) | required |
|---|---|---|---|
| ruff check | pass | 0.08 | True |
| ruff format --check | pass | 0.03 | True |
| derive_routing --check | pass | 0.07 | True |
| make_goldens --check | pass | 0.05 | True |
| derive_placement --check | pass | 0.04 | True |
| pytest | pass | 208.48 | True |
| clean install matrix | pass | 31.75 | True |
| artifacts verify | pass | 14.67 | True |
| manifest schema | pass | 0.0 | True |
| no silent fallback tests | pass | 0.01 | True |
| docs versions | pass | 0.0 | True |
| soak | skipped | 0.0 | False |
