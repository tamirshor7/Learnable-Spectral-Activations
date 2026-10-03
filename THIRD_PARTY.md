# Third-party provenance

Image upstream provenance is retained verbatim in `experiments/images/THIRD_PARTY.md`:

| Component | Recorded upstream commit | Bundled license text |
| --- | --- | --- |
| FINER/SIREN | liuzhen0212/FINER, 51a9254e371b1cbfacc408aee7a6d215d6fa0057 | No |
| STAF | AlirezaMorsali/STAF, 3303cf48c2501f56a4207960a88efe4d803b1f33 | MIT text retained under experiments/images/third_party/licenses |
| SL2A | moeinheidari7829/SL2A-INR, 3bee9c07890a1dade2207b76cb8cc623f6711784 | MIT text retained under experiments/images/third_party/licenses |

The audio archive includes FINER, SL2A-INR and fKAN source files without upstream commit/license records. Their exact uploaded bytes are tracked by `provenance/source_files.json`. Image commit IDs do not establish the provenance of the separately supplied audio copies. Confirm those copies against the upstream versions used by the authors and include the corresponding notices before public release.

The input archives provide no top-level project license or canonical citation metadata. These require an author decision. Dataset distribution terms remain those of Kodak, LibriSpeech and NSynth. This integration supplies code provenance and retains existing notices, and does not assign new licenses to inherited content.
