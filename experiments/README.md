# Verified CPU smoke runs

`smoke-results.json` records the complete configuration, corpus hash, parameter counts, software version, and loss history from three actual 100-step runs. Commands are documented in the main README. Checkpoints remain in ignored `runs/` directories on the development machine.

| Run | Parameters | Initial validation NLL | Final validation NLL |
| --- | ---: | ---: | ---: |
| tiny (character) | 27,840 | 3.7537 | 2.8939 |
| wide (character) | 104,832 | 3.7055 | 2.6049 |
| byte | 34,688 | 5.5626 | 3.2601 |

These are single-seed functional checks on synthetic prose, not a benchmark or evidence of general language ability. Only the character runs share the same vocabulary. The wide model uses more compute per step. Loss decreases show that the training path works; scientific conclusions require larger independent datasets, repeated seeds, and controlled budgets.
