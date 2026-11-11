# Task1 DAgger on machine2

Remote: /mnt/file4/junyao/task1_dagger_compare_20261111.
Environment: project-local .venv, Python 3.12, torch 2.7.0 CUDA12.8.
Spec: tools/task1_contact_recovery/env-spec-3090.json. No system environment changes.
GPUs: official 3,4; current 5,6. DDP global batch32, local batch16.
Status: environment installed; input transfer complete. numba0.61.2 and llvmlite0.44.0 installed; pip check and DDP/Dataset/Policy imports pass. GPU update witness pending free GPUs.
Fresh documented witness launched; two updates, EMA, checkpoint reload and DDPM100 per arm pending.
Invocation: tools/task1_contact_recovery/DUAL.md.
Training requires accepted teacher corrections for both variants.
