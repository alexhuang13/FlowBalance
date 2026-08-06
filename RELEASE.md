# Release Checklist

Before publishing a FlowSD release:

1. Run `python3 scripts/check_environment.py` in the target training image.
2. Run the maintained tests listed in the root README.
3. Confirm that `runtime_env.yaml` contains only placeholder credentials.
4. Exclude model weights, checkpoints, W&B directories, generation logs, raw
   judge traces, scheduler manifests, and private filesystem paths.
5. Review dataset and model licenses separately from the source-code license.
6. Record the Git commit, dependency versions, and public experiment manifest.
7. Verify that the root README and maintained recipe documentation are English.

The `provenance/` and vendored `verl/` trees may retain upstream wording and are
not treated as first-party documentation.
