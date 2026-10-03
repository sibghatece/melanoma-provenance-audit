# Results

This folder holds the files produced by our runs. Fill it before the first
commit with

    export MELANOMA_ROOT=/path/to/data
    python3 tools/collect_results.py

which copies tables, logs, prediction files, figures and the manifests,
removes local paths, and writes environment.txt. Images, model weights and
cached features are never copied.
