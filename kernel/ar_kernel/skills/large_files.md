---
name: large_files
description: Use when a file or command output is too large to read whole, for example transcripts, training logs, context.json, JSON or CSV annotations.
---

# Reading large files

- Size first: `wc -c -l file`, `ls -lh dir`.
- Parts: `head -n 50`, `tail -n 50`, `sed -n '200,260p' file`, or a file-reading tool's line range.
- Search: `grep -n "pattern" file`, `grep -c` to count, `grep -l "pattern" -r dir` to find files, `grep -A 5 -B 2` for context around a match.
- JSON: `python -c "import json; d = json.load(open('f.json')); print(list(d))"` to see the keys, then print one key.
- Tables (CSV, TSV, JSON lines): `pandas.read_csv` or `pandas.read_json(lines=True)`, then `df.shape`, `df.columns`, `df.head()`, `df.describe()`, `df.groupby(...)`.
- Logs with numbers: extract the fields you need with a regular expression in a short Python script and summarise them (mean, trend, groups) instead of reading the lines.
