# .skyignore Gotchas — SkyPilot Workdir Sync (2026-03-14)

## Issue 1: Unanchored patterns match too broadly

`data/` in `.skyignore` matches ANY directory named `data` at any path level,
including `src/minivess/data/` (a Python module). This caused:
```
ModuleNotFoundError: No module named 'minivess.data'
```

**Fix**: Use root-anchored patterns: `/data/` only matches the top-level directory.

## Issue 2: .skyignore REPLACES .gitignore

When `.skyignore` exists, SkyPilot uses ONLY `.skyignore` for filtering.
`.gitignore` rules are NOT applied. This means you must explicitly exclude
everything that `.gitignore` would have excluded (`.venv/`, `__pycache__/`, etc.).

## Issue 3: Hidden large directories

Our initial `.skyignore` missed:
- `.dvc/cache/` — 2.6 GB (DVC cache of data files)
- `.git/` — 973 MB
- `checkpoints/` — 7.8 GB (model checkpoints)
- `dataset_local/` — 940 MB (zip downloads)
- `deployment/pulumi/.venv/` — 16 MB

Total sync was 12 GB instead of 45 MB. Always run:
```bash
find . -not -path './excluded/*' -type f -print0 | du -ch --files0-from=- | tail -1
```

## Rule

When creating `.skyignore`, always:
1. Use `/` prefix for root-level exclusions to avoid matching nested directories
2. Check total sync size with `du` before first launch
3. Exclude ALL build artifacts: `.dvc/`, `.git/`, `checkpoints/`, `*.pth`
