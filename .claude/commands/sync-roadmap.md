Sync the GitHub Project Roadmap timeline fields for issues.

Read the skill definition at `.claude/skills/sync-roadmap/SKILL.md` for field IDs and rules.

Based on the user's request, run one of:

1. **Single issue** (if the user mentions a specific issue number):
   ```
   python3 scripts/sync_roadmap.py --issue NUM
   ```

2. **Recently closed** (default when user says "sync roadmap" without specifics):
   ```
   python3 scripts/sync_roadmap.py --mode recent --days 7
   ```

3. **Full backfill** (if the user says "backfill" or "sync all"):
   ```
   python3 scripts/sync_roadmap.py --mode backfill
   ```

After running, report what was updated. If the user just closed an issue and wants it on the roadmap, use single-issue mode.

If an issue is not yet in the project, the script will add it automatically.

$ARGUMENTS
