"""One-time move of forecasting outputs into the layout nested by imputation run id.

    python -m uncertainty_aware_planner.pipeline.migrate_layout --imputation-run-id imp-none-censored \
        [--root data] [--dry-run]

pipeline/forecast.py used to write <base>/<run_id>/ and now writes <base>/<imputation_run_id>/<run_id>/. Under
--root, every directory directly in one of BASES (missing bases are skipped) whose name starts with "fc-" is an
old-layout run and moves to <base>/<imputation-run-id>/<name>. Other children, such as already nested imp-*
directories, stay, so a second run finds nothing to move. Nothing moves if any destination exists or if a run
log names a different imputation run.

The run_log.json files in the moved directories hold paths into the old layout, often absolute and from
another machine. Every string in them is rewritten by segment, whatever comes before it:
    .../<base>/fc-<...>/...  ->  .../<base>/<imputation-run-id>/fc-<...>/...
Each base with moves gets MIGRATED.json, a list with one entry per migration (appended to).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from uncertainty_aware_planner.data.config import resolve

BASES = (
    "processed/forecast",
    "processed/forecast_paper_split",
    "processed/forecast_smoke",
    "runs/forecasting",
    "runs/forecasting_paper_split",
    "runs/forecasting_smoke",
)
RUN_PREFIX = "fc-"
RECORD = "MIGRATED.json"
# <base>/fc-<...> as a whole path segment: at the start of the string or after "/", then "/" or the end.
_OLD_SEGMENT = re.compile(
    r"(^|/)(" + "|".join(map(re.escape, BASES)) + r")/(fc-[^/]+)(?=/|$)"
)


def rewrite_paths(value, imputation_run_id: str):
    """(value with every old-layout path segment nested under imputation_run_id, strings changed).

    Strings are rewritten anywhere in nested dicts and lists; other values are returned unchanged.
    """
    if isinstance(value, str):
        new = _OLD_SEGMENT.sub(
            lambda m: f"{m[1]}{m[2]}/{imputation_run_id}/{m[3]}", value
        )
        return new, int(new != value)
    if isinstance(value, dict):
        items = {k: rewrite_paths(v, imputation_run_id) for k, v in value.items()}
        return {k: v for k, (v, _) in items.items()}, sum(n for _, n in items.values())
    if isinstance(value, list):
        items = [rewrite_paths(v, imputation_run_id) for v in value]
        return [v for v, _ in items], sum(n for _, n in items)
    return value, 0


def _check_run_id(imputation_run_id: str) -> None:
    if not re.fullmatch(r"imp-[A-Za-z0-9._-]+", imputation_run_id):
        raise ValueError(
            f"{imputation_run_id!r} is not an imputation run id (imp-<architecture>-<variant>[-<seed>], "
            "CONTRACTS.md identifiers)"
        )


def plan(root: Path, imputation_run_id: str) -> list[dict]:
    """Every old-layout run directory under root, with its run logs rewritten in memory.

    Raises FileExistsError if a destination exists and ValueError if a run log names another imputation run;
    nothing on disk changes.
    """
    _check_run_id(imputation_run_id)
    moves = []
    for base in BASES:
        directory = root / base
        if not directory.is_dir():
            continue
        for source in sorted(directory.iterdir()):
            if not (source.is_dir() and source.name.startswith(RUN_PREFIX)):
                continue
            logs = {}
            for path in sorted(source.rglob("run_log.json")):
                try:
                    data = json.loads(path.read_text())
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path} is not valid JSON: {error}") from error
                recorded = data.get("imputation_run_id")
                if recorded not in (None, imputation_run_id):
                    raise ValueError(
                        f"{path} records imputation run {recorded}, not {imputation_run_id}; nothing moved"
                    )
                logs[path.relative_to(source).as_posix()] = rewrite_paths(
                    data, imputation_run_id
                )
            moves.append(
                {
                    "base": base,
                    "source": source,
                    "destination": directory / imputation_run_id / source.name,
                    "logs": logs,
                }
            )
    clashes = [str(m["destination"]) for m in moves if m["destination"].exists()]
    if clashes:
        raise FileExistsError(
            f"destination exists, nothing moved: {', '.join(clashes)}"
        )
    return moves


def migrate(root: Path, imputation_run_id: str, dry_run: bool = False) -> dict:
    """Move the old-layout run directories under root; return the MIGRATED.json entry per base."""
    root = Path(root)
    moves = plan(root, imputation_run_id)
    if not moves:
        print(f"nothing to migrate under {root}")
        return {}
    if dry_run:
        print("dry run: nothing is moved or written")
    now = datetime.now(UTC).isoformat(timespec="seconds")
    entries = {}
    for move in moves:
        name = move["source"].name
        nested = f"{imputation_run_id}/{name}"
        entry = entries.setdefault(
            move["base"],
            {
                "migrated_at_utc": now,
                "imputation_run_id": imputation_run_id,
                "moved": [],
                "rewritten_run_logs": {},
            },
        )
        print(f"{move['base']}: {name} -> {nested}")
        # Logs are rewritten before the move: if it is interrupted, the next run redoes both steps.
        for relative, (data, changed) in move["logs"].items():
            if not changed:
                continue
            print(f"  {changed} paths in {name}/{relative}")
            entry["rewritten_run_logs"][f"{nested}/{relative}"] = changed
            if not dry_run:
                (move["source"] / relative).write_text(
                    json.dumps(data, indent=2) + "\n"
                )
        if not dry_run:
            move["destination"].parent.mkdir(exist_ok=True)
            move["source"].rename(move["destination"])
        entry["moved"].append({"from": name, "to": nested})

    if not dry_run:
        for base, entry in entries.items():
            path = root / base / RECORD
            history = json.loads(path.read_text()) if path.exists() else []
            history.append(entry)
            path.write_text(json.dumps(history, indent=2) + "\n")
            print(f"wrote {path}")
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--imputation-run-id",
        required=True,
        help="the imputation run all old-layout outputs belong to, e.g. imp-none-censored",
    )
    parser.add_argument(
        "--root", default="data", help="directory holding processed/ and runs/"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan; move and write nothing"
    )
    args = parser.parse_args(argv)
    try:
        migrate(resolve(args.root), args.imputation_run_id, dry_run=args.dry_run)
    except (FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
