from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


BIN_PATTERN = re.compile(r"dag_roster_cpr_(\d+\.\d+_\d+\.\d+)\.csv$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter dag_catalog.csv to keep only DAGs that appear in selected CPR bins."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/results/ex05_pre/05_pre1"),
        help="ex05_pre run directory that contains rosters/.",
    )
    parser.add_argument(
        "--roster-dir",
        type=Path,
        default=None,
        help="Roster directory (default: <run-dir>/rosters).",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help=(
            "Catalog CSV path. If omitted, auto-selects the largest of "
            "dag_catalog.csv, dag_catalog_prev.csv, dag_catalog.csv.bak."
        ),
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="all",
        help=(
            "Comma-separated CPR bin labels (e.g., 0.1_0.2,0.2_0.3) or 'all'."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["union", "intersection"],
        default="union",
        help="How to combine bins: union (default) or intersection.",
    )
    parser.add_argument(
        "--limit-per-bin",
        type=int,
        default=0,
        help="Max DAGs to keep per bin (0 disables limit).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <roster-dir>/dag_catalog_filtered.csv).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        default=False,
        help="Overwrite catalog file (creates .bak once if missing).",
    )
    return parser


def parse_bins(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def count_catalog_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                count += 1
    return count


def select_catalog_path(
    roster_dir: Path, requested: Optional[Path]
) -> Tuple[Path, Path, dict[Path, int]]:
    primary = roster_dir / "dag_catalog.csv"
    if requested is not None:
        return requested, primary, {}

    candidates = [
        primary,
        roster_dir / "dag_catalog_prev.csv",
        roster_dir / "dag_catalog.csv.bak",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise SystemExit(f"Catalog not found under {roster_dir}")

    counts = {path: count_catalog_rows(path) for path in existing}
    selected = max(existing, key=lambda path: counts.get(path, 0))
    return selected, primary, counts


def discover_bins(roster_dir: Path) -> List[str]:
    labels: List[Tuple[float, str]] = []
    for path in roster_dir.glob("dag_roster_cpr_*.csv"):
        match = BIN_PATTERN.match(path.name)
        if not match:
            continue
        label = match.group(1)
        left = label.split("_")[0]
        try:
            left_val = float(left)
        except ValueError:
            left_val = 0.0
        labels.append((left_val, label))
    labels.sort()
    return [label for _, label in labels]


def collect_ids(
    paths: Iterable[Path], limit_per_bin: int, mode: str
) -> Tuple[Set[int], List[Tuple[str, int]]]:
    ids_by_bin: List[Set[int]] = []
    counts: List[Tuple[str, int]] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Roster not found: {path}")
        label = path.name.replace("dag_roster_cpr_", "").replace(".csv", "")
        count = 0
        ids: Set[int] = set()
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "dag_id" not in reader.fieldnames:
                raise SystemExit(f"dag_id column missing in {path}")
            for row in reader:
                dag_id = parse_int(row.get("dag_id"))
                if dag_id is None:
                    continue
                if limit_per_bin > 0 and count >= limit_per_bin:
                    break
                ids.add(dag_id)
                count += 1
        counts.append((label, count))
        ids_by_bin.append(ids)

    if not ids_by_bin:
        return set(), counts

    if mode == "intersection":
        keep_ids = set.intersection(*ids_by_bin)
    else:
        keep_ids = set.union(*ids_by_bin)

    return keep_ids, counts


def filter_catalog(catalog: Path, output: Path, keep_ids: Set[int]) -> None:
    with catalog.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("Catalog has no header")
        fieldnames = list(reader.fieldnames)

        with output.open("w", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            total = 0
            kept = 0
            invalid = 0
            for row in reader:
                total += 1
                dag_id = parse_int(row.get("dag_id"))
                if dag_id is None:
                    invalid += 1
                    continue
                if dag_id in keep_ids:
                    writer.writerow(row)
                    kept += 1

    removed = total - kept - invalid
    print(
        f"Catalog filtered: total={total}, kept={kept}, removed={removed}, invalid={invalid}"
    )


def main() -> None:
    args = build_parser().parse_args()

    roster_dir = args.roster_dir or (args.run_dir / "rosters")
    if not roster_dir.exists():
        raise SystemExit(f"Roster directory not found: {roster_dir}")

    catalog_path, primary_catalog, catalog_counts = select_catalog_path(
        roster_dir, args.catalog
    )
    if not catalog_path.exists():
        raise SystemExit(f"Catalog not found: {catalog_path}")

    if args.limit_per_bin < 0:
        raise SystemExit("--limit-per-bin must be >= 0")

    bins_text = args.bins.strip().lower()
    if bins_text == "all":
        bins = discover_bins(roster_dir)
        if not bins:
            raise SystemExit("No CPR bin rosters found to include.")
    else:
        bins = parse_bins(args.bins)
        if not bins:
            raise SystemExit("--bins is empty")

    roster_paths = [roster_dir / f"dag_roster_cpr_{label}.csv" for label in bins]
    keep_ids, counts = collect_ids(roster_paths, args.limit_per_bin, args.mode)
    if not keep_ids:
        raise SystemExit("No DAG IDs collected from roster files.")

    if args.catalog is None:
        primary_rows = catalog_counts.get(primary_catalog, 0)
        selected_rows = catalog_counts.get(catalog_path, 0)
        print(f"Catalog input: {catalog_path} (rows={selected_rows})")
        if catalog_path != primary_catalog:
            print(f"Primary catalog rows: {primary_rows}")
    else:
        print(f"Catalog input: {catalog_path}")

    print("Selected bins:")
    for label, count in counts:
        print(f"  {label}: {count}")
    print(f"Combine mode: {args.mode}")

    output_path = args.output or (roster_dir / "dag_catalog_filtered.csv")
    if args.inplace:
        output_path = primary_catalog if args.catalog is None else catalog_path
        backup_path = output_path.with_suffix(output_path.suffix + ".bak")
        if output_path.exists() and not backup_path.exists():
            shutil.copy2(output_path, backup_path)
    else:
        if output_path.resolve() == catalog_path.resolve():
            raise SystemExit("Output equals catalog; use --inplace to overwrite.")

    filter_catalog(catalog_path, output_path, keep_ids)
    print(f"Saved filtered catalog to {output_path}")


if __name__ == "__main__":
    main()
