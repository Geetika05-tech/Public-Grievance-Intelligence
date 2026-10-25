"""
STAGE 9 — Emerging Issue Detection
Day 8 of the build plan.

Pipeline (matches the plan exactly):
    Cluster -> group by date/week -> count complaints ->
    compare current volume with baseline -> growth/anomaly score -> flag

Start simple (minimum complaint count + growth threshold), and only
add a proper historical baseline/anomaly model if time allows — this
module supports both.
"""

from collections import defaultdict
from datetime import datetime
import statistics


def _week_key(date_str: str) -> str:
    """Turn an ISO date string into a 'YYYY-Www' bucket key."""
    dt = datetime.fromisoformat(date_str)
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def weekly_counts_by_cluster(complaints: list[dict]) -> dict[int, dict[str, int]]:
    """
    complaints: list of {"cluster_id": int, "date": iso-string, ...}
    Returns: {cluster_id: {week_key: count}}
    """
    counts = defaultdict(lambda: defaultdict(int))
    for c in complaints:
        if c.get("cluster_id") is None:
            continue
        week = _week_key(c["date"])
        counts[c["cluster_id"]][week] += 1
    return {cid: dict(weeks) for cid, weeks in counts.items()}


def detect_emerging_issues(
    weekly_counts: dict[int, dict[str, int]],
    cluster_names: dict[int, str],
    min_complaints: int = 10,
    growth_threshold: float = 0.3,
) -> list[dict]:
    """
    Simple rule (as specified in the plan):
        current_week_count >= min_complaints
        AND
        growth vs baseline (mean of prior weeks) > growth_threshold

    growth = (current - baseline) / baseline

    Returns a list of flagged issues sorted by growth, highest first.
    """
    flagged = []

    for cluster_id, weeks in weekly_counts.items():
        sorted_weeks = sorted(weeks.keys())
        if len(sorted_weeks) < 2:
            continue  # not enough history to compute a baseline

        *prior_weeks, current_week = sorted_weeks
        current_count = weeks[current_week]
        prior_counts = [weeks[w] for w in prior_weeks]
        baseline = statistics.mean(prior_counts) if prior_counts else 0

        if current_count < min_complaints:
            continue

        growth = (current_count - baseline) / baseline if baseline > 0 else float("inf")

        if growth > growth_threshold:
            flagged.append({
                "cluster_id": cluster_id,
                "issue": cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
                "current_week": current_week,
                "current_count": current_count,
                "baseline": round(baseline, 1),
                "growth_pct": round(growth * 100, 1) if growth != float("inf") else None,
            })

    flagged.sort(key=lambda x: (x["growth_pct"] is None, -(x["growth_pct"] or 0)))
    return flagged


if __name__ == "__main__":
    # Simulated example matching the plan's "Water Supply spike" scenario
    weekly_counts = {
        0: {"2026-W01": 12, "2026-W02": 15, "2026-W03": 18, "2026-W04": 43},  # Water
        1: {"2026-W01": 8, "2026-W02": 9, "2026-W03": 8, "2026-W04": 10},     # Roads
    }
    cluster_names = {0: "Water Supply", 1: "Roads"}

    for issue in detect_emerging_issues(weekly_counts, cluster_names, min_complaints=10, growth_threshold=0.3):
        print(issue)
