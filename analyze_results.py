from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "raw_runs.csv"
FIGURE_DIR = ROOT / "figures"

EXPECTED_ROWS = 324
REQUIRED_COLUMNS = {
    "setting",
    "safety_on",
    "repeat",
    "temp",
    "level",
    "prompt",
    "reply",
    "warmth",
    "boundary",
    "policy_flag",
    "policy_codes",
}


def load_data() -> pd.DataFrame:
    """Load the stored experiment output without rerunning the API calls."""
    df = pd.read_csv(DATA_PATH)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} stored replies, found {len(df)}. "
            "Check whether raw_runs.csv has changed."
        )

    # Be explicit about booleans so the script behaves consistently across
    # pandas versions and CSV parsing settings.
    df["safety_on"] = (
        df["safety_on"].astype(str).str.strip().str.lower().eq("true")
    )
    df["policy_flag"] = (
        df["policy_flag"].astype(str).str.strip().str.lower().eq("true")
    )

    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Reply-level descriptive statistics used in the Substack write-up."""
    summary = (
        df.groupby(["safety_on", "level"], as_index=False)
        .agg(
            n=("policy_flag", "size"),
            policy_flag_rate=("policy_flag", "mean"),
            mean_warmth=("warmth", "mean"),
            mean_boundary=("boundary", "mean"),
        )
    )
    summary["condition"] = summary["safety_on"].map(
        {False: "Baseline", True: "Safety prompt"}
    )
    return summary


def print_results(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    display = summary[
        ["condition", "level", "n", "policy_flag_rate", "mean_warmth", "mean_boundary"]
    ].copy()
    display["policy_flag_rate"] = (100 * display["policy_flag_rate"]).round(1)

    print("Stored model replies:", len(df))
    print("Dialogue trajectories:", df["setting"].nunique())
    print()
    print("Reply-level summary by condition and intimacy level")
    print(display.to_string(index=False))
    print()

    rho = df["warmth"].corr(df["boundary"], method="spearman")
    print(f"Spearman correlation, warmth vs. boundary: rho = {rho:.3f}")

    # A few values discussed directly in the write-up.
    def row(safety: bool, level: int) -> pd.Series:
        return summary[
            (summary["safety_on"] == safety) & (summary["level"] == level)
        ].iloc[0]

    safety_l2 = row(True, 2)
    safety_l5 = row(True, 5)
    baseline_l5 = row(False, 5)

    print(
        "Safety-prompt level 2 judge-flag rate: "
        f"{100 * safety_l2['policy_flag_rate']:.1f}% "
        f"({int(round(safety_l2['policy_flag_rate'] * safety_l2['n']))}/{int(safety_l2['n'])})"
    )
    print(
        "Safety-prompt level 5 judge-flag rate: "
        f"{100 * safety_l5['policy_flag_rate']:.1f}% "
        f"({int(round(safety_l5['policy_flag_rate'] * safety_l5['n']))}/{int(safety_l5['n'])})"
    )
    print(
        "Baseline level 5 judge-flag rate: "
        f"{100 * baseline_l5['policy_flag_rate']:.1f}% "
        f"({int(round(baseline_l5['policy_flag_rate'] * baseline_l5['n']))}/{int(baseline_l5['n'])})"
    )


def save_line_plot(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    multiplier: float = 1.0,
) -> None:
    plt.figure(figsize=(8, 5))

    for condition in ["Baseline", "Safety prompt"]:
        subset = summary[summary["condition"] == condition].sort_values("level")
        plt.plot(
            subset["level"],
            subset[metric] * multiplier,
            marker="o",
            label=condition,
        )

    plt.xlabel("Intimacy level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(range(6))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close()


def make_figures(summary: pd.DataFrame) -> None:
    """Regenerate the three summary plots discussed in the write-up."""
    FIGURE_DIR.mkdir(exist_ok=True)

    save_line_plot(
        summary,
        "policy_flag_rate",
        "Evaluator flag rate (%)",
        "Evaluator policy flags by intimacy level",
        "policy_flags_by_level.png",
        multiplier=100,
    )
    save_line_plot(
        summary,
        "mean_warmth",
        "Mean warmth score (0–2)",
        "Warmth by intimacy level",
        "warmth_by_level.png",
    )
    save_line_plot(
        summary,
        "mean_boundary",
        "Mean boundary score (0–2)",
        "Boundary-setting by intimacy level",
        "boundary_by_level.png",
    )


def main() -> None:
    df = load_data()
    summary = summarize(df)
    print_results(df, summary)
    make_figures(summary)
    print(f"\nSaved figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
