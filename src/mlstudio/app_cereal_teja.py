"""app_cereal_teja.py - custom project (Phase 5).

Investigating a model from the outside, applied to a new problem.

The example project probes a deployed penguin classifier over HTTP. Here I
train a rating model on breakfast cereal nutrition, put it behind the same
payload-in / prediction-out contract, and then probe it with the same
techniques: baselines, one-feature sweeps, a sensitivity ranking, a two-feature
grid, and edge cases.

Author: Venkat Teja
Date: 2026-08

Process:
    - Load the cereals dataset.
    - Train a supervised regression model to predict rating.
    - Serve it behind an API-shaped predict() contract.
    - Probe the served model: sweeps, sensitivity ranking, grid, edge cases.
    - Create useful charts.

Data Source:
- data/raw/cereal.csv (80 Cereals, Kaggle: crawford/80-cereals)

Terminal command to run this file from the root project folder:

uv run python -m mlstudio.app_cereal_teja
"""

# === Section 1a. DECLARE IMPORTS (BRING IN FREE CODE) ===

from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("ML_CEREAL_TEJA", level="DEBUG")
log_header(LOG, "ML_CEREAL_TEJA")

# === Section 1c. Global Constants and Configuration ===

DATASET_NAME: Final[str] = "cereal"

# The published file uses -1 to mean "not recorded" instead of leaving the
# cell blank. It must be converted before modeling.
SENTINEL_MISSING: Final[int] = -1

# The target is the Consumer Reports rating carried in the dataset (0-100).
TARGET_COL: Final[str] = "rating"

# Nutrition facts only. Identity columns (name, mfr, type) and shelf placement
# are deliberately excluded: the question is whether nutrition alone explains
# the rating.
FEATURE_COLS: Final[list[str]] = [
    "calories",
    "protein",
    "fat",
    "sodium",
    "fiber",
    "carbo",
    "sugars",
    "potass",
    "vitamins",
]

TEST_SIZE: Final[float] = 0.30
RANDOM_STATE: Final[int] = 42

# Charts are written here so README.md and docs/index.md can display them.
IMAGES_DIR: Final[Path] = Path("docs") / "images"

# === Section 1d. Pandas Configuration for Display ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load the cereals dataset from the data/raw folder."""
    LOG.info(f"Loading dataset: {DATASET_NAME}")

    df: pd.DataFrame = pd.read_csv(f"data/raw/{DATASET_NAME}.csv")

    LOG.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    LOG.debug(f"\n{df.head()}")

    return df


# === Section 3. Inspect Data Shape and Structure ===


def inspect_basic(df: pd.DataFrame) -> None:
    """Inspect basic dataset structure."""
    LOG.info("Column names")
    LOG.debug(f"{list(df.columns)}")

    LOG.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Counted through numpy so the result is plain ints, not a pandas object.
    labels, counts = np.unique(df["mfr"].to_numpy(), return_counts=True)
    mfr_counts = {str(k): int(v) for k, v in zip(labels, counts, strict=True)}
    LOG.info(f"Manufacturers: {mfr_counts}")

    ratings = df[TARGET_COL].to_numpy()
    LOG.info(f"Rating range: {ratings.min():.1f} to {ratings.max():.1f}")


# === Section 4. Check Data Quality ===


def check_quality(df: pd.DataFrame) -> None:
    """Check missing values and duplicate rows."""
    LOG.info("Missing values by column")
    missing = df.isna().sum()
    LOG.debug(f"\n{missing}")

    # Built through numpy so the counts are plain ints, not a pandas object.
    missing_counts = {
        str(column): int(count)
        for column, count in zip(df.columns, missing.to_numpy(), strict=True)
        if count > 0
    }
    LOG.info(f"Columns with blank (NaN) values: {missing_counts}")

    # isna() reports nothing for this file, because the missing values are
    # written as -1 rather than left blank. Counting the sentinel separately
    # is what keeps that from passing as clean data.
    sentinel_counts: dict[str, int] = {}
    for name in FEATURE_COLS:
        count = int((df[name].to_numpy() == SENTINEL_MISSING).sum())
        if count > 0:
            sentinel_counts[name] = count

    LOG.info(f"Columns using the {SENTINEL_MISSING} sentinel: {sentinel_counts}")
    if sentinel_counts and not missing_counts:
        LOG.warning(
            "No blank values, but sentinels are present: this data is NOT clean."
        )

    duplicate_count: int = df.duplicated().sum()
    LOG.info(f"Duplicate row count: {duplicate_count}")


# === Section 5. Create a Clean View ===


def make_clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned view for modeling.

    The published file encodes missing values as the sentinel -1 rather than
    leaving them blank. Treating -1 as a real measurement would feed a negative
    gram count into the model, so the sentinel is converted to NaN and those
    rows are dropped. See docs/index.md for why dropping is the right call
    here: the published rating for those rows was itself computed from the -1,
    so their target values are unreliable, not just their features.
    """
    LOG.info("Creating clean modeling view")

    selected_cols: list[str] = FEATURE_COLS + [TARGET_COL]

    df_selected: pd.DataFrame = df[selected_cols]  # type: ignore[assignment]
    df_clean: pd.DataFrame = df_selected.copy()

    sentinel_total: int = 0
    for name in FEATURE_COLS:
        is_sentinel = df_clean[name] == SENTINEL_MISSING
        sentinel_total += int(is_sentinel.sum())
        df_clean.loc[is_sentinel, name] = np.nan

    LOG.info(f"Converted {sentinel_total} sentinel {SENTINEL_MISSING} values to NaN")

    df_clean = df_clean.dropna().copy()

    dropped: int = df.shape[0] - df_clean.shape[0]
    LOG.info(f"Clean view: {df_clean.shape[0]} rows ({dropped} dropped for missing)")
    return df_clean


# === Section 6. Train Supervised Model ===


def train_model(df_clean: pd.DataFrame) -> LinearRegression:
    """Train a supervised regression model to predict cereal rating."""
    LOG.info("Training LinearRegression model")

    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae: float = mean_absolute_error(y_test, y_pred)
    r2: float = r2_score(y_test, y_pred)

    LOG.info(f"Mean absolute error: {mae:.2e}")
    LOG.info(f"R-squared: {r2:.6f}")

    # A near-zero residual is the finding, not a bug: it means the published
    # rating is a fixed formula over these columns, not a noisy judgment.
    residual_max: float = float(np.abs(y_test - y_pred).max())
    LOG.info(f"Largest test residual: {residual_max:.2e} rating points")

    return model


# === Section 7. Serve the Model Behind an API-Shaped Contract ===


def predict(model: LinearRegression, payload: dict[str, float]) -> float:
    """Return the predicted rating for one payload of nutrition facts.

    WHY: The example project calls a deployed model through a single function
    that takes a payload dict. Keeping that same shape here means every probing
    technique from the example transfers over unchanged.

    Args:
        model: the trained regression model.
        payload: dict of feature name -> value; every feature is required.

    Returns:
        Predicted rating.

    Raises:
        KeyError: if any required feature is absent, mirroring the HTTP 400
            the deployed penguin API returns for an incomplete payload.
    """
    missing: list[str] = [name for name in FEATURE_COLS if name not in payload]
    if missing:
        raise KeyError(f"missing required features: {missing}")

    row = pd.DataFrame([{name: payload[name] for name in FEATURE_COLS}])
    return float(model.predict(row)[0])


def baseline_payload(df_clean: pd.DataFrame) -> dict[str, float]:
    """Build one representative payload: the median cereal."""
    return {name: float(np.median(df_clean[name].to_numpy())) for name in FEATURE_COLS}


# === Section 8. Probe: Vary One Feature at a Time ===


def sweep_feature(
    model: LinearRegression,
    base: dict[str, float],
    feature: str,
    values: Sequence[float],
) -> pd.DataFrame:
    """Vary one feature across a range and collect predicted ratings.

    Holding all other features fixed isolates the effect of one input, exactly
    as in the example project.
    """
    rows = [
        {feature: value, "predicted_rating": predict(model, {**base, feature: value})}
        for value in values
    ]
    return pd.DataFrame(rows)


def rank_sensitivity(model: LinearRegression, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Rank features by how far each one can actually move the prediction.

    WHY: A coefficient is "rating points per unit", which is not comparable
    across features measured on different scales. Multiplying by the range the
    feature actually spans in the data gives the influence each one can really
    exert, which is a different ranking.
    """
    rows = []
    for index, name in enumerate(FEATURE_COLS):
        column = df_clean[name].to_numpy()
        low = float(column.min())
        high = float(column.max())
        coefficient = float(model.coef_[index])
        rows.append(
            {
                "feature": name,
                "coefficient": round(coefficient, 4),
                "observed_range": round(high - low, 1),
                "influence_points": round(coefficient * (high - low), 1),
            }
        )

    df_rank = pd.DataFrame(rows)
    df_rank["abs_influence"] = df_rank["influence_points"].abs()
    return df_rank.sort_values("abs_influence", ascending=False, ignore_index=True)


# === Section 9. Probe: Edge Cases ===


def probe_edge_cases(model: LinearRegression, base: dict[str, float]) -> None:
    """Send unusual and invalid payloads and report what comes back."""
    LOG.info("Edge case results:")

    edge_cases: list[tuple[str, dict[str, float]]] = [
        ("negative sugars", {**base, "sugars": -5.0}),
        ("impossible fiber", {**base, "fiber": 1000.0}),
        ("zero everything", dict.fromkeys(FEATURE_COLS, 0.0)),
        ("candy bar calories", {**base, "calories": 5000.0}),
    ]

    for label, payload in edge_cases:
        rating = predict(model, payload)
        flag = "OUT OF RANGE" if not 0.0 <= rating <= 100.0 else "in range"
        LOG.info(f"  {label:<20} -> rating={rating:>10.1f}  [{flag}]")

    # A payload missing a required field should be refused, not answered.
    incomplete = {name: value for name, value in base.items() if name != "fiber"}
    try:
        predict(model, incomplete)
        LOG.warning("  missing feature       -> ACCEPTED (should have been refused)")
    except KeyError as exc:
        LOG.info(f"  {'missing feature':<20} -> refused: {exc}")


# === Section 10. Create Visualizations ===


def make_plots(
    model: LinearRegression,
    df_clean: pd.DataFrame,
    df_rank: pd.DataFrame,
    base: dict[str, float],
) -> None:
    """Create charts for the cereal rating investigation."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    LOG.info("Creating chart: influence ranking")

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_plt: Axes = sns.barplot(
        data=df_rank,
        x="influence_points",
        y="feature",
        hue="feature",
        legend=False,
        ax=ax,
    )
    bar_plt.set_title("Rating points each feature can move (CLOSE chart to continue)")
    bar_plt.set_xlabel("Coefficient x observed range (rating points)")
    bar_plt.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "cereal_influence_teja.png", dpi=120, bbox_inches="tight")

    LOG.info("Creating chart: fiber and sugars response curves")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, feature in zip(axes, ["fiber", "sugars"], strict=True):
        column = df_clean[feature].to_numpy()
        low = float(column.min())
        high = float(column.max())
        sweep = sweep_feature(
            model, base, feature, [float(v) for v in np.linspace(low, high, 25)]
        )
        ax.plot(sweep[feature], sweep["predicted_rating"], marker="o", markersize=4)
        ax.set_xlabel(feature)
        ax.set_ylabel("predicted rating")
        ax.set_title(f"Rating as {feature} varies")
    fig.suptitle("Response curves, other features held at the median cereal")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "cereal_response_teja.png", dpi=120, bbox_inches="tight")

    LOG.info("Creating chart: fiber x sugars grid")

    fiber_values = [float(v) for v in np.linspace(0, 14, 12)]
    sugar_values = [float(v) for v in np.linspace(0, 15, 12)]
    grid_rows = [
        {
            "fiber": round(fiber, 1),
            "sugars": round(sugar, 1),
            "predicted_rating": predict(
                model, {**base, "fiber": fiber, "sugars": sugar}
            ),
        }
        for fiber in fiber_values
        for sugar in sugar_values
    ]
    pivot = pd.DataFrame(grid_rows).pivot(
        index="sugars", columns="fiber", values="predicted_rating"
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    heat_plt: Axes = sns.heatmap(pivot, cmap="viridis", linewidths=0.3, ax=ax)
    heat_plt.set_title("Predicted rating: fiber vs sugars (CLOSE chart to continue)")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "cereal_grid_teja.png", dpi=120, bbox_inches="tight")

    LOG.info(f"Charts saved to {IMAGES_DIR.resolve()}")


# === Section 11. Summary and Next Steps ===


def summarize(df: pd.DataFrame, df_clean: pd.DataFrame, df_rank: pd.DataFrame) -> None:
    """Log a brief summary."""
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")
    LOG.info(f"Dataset: {DATASET_NAME}")
    LOG.info(f"Original rows: {df.shape[0]}")
    LOG.info(f"Clean rows: {df_clean.shape[0]}")
    LOG.info(f"Features: {FEATURE_COLS}")
    LOG.info(f"Target: {TARGET_COL}")
    LOG.info("Influence ranking (rating points across the observed range):")
    for row in df_rank.to_dict("records"):
        LOG.info(
            f"  {row['feature']:<10} coef={row['coefficient']:+8.4f}  "
            f"range={row['observed_range']:>6}  "
            f"influence={row['influence_points']:+7.1f}"
        )


# === DEFINE THE MAIN FUNCTION THAT CALLS OTHER FUNCTIONS ===


def main() -> None:
    """Main function to run the cereal rating investigation."""
    log_header(LOG, "ML_CEREAL_TEJA")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    LOG.info("Load dataset..............")
    df = load_data()

    LOG.info("Inspect dataset...........")
    inspect_basic(df)

    LOG.info("Check data quality........")
    check_quality(df)

    LOG.info("Create clean view.........")
    df_clean = make_clean_view(df)

    LOG.info("Train supervised model....")
    model = train_model(df_clean)

    LOG.info("Probe baseline............")
    base = baseline_payload(df_clean)
    LOG.info(f"Median cereal payload: {base}")
    LOG.info(f"Predicted rating: {predict(model, base):.2f}")

    LOG.info("Rank feature influence....")
    df_rank = rank_sensitivity(model, df_clean)
    LOG.info(f"\n{df_rank.drop(columns='abs_influence').to_string(index=False)}")

    LOG.info("Probe edge cases..........")
    probe_edge_cases(model, base)

    LOG.info("Create charts.............")
    make_plots(model, df_clean, df_rank, base)

    LOG.info("Summarize workflow........")
    summarize(df, df_clean, df_rank)

    LOG.info(
        "----- in a script, call plt.show() once at the end to display all charts -----"
    )
    LOG.info(
        "----- in a script, CLOSE the chart windows with the close button to CONTINUE -----"
    )

    plt.show()

    LOG.info("Workflow complete")
    LOG.info("IMPORTANT: This script creates chart windows.")
    LOG.info("Close chart windows and terminate this process with CTRL+c as needed.")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
