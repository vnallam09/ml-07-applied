# ml-07-applied

[![Workflow Guide](https://img.shields.io/badge/Pro--Guide-pro--analytics--02-green)](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/)
[![Python 3.14](https://img.shields.io/badge/python-3.14%2B-blue?logo=python)](./pyproject.toml)
[![MIT](https://img.shields.io/badge/license-see%20LICENSE-yellow.svg)](./LICENSE)

> Professional Python project: investigating a deployed machine learning model.

## Project Description

This project focuses on learning to interrogate a deployed ML model
by probing it systematically with different inputs.

We learn to:

- call a live prediction API from a notebook
- vary input features and observe how predictions change
- identify decision boundaries and edge cases
- interpret model behavior from the outside

## Example Notebook + Your Notebook

Keep the example notebook as it is.
Either copy it or use it to build a new notebook that ends in _yourname.
See [docs/your-files.md] for more.

Links:

- [ml_07_case.ipynb](notebooks/ml_07_case.ipynb)

## Working Files

You'll work with these areas:

- **data/raw** - raw data for exploration (only if you add a dataset)
- **docs/** - project narrative and documentation
- **src/mlstudio/** - the app is an example; run only (no need to modify)
- **notebooks/** - interactive analysis
- **pyproject.toml** - update authorship & links
- **zensical.toml** - update authorship & links

## My Files (Phase 4)

I copied the example notebook and made my technical change in my copy.
The example notebook is unchanged and still runs.

| Example (unchanged)            | My copy                          |
| ------------------------------ | -------------------------------- |
| `notebooks/ml_07_case.ipynb`   | `notebooks/ml_07_teja_p4.ipynb`  |

**What I changed:** the example sweeps one feature (`bill_length_mm`) and then
asks which feature produces the sharpest boundary - a question a single sweep
cannot answer. I added **Section 3b**, which reuses the example's own
`sweep_feature()` function on **all four** features across their realistic
ranges (4 x 20 = 80 calls) and ranks them by how much each one moves the
prediction.

**Why:** "which input matters most" was the notebook's own open question, and
turning it from a guess into a measurement costs one loop and one chart.

**Result:** `bill_length_mm` changed the prediction on **60%** of its swept
range with a single clean boundary near **41.8 mm**. The other three features
changed it on **0%** of theirs - 60 calls without ever leaving Adelie.

![Only bill_length_mm changes the prediction; the other three features are flat across their full range](./docs/images/feature_sensitivity_teja_p4.png)

**Three bugs the measurement caught,** each fixed in my copy:

1. **A false `MISMATCH`.** The 30s timeout is shorter than this free-tier
   API's cold start (measured ~50s), so the first baseline call timed out and
   was scored as a wrong prediction. Longer timeout plus a warm-up call.
2. **A fake decision boundary.** One dropped connection mid-sweep left an
   error string where a species belonged, and my boundary detector counted it
   as a boundary. Transient failures now retry; an HTTP 4xx is a real answer
   and is never retried.
3. **A mislabeled heatmap.** Section 3b proved Gentoo is never predicted from
   this baseline, yet the grid showed large seagreen regions - the color its
   own title assigns to Gentoo. With only two species present, the color list
   stretched and Chinstrap took Gentoo's color. Pinning `vmin`/`vmax` fixed it.

![Corrected prediction grid: Chinstrap now renders in its own color, and the boundary steps out at flipper lengths of 210mm and above](./docs/images/prediction_grid_teja_p4.png)

I also cleared the type errors VS Code reports on the example. `sweep_feature()`
declared `values: list[float]`, but `np.linspace()` returns numpy floats, and
`list` is invariant - so `list[np.float64]` is rejected even though
`np.float64` subclasses `float`. Widening the parameter to the covariant
`Sequence[float]` fixes every call site at once without touching a single call.
My notebook now type-checks with **0 errors**.

Run my notebook with **Run All** in VS Code, or headlessly:

```shell
uv run python -m nbconvert --to notebook --execute --inplace notebooks/ml_07_teja_p4.ipynb
```

Full write-up: [docs/index.md](docs/index.md).

## My Custom Project (Phase 5): What Makes a Cereal Highly Rated?

I applied the example's skills - probing a model from the outside - to a new
problem: predicting the Consumer Reports `rating` of 77 breakfast cereals from
their nutrition facts.

| My files | Purpose |
| --- | --- |
| `notebooks/ml_07_teja_p5.ipynb` | the investigation and narrative |
| `src/mlstudio/app_cereal_teja.py` | the model, served behind an API-shaped contract |
| `tests/test_app_cereal_teja.py` | smoke test + the refuses-incomplete-payload test |
| `data/raw/cereal.csv` | [80 Cereals](https://www.kaggle.com/datasets/crawford/80-cereals), 77 rows x 16 columns |

```shell
uv run python -m mlstudio.app_cereal_teja
```

There is no cereal API to call, so I trained the model and then put it behind
the same contract the example uses - a payload dict in, a prediction out - and
investigated it from the outside with the same techniques: baselines, sweeps,
an influence ranking, a two-feature grid, and edge cases.

### Finding 1: the data looks clean and is not

The file has no blank cells, so a missing-value check reports zero problems.
But four values across three cereals are recorded as `-1`, meaning "not
measured". Fed to the model with `-1` taken literally, all three cereals'
published ratings come back **exactly**; replaced with a sensible `0`, all
three miss.

So the rating column was computed without cleaning the sentinel first - those
rows carry ratings derived from a negative gram count. That makes their
**target** unreliable, not just their features, so I drop them rather than
impute. Probing the model from the outside ended up revealing how the dataset
itself was built.

### Finding 2: the target is a formula, not a judgment

R-squared is **1.000000** and the largest residual on held-out cereals is
**5e-07** rating points. The model does not approximate the rating, it
reproduces it exactly. A score carrying human judgment could never fit like
that, so the rating in this dataset is a fixed linear function of the nine
nutrition columns - and probing the model recovers the formula itself.

### Finding 3: ranking by coefficient gives the wrong answer

![Fiber dominates, and sodium outranks sugar once the observed range is accounted for](./docs/images/cereal_influence_teja.png)

`protein` has nearly fiber's coefficient (+3.27 vs +3.44) but cereals only span
1g to 6g of protein, so it can move a rating 16 points at most. `calories` has
a coefficient 15x smaller and **1.5x the real influence**, because it ranges
over 110 units. Fiber wins either way, and wins big: 48 rating points, more than
the next two features combined. One gram of fiber cancels about 4.8g of sugar.

Sugar, the feature I expected to dominate a healthiness score, ranks **sixth** -
behind sodium.

### Finding 4: the model has no guardrails

| probe | predicted rating |
| --- | --- |
| impossible fiber (1000g) | **3477.0** |
| candy bar calories (5000) | **-1048.7** |
| a box of literally nothing | **54.9**, beating 62 of 74 real cereals |
| missing a required field | refused |

An empty box scores 54.9 - the model's intercept - and beats **84%** of real
cereals, because the formula starts high and deducts for what a cereal
contains. Only the bran and shredded wheat end of the aisle does better than
nothing at all. The same weakness the example finds in the penguin API (a 999mm
bill classified without hesitation) shows up here: a linear model cannot know
the edge of its own training range.

Full write-up: [docs/index.md](docs/index.md).

## Additional Packages

This project uses `requests` to make the calls.
Be sure the requests package is listed in `pyproject.toml`.

## Instructions (pro-analytics-02)

Follow the
[step-by-step workflow guide](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/)
to complete:

1. Phase 1. **Start & Run**
2. Phase 2. **Change Authorship**
3. Phase 3. **Read & Understand**
4. Phase 4. **Modify**
5. Phase 5. **Apply**

## Challenges

Challenges are expected.
Sometimes instructions may not quite match your operating system.
When issues occur, share screenshots, error messages, and details about what you tried.
Working through issues is part of implementing professional projects.

## Success

After completing Phase 1. **Start & Run**, you'll have your own GitHub project,
with the example notebook executed and committed,
and running the example module will print out:

```shell
========================
Executed successfully!
========================
```

A new file `project.log` will appear in the root project folder.

## Command Reference

<details>
<summary>Show command reference</summary>

### In a machine terminal (open in your `Repos` folder)

After you get a copy of this repo in your own GitHub account,
open a machine terminal in your `Repos` folder:

```shell
# Replace username with YOUR GitHub username.
git clone https://github.com/vnallam09/ml-07-applied

cd ml-07-applied
code .
```

### In a VS Code terminal

These are listed for convenience.
For best results, follow the detailed instructions in
[pro-analytics-02 guide](https://denisecase.github.io/pro-analytics-02/).

```shell
uv self update
uv python pin 3.14
uv lock --upgrade
uv sync --extra dev --extra docs --upgrade

uvx pre-commit install
uvx pre-commit autoupdate

git add -A
uvx pre-commit run --all-files
# repeat if changes were made
uvx pre-commit run --all-files

# run the example module to verify the environment (.venv/)
uv run python -m mlstudio.app_case

# run common chores
uv run ruff format .
uv run ruff check . --fix
uv run python -m pyright
uv run python -m pytest
uv run python -m zensical build

# save progress
git add -A
git commit -m "update"
git push -u origin main
```

</details>

## Notes

- Use the **UP ARROW** and **DOWN ARROW** in the terminal to scroll through past commands.
- Use `CTRL+f` to find (and replace) text within a file.
- You do not need to add to or modify `tests/`. They are provided for example only.
- Many files are silent helpers. Explore as you like, but nothing is required.
- You do NOT need to understand everything; understanding builds naturally over time.

## Troubleshooting >>>

If you see something like this in your terminal: `>>>` or `...`
You accidentally started Python interactive mode.
It happens.
Press `Ctrl+c` (both keys together) or `Ctrl+Z` then `Enter` on Windows.

## Example Output (Can Remove this Section after You Verify)

```shell
| INFO | ML | Summarize workflow........
| INFO | ML | ========================
| INFO | ML | SUMMARY
| INFO | ML | ========================
| INFO | ML | Dataset: hours_scores_case
| INFO | ML | Original rows: 10
| INFO | ML | Clean rows: 10
| INFO | ML | Features: ['hours_studied', 'practice_quizzes', 'attendance_pct', 'sleep_hours', 'prior_score']
| INFO | ML | Target: score
| INFO | ML | ----- in a script, call plt.show() once at the end to display all charts -----
| INFO | ML | ----- in a script, CLOSE the chart windows with the close button to CONTINUE -----
| INFO | ML | Workflow complete
| INFO | ML | IMPORTANT: This script creates chart windows.
| INFO | ML | Close chart windows and terminate this process with CTRL+c as needed.
| INFO | ML | ========================
| INFO | ML | Executed successfully!
| INFO | ML | ========================
```

## Findings and Visuals

Take screenshots of your charts and provide them here with a discussion.
In Markdown, display a figure by using:
an exclamation mark immediately followed by square brackets containing a useful caption
immediately followed by parentheses containing the relative path to your figure.
Note: When you start typing the path with a dot (.) for "here, in this directory",
the IDE may help complete the path.

In your custom project, follow this example, but

- your figures and narrative should reflect your work,
- this `README.md` should include your commands, process, and visuals, and
- `docs/index.md` should include your narrative.

Remove unnecessary instructional comments in your custom files.

Update figures to present interesting results from your custom project:

![Provide a Useful Caption](./docs/images/Figure_1.png)

![Provide a Useful Caption](./docs/images/Figure_2.png)

## Project Documentation

Additional project instructions, terms, and notes:

[docs/index.md](docs/index.md)

## Citation

[CITATION.cff](./CITATION.cff)

## License

[MIT](./LICENSE)
