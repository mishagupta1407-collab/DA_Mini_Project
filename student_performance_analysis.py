"""IA3 Student Performance Analysis
Dataset: Kaggle Students Performance in Exams
https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

Place StudentsPerformance.csv beside this script.
Install: pip install pandas numpy matplotlib scipy
Run: python student_performance_analysis.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DATA = Path("StudentsPerformance.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
if not DATA.exists():
    raise FileNotFoundError("Download StudentsPerformance.csv from the Kaggle page and place it beside this script.")

raw = pd.read_csv(DATA)
print("Original shape:", raw.shape)
print("Missing values:", int(raw.isna().sum().sum()))
print("Duplicate rows:", int(raw.duplicated().sum()))

df = raw.copy()
df.columns = [c.strip().lower().replace("/", "_").replace(" ", "_") for c in df.columns]
score_cols = ["math_score", "reading_score", "writing_score"]
for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.drop_duplicates().dropna(subset=score_cols + ["test_preparation_course"])
for col in score_cols:
    df = df[df[col].between(0, 100)]
df["average_score"] = df[score_cols].mean(axis=1)
print("Rows after cleaning:", len(df))
print(df[score_cols + ["average_score"]].describe().round(2))

# EDA 1: distributions of subject scores
plt.figure(figsize=(8, 5))
for col in score_cols:
    plt.hist(df[col], bins=14, alpha=0.42, label=col.replace("_score", "").title())
plt.xlabel("Exam score (0–100)")
plt.ylabel("Number of students")
plt.title("Distribution of Examination Scores")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "figure1_score_distributions.png", dpi=220)
plt.show()

# EDA 2: mathematics score by test preparation
completed = df.loc[df["test_preparation_course"].eq("completed"), "math_score"]
not_completed = df.loc[df["test_preparation_course"].eq("none"), "math_score"]
plt.figure(figsize=(7, 4))
plt.boxplot([not_completed, completed], tick_labels=["Not completed", "Completed"])
plt.ylabel("Mathematics score (0–100)")
plt.title("Mathematics Scores by Test-Preparation Status")
plt.tight_layout()
plt.savefig(OUT / "figure2_math_by_prep.png", dpi=220)
plt.show()

# EDA 3: subject correlation heatmap
corr = df[score_cols].corr()
plt.figure(figsize=(6, 5))
plt.imshow(corr, vmin=-1, vmax=1)
plt.xticks(range(3), ["Math", "Reading", "Writing"])
plt.yticks(range(3), ["Math", "Reading", "Writing"])
plt.colorbar(label="Pearson correlation")
for i in range(3):
    for j in range(3):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
plt.title("Correlation Between Subject Scores")
plt.tight_layout()
plt.savefig(OUT / "figure3_subject_correlation.png", dpi=220)
plt.show()

# CDA: two-sided Welch independent-samples t-test
t_stat, p_value = stats.ttest_ind(completed, not_completed, equal_var=False)
difference = completed.mean() - not_completed.mean()
n1, n0 = len(completed), len(not_completed)
s1, s0 = completed.var(ddof=1), not_completed.var(ddof=1)
se2 = s1/n1 + s0/n0
df_welch = se2**2 / ((s1/n1)**2/(n1-1) + (s0/n0)**2/(n0-1))
critical = stats.t.ppf(0.975, df_welch)
ci_low = difference - critical*np.sqrt(se2)
ci_high = difference + critical*np.sqrt(se2)
print("\\nWelch independent-samples t-test")
print(f"Completed: n={n1}, mean={completed.mean():.2f}, SD={completed.std(ddof=1):.2f}")
print(f"Not completed: n={n0}, mean={not_completed.mean():.2f}, SD={not_completed.std(ddof=1):.2f}")
print(f"Mean difference (completed - not completed) = {difference:.2f}")
print(f"t={t_stat:.3f}, df={df_welch:.2f}, p={p_value:.4g}, 95% CI=({ci_low:.2f}, {ci_high:.2f})")
print("Decision:", "Reject H0" if p_value < 0.05 else "Fail to reject H0")
print("Interpretation: association does not establish causation.")

df.to_csv(OUT / "cleaned_students_performance.csv", index=False)
pd.DataFrame([{
    "completed_n": n1, "completed_mean": completed.mean(),
    "not_completed_n": n0, "not_completed_mean": not_completed.mean(),
    "mean_difference": difference, "welch_t": t_stat, "welch_df": df_welch,
    "p_value": p_value, "ci95_low": ci_low, "ci95_high": ci_high
}]).to_csv(OUT / "ia3_results_summary.csv", index=False)
print("Saved cleaned data and results to outputs/.")
