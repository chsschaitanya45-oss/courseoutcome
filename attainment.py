from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class AttainmentConfig:
    mark_threshold_percent: float
    direct_weight: float
    indirect_weight: float
    target_level: float
    level_1_min_percent: float
    level_2_min_percent: float
    level_3_min_percent: float

    @property
    def threshold_fraction(self) -> float:
        return self.mark_threshold_percent / 100


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip().lower().replace(" ", "_") for col in cleaned.columns]
    return cleaned


def attainment_level(percent_crossing: float, config: AttainmentConfig) -> int:
    if percent_crossing >= config.level_3_min_percent:
        return 3
    if percent_crossing >= config.level_2_min_percent:
        return 2
    if percent_crossing >= config.level_1_min_percent:
        return 1
    return 0


def validate_required_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> list[str]:
    normalized = normalize_columns(df)
    missing = [col for col in columns if col not in normalized.columns]
    return [f"{name} is missing column: {col}" for col in missing]


def compute_direct_attainment(marks: pd.DataFrame, config: AttainmentConfig) -> pd.DataFrame:
    df = normalize_columns(marks)
    df["max_marks"] = pd.to_numeric(df["max_marks"], errors="coerce")
    df["marks_obtained"] = pd.to_numeric(df["marks_obtained"], errors="coerce")
    df = df.dropna(subset=["co", "student_id", "max_marks", "marks_obtained"])
    df["crossed_threshold"] = df["marks_obtained"] >= df["max_marks"] * config.threshold_fraction

    by_student = (
        df.groupby(["co", "student_id"], as_index=False)
        .agg(marks_obtained=("marks_obtained", "sum"), max_marks=("max_marks", "sum"))
        .assign(crossed_threshold=lambda data: data["marks_obtained"] >= data["max_marks"] * config.threshold_fraction)
    )
    direct = (
        by_student.groupby("co", as_index=False)
        .agg(
            students_assessed=("student_id", "nunique"),
            students_crossing=("crossed_threshold", "sum"),
            average_percent=("marks_obtained", lambda values: 0.0),
        )
    )
    averages = by_student.assign(score_percent=by_student["marks_obtained"] / by_student["max_marks"] * 100)
    avg_lookup = averages.groupby("co")["score_percent"].mean().to_dict()
    direct["average_percent"] = direct["co"].map(avg_lookup).round(2)
    direct["direct_percent"] = (direct["students_crossing"] / direct["students_assessed"] * 100).round(2)
    direct["direct_level"] = direct["direct_percent"].apply(lambda value: attainment_level(value, config))
    return direct.sort_values("co").reset_index(drop=True)


def compute_question_diagnostics(marks: pd.DataFrame, config: AttainmentConfig) -> pd.DataFrame:
    df = normalize_columns(marks)
    df["max_marks"] = pd.to_numeric(df["max_marks"], errors="coerce")
    df["marks_obtained"] = pd.to_numeric(df["marks_obtained"], errors="coerce")
    df = df.dropna(subset=["co", "assessment", "question", "max_marks", "marks_obtained"])
    df["score_percent"] = df["marks_obtained"] / df["max_marks"] * 100
    df["crossed_threshold"] = df["score_percent"] >= config.mark_threshold_percent
    diagnostics = (
        df.groupby(["co", "assessment", "question"], as_index=False)
        .agg(
            attempts=("student_id", "nunique"),
            average_percent=("score_percent", "mean"),
            threshold_crossing_percent=("crossed_threshold", "mean"),
        )
    )
    diagnostics["average_percent"] = diagnostics["average_percent"].round(2)
    diagnostics["threshold_crossing_percent"] = (diagnostics["threshold_crossing_percent"] * 100).round(2)
    return diagnostics.sort_values(["co", "threshold_crossing_percent", "average_percent"]).reset_index(drop=True)


def compute_co_attainment(
    courses: pd.DataFrame,
    marks: pd.DataFrame,
    indirect: pd.DataFrame,
    config: AttainmentConfig,
) -> pd.DataFrame:
    course_df = normalize_columns(courses)
    indirect_df = normalize_columns(indirect)
    direct = compute_direct_attainment(marks, config)

    indirect_df["survey_percent"] = pd.to_numeric(indirect_df["survey_percent"], errors="coerce").fillna(0)
    indirect_df["indirect_level"] = indirect_df["survey_percent"].apply(lambda value: attainment_level(value, config))

    merged = course_df.merge(direct, on="co", how="left").merge(
        indirect_df[["co", "survey_percent", "indirect_level"]], on="co", how="left"
    )
    for column in ["students_assessed", "students_crossing", "average_percent", "direct_percent", "direct_level", "survey_percent", "indirect_level"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)

    total_weight = config.direct_weight + config.indirect_weight
    direct_weight = config.direct_weight / total_weight if total_weight else 0
    indirect_weight = config.indirect_weight / total_weight if total_weight else 0
    merged["combined_level"] = (
        merged["direct_level"] * direct_weight + merged["indirect_level"] * indirect_weight
    ).round(2)
    merged["target_level"] = config.target_level
    merged["gap"] = (merged["target_level"] - merged["combined_level"]).round(2)
    merged["status"] = merged["gap"].apply(lambda gap: "Below target" if gap > 0 else "Met target")
    return merged.sort_values("co").reset_index(drop=True)


def compute_outcome_attainment(co_attainment: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    map_df = normalize_columns(mapping)
    map_df["strength"] = pd.to_numeric(map_df["strength"], errors="coerce").fillna(0)
    merged = map_df.merge(co_attainment[["co", "combined_level"]], on="co", how="left")
    merged["weighted_score"] = merged["combined_level"].fillna(0) * merged["strength"]
    outcome = (
        merged.groupby(["outcome_type", "outcome"], as_index=False)
        .agg(total_weight=("strength", "sum"), weighted_score=("weighted_score", "sum"), contributing_cos=("co", lambda values: ", ".join(sorted(set(values)))))
    )
    outcome["attainment_level"] = (outcome["weighted_score"] / outcome["total_weight"].replace(0, pd.NA)).fillna(0).round(2)
    return outcome.sort_values(["outcome_type", "outcome"]).reset_index(drop=True)


def compute_student_below_target(marks: pd.DataFrame, config: AttainmentConfig) -> pd.DataFrame:
    df = normalize_columns(marks)
    df["max_marks"] = pd.to_numeric(df["max_marks"], errors="coerce").fillna(0)
    df["marks_obtained"] = pd.to_numeric(df["marks_obtained"], errors="coerce").fillna(0)
    df = df.dropna(subset=["co", "student_id", "max_marks", "marks_obtained"]).copy()
    per_student = (
        df.groupby(["co", "student_id"], as_index=False)
        .agg(total_marks_obtained=("marks_obtained", "sum"), total_max_marks=("max_marks", "sum"))
    )
    per_student["score_percent"] = (
        per_student["total_marks_obtained"] / per_student["total_max_marks"].replace(0, pd.NA) * 100
    ).fillna(0)
    per_student["below_target"] = per_student["score_percent"] < config.mark_threshold_percent
    return per_student[per_student["below_target"]][["co", "student_id", "score_percent"]].sort_values(["co", "student_id"]).reset_index(drop=True)


def build_gap_analysis(
    co_attainment: pd.DataFrame,
    diagnostics: pd.DataFrame,
    marks: pd.DataFrame | None = None,
    config: AttainmentConfig | None = None,
) -> pd.DataFrame:
    gaps = co_attainment[co_attainment["status"] == "Below target"].copy()
    if gaps.empty:
        return pd.DataFrame(columns=["co", "gap", "students_below_target", "probable_cause", "lowest_evidence", "hod_verification_required"])

    students_below_target = pd.DataFrame(columns=["co", "student_id", "score_percent"])
    if marks is not None and config is not None:
        students_below_target = compute_student_below_target(marks, config)

    rows = []
    for _, row in gaps.iterrows():
        co = row["co"]
        weak_items = diagnostics[diagnostics["co"] == co].head(3)
        evidence = "; ".join(
            f"{item.assessment} {item.question}: {item.threshold_crossing_percent}% crossed"
            for item in weak_items.itertuples()
        )
        students = (
            students_below_target.loc[students_below_target["co"] == co, "student_id"].astype(str).tolist()
            if not students_below_target.empty
            else []
        )
        student_list = ", ".join(students) if students else "No register numbers recorded"
        cause = (
            "Low direct performance in tagged assessment questions"
            if row["direct_level"] <= row["indirect_level"]
            else "Survey feedback indicates weaker learner confidence than exam performance"
        )
        rows.append(
            {
                "co": co,
                "gap": row["gap"],
                "students_below_target": student_list,
                "probable_cause": cause,
                "lowest_evidence": evidence or "No question-level evidence available",
                "hod_verification_required": "Yes — HOD verification required for the listed students before closure.",
            }
        )
    return pd.DataFrame(rows)


def create_excel_report(
    co_attainment: pd.DataFrame,
    outcome_attainment: pd.DataFrame,
    gap_analysis: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        co_attainment.to_excel(writer, sheet_name="CO attainment", index=False)
        outcome_attainment.to_excel(writer, sheet_name="PO PSO attainment", index=False)
        gap_analysis.to_excel(writer, sheet_name="Gap analysis", index=False)
        diagnostics.to_excel(writer, sheet_name="Traceability", index=False)
    return buffer.getvalue()


def report_context(co_attainment: pd.DataFrame, outcome_attainment: pd.DataFrame, gap_analysis: pd.DataFrame) -> str:
    return "\n\n".join(
        [
            "CO attainment\n" + co_attainment.to_csv(index=False),
            "PO and PSO attainment\n" + outcome_attainment.to_csv(index=False),
            "Gap analysis evidence\n" + gap_analysis.to_csv(index=False),
            "HOD verification requirement\nStudents with below-target performance must be reviewed by the Head of Department before closure.",
        ]
    )
