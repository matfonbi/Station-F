"""
Shared feature engineering utilities for the Station F project.

This module centralizes data normalization logic so that both the API
and the training pipeline transform raw inputs identically.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

LEVEL_ORDER = ["Aucun", "Certificat", "Licence", "Master", "Doctorat"]
LEVEL_ALIASES = {
    "aucun": "Aucun",
    "certificat": "Certificat",
    "licence": "Licence",
    "license": "Licence",
    "bachelor": "Licence",
    "maitrise": "Master",
    "maîtrise": "Master",
    "master": "Master",
    "doctorat": "Doctorat",
    "phd": "Doctorat",
}


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def normalize_degree_level(level: Any) -> str | None:
    if level in (None, ""):
        return None
    text = str(level).strip()
    if not text:
        return None
    key = _strip_accents(text).lower()
    normalized = LEVEL_ALIASES.get(key)
    if normalized:
        return normalized
    if text in LEVEL_ORDER:
        return text
    return None


def degree_rank(level: Any) -> int:
    if level is None:
        return -1
    try:
        return LEVEL_ORDER.index(level)
    except ValueError:
        return -1


def parse_duration_to_years(exp: Dict[str, Any]) -> float:
    """
    Convertit une expérience en durée (années), en reproduisant la logique du notebook.
    """
    if not isinstance(exp, dict):
        return np.nan

    now = pd.Timestamp.now()
    start = exp.get("start_date") or exp.get("startDate")
    end = exp.get("end_date") or exp.get("endDate")
    duration = exp.get("duration")
    dates = exp.get("dates")

    if not any([start, end, duration, dates]):
        return np.nan

    if start and end:
        start_dt = pd.to_datetime(start, errors="coerce")
        end_raw = end
        if isinstance(end_raw, str) and re.search(r"présent|present", end_raw, re.IGNORECASE):
            end_dt = now
        else:
            end_dt = pd.to_datetime(end_raw, errors="coerce")
        if pd.notna(start_dt) and pd.notna(end_dt):
            return round((end_dt - start_dt).days / 365, 2)

    if duration:
        text = str(duration).lower().strip()
        if re.search(r"\d+\s*mois", text):
            months = int(re.search(r"\d+", text).group())
            return round(months / 12, 2)
        if re.search(r"\d+\s*an", text):
            years = int(re.search(r"\d+", text).group())
            return float(years)
        if re.search(r"\d{4}", text):
            years = re.findall(r"\d{4}", text)
            if len(years) == 2:
                return int(years[1]) - int(years[0])
            if len(years) == 1 and re.search(r"présent|present", text):
                return now.year - int(years[0])

    if dates:
        text = str(dates).lower().strip()
        if re.search(r"\d{4}", text):
            years = re.findall(r"\d{4}", text)
            if len(years) == 2:
                return int(years[1]) - int(years[0])
            if len(years) == 1 and re.search(r"présent|present", text):
                return now.year - int(years[0])

    return np.nan


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        return None


def extract_highest_degree_info(diplomas: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not diplomas:
        return "Unknown", "Unknown"

    best_level = "Unknown"
    best_title = "Unknown"
    best_rank = -1

    for diploma in diplomas:
        level = normalize_degree_level(diploma.get("level"))
        if level is None:
            level = "Unknown"
        rank = degree_rank(level)
        if rank > best_rank:
            best_rank = rank
            best_level = level
            title = (diploma.get("title") or "").strip()
            best_title = title if title else "Unknown"

    if best_rank == -1:
        return "Unknown", "Unknown"
    return best_level, best_title


def compute_mean_past_rating(past_courses: Iterable[Dict[str, Any]]) -> float:
    if not past_courses:
        return np.nan
    values = []
    for course in past_courses:
        if not isinstance(course, dict):
            continue
        rating = _safe_float(course.get("numberOfStars") or course.get("rating"))
        values.append(np.nan if rating is None else rating)
    if not values:
        return np.nan
    return float(np.nanmean(values))


def count_experiences(experiences: Any) -> int:
    return len(experiences) if isinstance(experiences, list) else 0


def summarize_experiences(experiences: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    cleaned: List[Dict[str, Any]] = []
    total_years = 0.0
    for exp in experiences or []:
        if not isinstance(exp, dict):
            continue
        description = (exp.get("description") or exp.get("title") or exp.get("company") or "").strip()
        duration_text = (exp.get("duration") or exp.get("dates") or "").strip()
        years = parse_duration_to_years(exp)
        if pd.notna(years):
            total_years += years
            if not duration_text:
                duration_text = f"{years:.1f} an(s)"
        elif not duration_text and (exp.get("start_date") or exp.get("startDate") or exp.get("end_date") or exp.get("endDate")):
            start = exp.get("start_date") or exp.get("startDate") or "?"
            end = exp.get("end_date") or exp.get("endDate") or "Présent"
            duration_text = f"{start} - {end}"
        cleaned.append({
            "description": description or "Expérience",
            "duration": duration_text or None,
        })
    total_years = round(total_years, 2) if total_years > 0 else np.nan
    return cleaned, total_years


def summarize_past_courses(past_courses: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    cleaned: List[Dict[str, Any]] = []
    ratings: List[float] = []
    for course in past_courses or []:
        if not isinstance(course, dict):
            continue
        title = (course.get("title") or "").strip()
        description = (course.get("description") or course.get("school") or "").strip()
        rating = _safe_float(course.get("numberOfStars") or course.get("rating"))
        ratings.append(np.nan if rating is None else rating)
        cleaned.append({
            "title": title,
            "school": description or None,
            "rating": rating,
        })
    mean_rating = float(np.nanmean(ratings)) if ratings else np.nan
    return cleaned, mean_rating


def build_features_from_entities(
    professor_raw: Dict[str, Any],
    course_raw: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    professor = professor_raw or {}
    course = course_raw or {}

    diplomas_input = professor.get("diplomas") or []
    cleaned_diplomas: List[Dict[str, Any]] = []
    for item in diplomas_input:
        if not isinstance(item, dict):
            continue
        level = normalize_degree_level(item.get("level"))
        level = level if level else "Unknown"
        title = (item.get("title") or item.get("field") or "").strip()
        title = title if title else "Unknown"
        cleaned_diplomas.append({"level": level, "title": title})

    highest_degree, highest_domain = extract_highest_degree_info(cleaned_diplomas)
    experiences_input: List[Dict[str, Any]] = professor.get("experiences") or []
    cleaned_experiences, total_years = summarize_experiences(experiences_input)

    past_courses_input = professor.get("pastCourses") or professor.get("past_courses") or []
    cleaned_courses, mean_rating = summarize_past_courses(past_courses_input)

    city_raw = (professor.get("city") or "").strip()
    city = city_raw if city_raw else "Unknown"
    firstname = (professor.get("fistname") or professor.get("firstname") or "").strip()
    lastname = (professor.get("lastname") or "").strip()
    description = (professor.get("description") or "").strip()
    description_for_model = description if description else "Inconnu"

    course_title_raw = (course.get("title") or "").strip()
    course_title = course_title_raw if course_title_raw else "Inconnu"

    features = {
        "city": city,
        "highest_degree": highest_degree,
        "highest_degree_domain": highest_domain,
        "num_diplomas": len(diplomas_input),
        "num_experiences": count_experiences(experiences_input),
        "total_experience_years": total_years,
        "mean_past_rating": mean_rating,
        "description": description_for_model,
        "course_title": course_title,
    }

    details = {
        "nom": lastname,
        "prenom": firstname,
        "ville": city or "Non renseignée",
        "degrees": [
            {"level": d.get("level"), "field": None if d.get("title") == "Unknown" else d.get("title")}
            for d in cleaned_diplomas
        ],
        "description": description if description else "Aucune description fournie",
        "course_title": course_title_raw or "Non renseignée",
        "courses": cleaned_courses,
        "experiences": cleaned_experiences,
        "total_experience_years": None if pd.isna(total_years) else float(total_years),
        "mean_past_rating": None if pd.isna(mean_rating) else round(float(mean_rating), 2),
    }
    return features, details


def flatten_training_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    courses = entry.get("pastCourses") or []
    for course in courses:
        course_payload = {
            "title": course.get("title"),
            "description": course.get("description"),
        }
        features, _ = build_features_from_entities(entry, course_payload)
        rating = _safe_float(course.get("numberOfStars") or course.get("rating"))
        row = features.copy()
        row["course_rating"] = rating
        rows.append(row)
    return rows


def build_training_dataframe(raw_data: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    unresolved_durations: List[Dict[str, Any]] = []

    for teacher in raw_data:
        experiences = teacher.get("experiences") or []
        for exp in experiences:
            if isinstance(exp, dict):
                val = parse_duration_to_years(exp)
                if isinstance(val, float) and math.isnan(val):
                    unresolved_durations.append(exp)
        rows.extend(flatten_training_entry(teacher))

    df = pd.DataFrame(rows)
    df.dropna(subset=["course_rating"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
