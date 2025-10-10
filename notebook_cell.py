# ======================================================
#  ⚙️ Préparation complète des données - Station F / Colab
#  Version améliorée :
#   - ignore les expériences sans date
#   - ne met plus 0.0 si durée inconnue → NaN
# ======================================================

import json
import pandas as pd
import numpy as np
import re
from datetime import datetime

# --- 1. Charger le fichier ---
DATA_PATH = "data/data_train.json"

# --- 2. Liste globale pour auditer les durées non reconnues ---
unrecognized_durations = []

# --- 3. Fonctions utilitaires ---

def extract_highest_degree_info(diplomas):
    """Retourne le niveau et le domaine du diplôme le plus élevé."""
    if not diplomas:
        return "Unknown", "Unknown"

    levels_order = ["Certificat", "Licence", "Master", "Doctorat"]
    best_level_idx = -1
    best_diploma = None

    for d in diplomas:
        level = d.get("level", "Unknown")
        if level in levels_order:
            idx = levels_order.index(level)
            if idx > best_level_idx:
                best_level_idx = idx
                best_diploma = d

    if best_diploma:
        return (
            best_diploma.get("level", "Unknown"),
            best_diploma.get("title", "Unknown")
        )
    else:
        return "Unknown", "Unknown"


def parse_duration_to_years(exp):
    """
    Retourne la durée en années pour une expérience individuelle.
    Essaie les formats : start_date/end_date, duration, dates.
    Renvoie np.nan si non reconnu ou non applicable.
    """
    now = pd.Timestamp.now()

    # Vérifier qu’il existe au moins une info temporelle
    if not any(k in exp for k in ["start_date", "end_date", "duration", "dates"]):
        return np.nan  # pas de durée du tout → on ignore

    # 1️⃣ Cas : start_date / end_date
    if "start_date" in exp and "end_date" in exp:
        start = pd.to_datetime(exp.get("start_date"), errors="coerce")
        end = exp.get("end_date")

        if isinstance(end, str) and re.search(r"présent|present", end, re.IGNORECASE):
            end = now
        else:
            end = pd.to_datetime(end, errors="coerce")

        if pd.notna(start) and pd.notna(end):
            return round((end - start).days / 365, 2)

    # 2️⃣ Cas : champ unique "duration"
    elif "duration" in exp and exp["duration"]:
        text = str(exp["duration"]).lower().strip()

        # Exemples : "6 mois", "5 ans", "1 an", "2015 - Présent"
        if re.search(r"\d+\s*mois", text):
            months = int(re.search(r"\d+", text).group())
            return round(months / 12, 2)

        elif re.search(r"\d+\s*an", text):
            years = int(re.search(r"\d+", text).group())
            return float(years)

        elif re.search(r"\d{4}", text):
            years = re.findall(r"\d{4}", text)
            if len(years) == 2:
                return int(years[1]) - int(years[0])
            elif len(years) == 1 and re.search(r"présent|present", text):
                return now.year - int(years[0])

    # 3️⃣ Cas : champ unique "dates"
    elif "dates" in exp and exp["dates"]:
        text = str(exp["dates"]).lower().strip()

        if re.search(r"\d{4}", text):
            years = re.findall(r"\d{4}", text)
            if len(years) == 2:
                return int(years[1]) - int(years[0])
            elif len(years) == 1 and re.search(r"présent|present", text):
                return now.year - int(years[0])

    # ❌ Cas non reconnu malgré une donnée
    if any(v for v in exp.values()):
        unrecognized_durations.append(exp)
    return np.nan


def total_experience_years(experiences):
    """
    Calcule la durée cumulée (en années) de toutes les expériences d’un formateur.
    Retourne NaN si aucune durée exploitable n’est trouvée.
    """
    if not experiences:
        return np.nan  # ⚠️ ne pas considérer comme 0 an

    durations = [parse_duration_to_years(exp) for exp in experiences]
    durations = [d for d in durations if pd.notna(d)]

    if not durations:
        return np.nan  # ⚠️ aucune info exploitable

    return round(sum(durations), 2)


def count_experiences(experiences):
    """Compte le nombre d'expériences professionnelles."""
    return len(experiences) if experiences else 0


def mean_past_rating(pastCourses):
    """Calcule la moyenne des notes passées."""
    if not pastCourses:
        return np.nan
    return np.mean([c.get("numberOfStars", np.nan) for c in pastCourses])


def flatten_entry(entry):
    """Aplati un professeur en plusieurs lignes (une par cours)."""
    flattened = []
    highest_level, highest_domain = extract_highest_degree_info(entry.get("diplomas"))
    total_years = total_experience_years(entry.get("experiences"))  # ✅ uniquement sur experiences

    for course in entry.get("pastCourses", []):
        flattened.append({
            "firstname": entry.get("fistname") or entry.get("firstname"),
            "lastname": entry.get("lastname"),
            "city": entry.get("city", "Unknown"),
            "description": entry.get("description", ""),
            "highest_degree": highest_level,
            "highest_degree_domain": highest_domain,
            "num_diplomas": len(entry.get("diplomas", [])),
            "num_experiences": count_experiences(entry.get("experiences")),
            "total_experience_years": total_years,  # 🆕 NaN si inconnu, pas 0
            "mean_past_rating": mean_past_rating(entry.get("pastCourses")),
            "course_title": course.get("title", "Unknown"),
            "course_rating": course.get("numberOfStars", np.nan)
        })
    return flattened


# --- 4. Charger et transformer les données ---
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for teacher in data:
    rows.extend(flatten_entry(teacher))

df = pd.DataFrame(rows)

# --- 5. Nettoyage rapide ---
df.dropna(subset=["course_rating"], inplace=True)
df.reset_index(drop=True, inplace=True)

# --- 6. Résumé + audit ---
print(f"Nombre total de cours : {len(df)}")
print(f"Nombre d'expériences non reconnues : {len(unrecognized_durations)}")

if unrecognized_durations:
    print("\n🔍 Exemple d’expérience non reconnue :")
    for e in unrecognized_durations[:3]:
        print(e)

print("\n✅ Exemple de lignes formatées :")
df.head(10)
