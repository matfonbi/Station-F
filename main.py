from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pyngrok import ngrok
import joblib
import numpy as np
import pandas as pd
import json, os, threading, time, uvicorn
from typing import Dict, Any, Tuple, List

from feature_engineering import (
    build_features_from_entities,
    normalize_degree_level,  # re-exported for completeness if needed elsewhere
    parse_duration_to_years,  # kept for potential future form enhancements
    _safe_float,
)

# ---------------------------------------------------------
# ⚙️ CONFIGURATION GLOBALE
# ---------------------------------------------------------
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "ratings.json")

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(DATA_PATH):
    json.dump([], open(DATA_PATH, "w"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---------------------------------------------------------
# 🤖 CHARGEMENT DES MODÈLES
# ---------------------------------------------------------
try:
    model_prospectif = joblib.load(os.path.join(BASE_DIR, "models", "model_prospectif.pkl"))
    model_robuste = joblib.load(os.path.join(BASE_DIR, "models", "model_robust.pkl"))
    print("✅ Modèles Prospectif et Robuste chargés avec succès.")
except Exception as e:
    model_prospectif = None
    model_robuste = None
    print("⚠️ Impossible de charger les modèles :", e)

def run_prediction(features: Dict[str, Any]) -> Tuple[float, str]:
    mean_rating = features.get("mean_past_rating")
    has_history = mean_rating is not None and not pd.isna(mean_rating)

    if model_robuste and has_history:
        model = model_robuste
        model_used = "Robuste"
    elif model_prospectif:
        model = model_prospectif
        model_used = "Prospectif"
    elif model_robuste:
        model = model_robuste
        model_used = "Robuste"
    else:
        raise RuntimeError("Aucun modèle n'est disponible pour la prédiction.")

    X = pd.DataFrame([features])
    prediction = float(model.predict(X)[0])
    return prediction, model_used


def build_payload_from_form(form_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    nom = form_dict.get("nom", "").strip()
    prenom = form_dict.get("prenom", "").strip()
    ville = form_dict.get("ville", "").strip()
    description = form_dict.get("description", "").strip()

    degree_map: Dict[str, Dict[str, str]] = {}
    experience_map: Dict[str, Dict[str, str]] = {}
    course_map: Dict[str, Dict[str, str]] = {}

    for key, value in form_dict.items():
        if not isinstance(value, str):
            continue
        clean_value = value.strip()

        if key.startswith("degree_"):
            parts = key.split("_", 2)
            if len(parts) == 3:
                idx = parts[1]
                field = parts[2]
                degree_map.setdefault(idx, {})[field] = clean_value
        elif key.startswith("experience_"):
            parts = key.split("_", 2)
            if len(parts) == 3:
                idx = parts[1]
                field = parts[2]
                experience_map.setdefault(idx, {})[field] = clean_value
        elif key.startswith("course_"):
            parts = key.split("_", 2)
            if len(parts) == 3 and parts[1].isdigit():
                idx = parts[1]
                field = parts[2]
                course_map.setdefault(idx, {})[field] = clean_value

    def sorted_indices(data: Dict[str, Any]) -> List[str]:
        return sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else x)

    diplomas = []
    for idx in sorted_indices(degree_map):
        entry = degree_map[idx]
        level = entry.get("level", "").strip()
        field = entry.get("field", "").strip()
        if level or field:
            diplomas.append({
                "level": level,
                "title": field
            })

    experiences = []
    for idx in sorted_indices(experience_map):
        entry = experience_map[idx]
        desc = entry.get("description", "").strip()
        duration = entry.get("duration", "").strip()
        if desc or duration:
            experiences.append({
                "description": desc,
                "duration": duration
            })

    past_courses = []
    for idx in sorted_indices(course_map):
        entry = course_map[idx]
        title = entry.get("title", "").strip()
        school = entry.get("school", "").strip()
        rating_raw = entry.get("rating", "").strip()
        rating = _safe_float(rating_raw) if rating_raw else None
        if title or school or rating is not None:
            past_courses.append({
                "title": title,
                "description": school or None,
                "numberOfStars": rating
            })

    professor_payload = {
        "fistname": prenom,
        "firstname": prenom,
        "lastname": nom,
        "city": ville,
        "description": description,
        "diplomas": diplomas,
        "experiences": experiences,
        "pastCourses": past_courses
    }
    course_payload = {
        "title": form_dict.get("course_title", "").strip(),
        "description": form_dict.get("course_description", "").strip()
    }
    return professor_payload, course_payload

# ---------------------------------------------------------
# 🏠 PAGE D’ACCUEIL
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------------------------------------------------
# 🔮 PAGE /api/predict — prédiction avec IA
# ---------------------------------------------------------
@app.get("/api/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Affiche la page de prédiction."""
    return templates.TemplateResponse("predict.html", {"request": request, "prediction": None})

@app.post("/api/predict")
async def predict_submit(request: Request):
    """
    Traite le formulaire HTML et renvoie la note prédite + un récap complet.
    """
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Corps JSON invalide: {exc}") from exc

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Le corps de la requête doit être un objet JSON.")

        data = payload.get("data") if "data" in payload and isinstance(payload.get("data"), dict) else payload
        professor_payload = data.get("professor") if isinstance(data.get("professor"), dict) else {}
        course_payload = data.get("course") if isinstance(data.get("course"), dict) else {}

        features, _details = build_features_from_entities(professor_payload, course_payload)
        try:
            prediction, _ = run_prediction(features)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Erreur pendant la prédiction: {exc}") from exc

        return {"gradeAverage": round(prediction, 2)}

    form = await request.form()
    form_dict = dict(form)
    print("📦 Champs reçus :", list(form_dict.keys()))

    professor_payload, course_payload = build_payload_from_form(form_dict)
    features, details = build_features_from_entities(professor_payload, course_payload)

    try:
        prediction, model_used = run_prediction(features)
    except Exception as e:
        print("⚠️ Erreur prédiction :", e)
        prediction = 3.7
        mean_rating = features.get("mean_past_rating")
        model_used = "Robuste" if mean_rating is not None and not pd.isna(mean_rating) else "Prospectif"

    try:
        path_hist = os.path.join(DATA_DIR, "predictions.json")
        history = json.load(open(path_hist))
    except Exception:
        history = []

    history.append({
        "nom": professor_payload.get("lastname"),
        "prenom": professor_payload.get("fistname"),
        "ville": professor_payload.get("city"),
        "highest_degree": features.get("highest_degree"),
        "highest_degree_domain": features.get("highest_degree_domain"),
        "num_diplomas": features.get("num_diplomas"),
        "num_experiences": features.get("num_experiences"),
        "cours": details.get("course_title"),
        "mean_past_rating": None if pd.isna(features.get("mean_past_rating")) else round(float(features.get("mean_past_rating")), 2),
        "total_experience_years": None if pd.isna(features.get("total_experience_years")) else float(features.get("total_experience_years")),
        "prediction": round(prediction, 2),
        "model_used": model_used,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    json.dump(history, open(path_hist, "w"), indent=2, ensure_ascii=False)

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "prediction": round(prediction, 2),
        "model_used": model_used,
        "infos": details
    })


# ---------------------------------------------------------
# ⭐ PAGE /rate-form — notation professeur
# ---------------------------------------------------------
@app.get("/rate-form", response_class=HTMLResponse)
async def rate_form(request: Request):
    return templates.TemplateResponse("rate.html", {"request": request})

@app.post("/rate-form", response_class=HTMLResponse)
async def rate_submit(
    request: Request,
    nom: str = Form(...),
    prenom: str = Form(""),
    cours: str = Form(...),
    note: float = Form(...),
    commentaire: str = Form("")
):
    new_rating = {
        "nom": nom.strip(),
        "prenom": prenom.strip(),
        "cours": cours.strip(),
        "note": float(note),
        "commentaire": commentaire.strip() or None
    }

    try:
        data = json.load(open(DATA_PATH))
    except Exception:
        data = []

    data.append(new_rating)
    json.dump(data, open(DATA_PATH, "w"), indent=2, ensure_ascii=False)

    return RedirectResponse("/", status_code=303)

# ---------------------------------------------------------
# 🔁 REDIRECTION /predict-form → /api/predict
# ---------------------------------------------------------
@app.get("/predict-form")
async def redirect_predict():
    return RedirectResponse("/api/predict")

# ---------------------------------------------------------
# 📜 HISTORIQUE DES PRÉDICTIONS
# ---------------------------------------------------------
@app.get("/history", response_class=HTMLResponse)
async def show_history(request: Request):
    """Affiche l'historique des prédictions IA."""
    try:
        data = json.load(open(os.path.join(DATA_DIR, "predictions.json")))
    except Exception:
        data = []
    data = list(reversed(data))  # plus récentes d'abord
    return templates.TemplateResponse("history.html", {"request": request, "predictions": data})


# ---------------------------------------------------------
# 🚀 LANCEMENT AUTOMATIQUE AVEC NGROK
# ---------------------------------------------------------
if __name__ == "__main__":
    port = 8000

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

    # Lancer FastAPI dans un thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Attendre le démarrage
    time.sleep(2)

    # Lancer ngrok
    public_url = ngrok.connect(port).public_url
    print("\n" + "=" * 60)
    print(f"🌍 Site public       : {public_url}")
    print(f"🔮 Page prédiction   : {public_url}/api/predict")
    print(f"⭐ Page notation      : {public_url}/rate-form")
    print("=" * 60)
    print("✅ Application en ligne — CTRL+C pour arrêter.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ngrok.kill()
        print("\n👋 Fermeture du serveur.")
