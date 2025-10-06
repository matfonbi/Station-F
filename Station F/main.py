from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pyngrok import ngrok
import joblib
import pandas as pd
import json, os, threading, time, uvicorn

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

@app.post("/api/predict", response_class=HTMLResponse)
async def predict_submit(request: Request):
    """
    Traite le formulaire HTML et renvoie la note prédite + un récap complet.
    """
    form = await request.form()
    form_dict = dict(form)

    # ----------- INFOS GÉNÉRALES DU PROF -----------
    nom = form_dict.get("nom", "").strip()
    prenom = form_dict.get("prenom", "").strip()
    ville = form_dict.get("ville", "").strip()
    description = form_dict.get("description", "").strip()
    diplome_niveau = form_dict.get("diplome_niveau", "").strip()
    diplome_domaine = form_dict.get("diplome_domaine", "").strip()
    course_title = form_dict.get("course_title", "").strip()

    # ----------- COURS PRÉCÉDENTS -----------
    courses = []
    notes = []
    for key in form_dict:
        if key.startswith("course_") and key.endswith("_title"):
            prefix = key[:-6]  # ex: "course_1_"
            title = form_dict.get(f"{prefix}title", "").strip()
            school = form_dict.get(f"{prefix}school", "").strip()
            rating = form_dict.get(f"{prefix}rating", "")
            if title or school:
                try:
                    rating_val = float(rating)
                    if 0 <= rating_val <= 5:
                        notes.append(rating_val)
                except Exception:
                    rating_val = None
                courses.append({
                    "title": title,
                    "school": school,
                    "rating": rating_val
                })

    mean_past_rating = round(sum(notes) / len(notes), 2) if notes else None

    # ----------- EXPÉRIENCES -----------
    experiences = []
    for key in form_dict:
        if key.startswith("experience_") and key.endswith("_description"):
            prefix = key[:-12]  # ex: "experience_1_"
            desc = form_dict.get(f"{prefix}description", "").strip()
            duration = form_dict.get(f"{prefix}duration", "").strip()
            if desc or duration:
                experiences.append({
                    "description": desc,
                    "duration": duration
                })

    # ----------- FEATURES POUR LE MODÈLE -----------
    infos = {
        "city": ville or None,
        "highest_degree": diplome_niveau or None,
        "highest_degree_domain": diplome_domaine or None,
        "num_diplomas": None,
        "num_experiences": len(experiences) if experiences else None,
        "total_experience_years": None,
        "mean_past_rating": mean_past_rating,
        "description": description or None,
        "course_title": course_title or None,
    }

    # ----------- SÉLECTION DU MODÈLE -----------
    if mean_past_rating is not None and model_robuste:
        model = model_robuste
        model_used = "Robuste"
    else:
        model = model_prospectif
        model_used = "Prospectif"

    # ----------- PRÉDICTION -----------
    try:
        X = pd.DataFrame([infos])
        prediction = float(model.predict(X)[0]) if model else 3.5
    except Exception as e:
        print("⚠️ Erreur prédiction :", e)
        prediction = 3.7

    # ----------- RENDU TEMPLATE -----------
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "prediction": round(prediction, 2),
        "model_used": model_used,
        "infos": {
            "nom": nom,
            "prenom": prenom,
            "ville": ville or "Non renseignée",
            "diplome_niveau": diplome_niveau,
            "diplome_domaine": diplome_domaine,
            "description": description or "—",
            "course_title": course_title or "—",
            "mean_past_rating": mean_past_rating,
            "courses": courses,
            "experiences": experiences
        }
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
