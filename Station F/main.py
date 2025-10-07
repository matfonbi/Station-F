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
    print("📦 Champs reçus :", list(form_dict.keys()))


    # ----------- INFOS GÉNÉRALES DU PROF -----------
    nom = form_dict.get("nom", "").strip()
    prenom = form_dict.get("prenom", "").strip()
    ville = form_dict.get("ville", "").strip()
    description = form_dict.get("description", "").strip()

    # ----------- DIPLÔMES -----------
    degrees = []
    for key in form_dict:
        if key.startswith("degree_") and key.endswith("_level"):
            prefix = key[:-5]  # retire "_level"
            level = form_dict.get(f"{prefix}level", "").strip()
            field = form_dict.get(f"{prefix}field", "").strip()
            if level or field:
                degrees.append({
                    "level": level,
                    "field": field
                })
    print("📋 Diplomes :", degrees)

    if degrees:
        highest_degree = sorted(
            degrees,
            key=lambda d: ["Aucun", "Certificat", "Licence", "Master", "Doctorat"].index(d["level"])
        )[-1]["level"]
        highest_degree_domain = ", ".join(d["field"] for d in degrees if d["field"])
    else:
        highest_degree = None
        highest_degree_domain = None


    course_title = form_dict.get("course_title", "").strip()

    # ----------- COURS PRÉCÉDENTS -----------
    courses = []
    notes = []
    for key in form_dict:
        if key.startswith("course_") and key.endswith("_title") and key != "course_title":
            prefix = key[:-5]  # ex: "course_1_"
            title = form_dict.get(f"{prefix}title", "").strip()
            school = form_dict.get(f"{prefix}school", "").strip()
            rating = form_dict.get(f"{prefix}rating", "").strip()

            if title or school or rating:
                rating_val = None
                if rating:
                    try:
                        # ✅ Tolère les virgules ou points
                        rating_val = float(rating.replace(",", "."))
                        if 0 <= rating_val <= 5:
                            notes.append(rating_val)
                        else:
                            print(f"⚠️ Note invalide ignorée ({rating_val})")
                    except ValueError:
                        print(f"⚠️ Note non convertible : '{rating}'")

                courses.append({
                    "title": title,
                    "school": school,
                    "rating": rating_val
                })
    print("📋 Expériences reçues :", courses)

    # Calcul de la moyenne des notes
    mean_past_rating = round(sum(notes) / len(notes), 2) if notes else None
    print(f"📊 Moyenne calculée : {mean_past_rating}")

    # ----------- EXPÉRIENCES -----------
    experiences = []
    for key in form_dict:
        if key.startswith("experience_") and key.endswith("_description"):
            prefix1 = key[:-11]  # ex: "experience_1_"
            desc = form_dict.get(f"{prefix1}description", "").strip()
            if desc:
                experiences.append({
                    "description": desc
                })
        if key.startswith("experience_") and key.endswith("_duration"):
            prefix2 = key[:-8]  # ex: "experience_1_"
            duration = form_dict.get(f"{prefix2}duration", "").strip()
            if desc or duration:
                experiences.append({
                    "duration": duration
                })
    

    # ----------- CALCUL DU TEMPS TOTAL D’EXPÉRIENCE -----------
    import re

    def parse_duration(text):
        """
        Convertit une durée textuelle en années (float)
        Exemples : "2 ans" -> 2.0, "6 mois" -> 0.5, "18 mois" -> 1.5
        """
        if not text:
            return 0.0

        text = text.lower().replace(",", ".").strip()

        # Extraire le premier nombre (ex: "2", "1.5", "18")
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return 0.0

        value = float(match.group(1))

        # Vérifie si c’est en mois ou en années
        if "mois" in text:
            return round(value / 12, 2)
        elif "an" in text or "ans" in text or "année" in text:
            return round(value, 2)
        else:
            # Valeur par défaut : supposée en années
            return round(value, 2)

    # Calcul du total des années d’expérience
    total_experience_years = round(sum(parse_duration(e.get("duration", "")) for e in experiences),2)
    print(f"💼 Total d’expérience calculé : {total_experience_years} ans")


    # ----------- FEATURES POUR LE MODÈLE -----------
    infos = {
        "city": ville or None,
        "highest_degree": highest_degree,
        "highest_degree_domain": highest_degree_domain,
        "num_diplomas": len(degrees),
        "num_experiences": len(experiences) if experiences else None,
        "total_experience_years": total_experience_years if total_experience_years > 0 else None,
        "mean_past_rating": mean_past_rating,
        "description": description or "-",
        "course_title": course_title or "-",
    }

    # ----------- SÉLECTION DU MODÈLE -----------
    if mean_past_rating is not None and model_robuste:
        model = model_robuste
        model_used = "Robuste"
    else:
        model = model_prospectif
        model_used = "Prospectif"

    print(f"🎯 Modèle sélectionné : {model_used}")

    # ----------- PRÉDICTION -----------
    try:
        X = pd.DataFrame([infos])
        prediction = float(model.predict(X)[0]) if model else 3.5
    except Exception as e:
        print("⚠️ Erreur prédiction :", e)
        prediction = 3.7

    # ----------- ENREGISTREMENT HISTORIQUE -----------
    try:
        path_hist = os.path.join(DATA_DIR, "predictions.json")
        history = json.load(open(path_hist))
    except Exception:
        history = []

    history.append({
        "nom": nom,
        "prenom": prenom,
        "ville": ville,
        "highest_degree": highest_degree,
        "highest_degree_domain": highest_degree_domain,
        "num_diplomas": len(degrees),
        "cours": course_title,
        "mean_past_rating": mean_past_rating,
        "total_experience_years": total_experience_years if total_experience_years > 0 else None,
        "prediction": round(prediction, 2),
        "model_used": model_used,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    json.dump(history, open(path_hist, "w"), indent=2, ensure_ascii=False)

    # ----------- RENDU DU TEMPLATE -----------
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "prediction": round(prediction, 2),
        "model_used": model_used,
        "infos": {
            "nom": nom,
            "prenom": prenom,
            "ville": ville or "Non renseignée",
            "degrees": degrees,
            "description": description if description else "Aucune description fournie",
            "course_title": course_title or "Non renseignée",
            "mean_past_rating": mean_past_rating,
            "courses": courses,
            "experiences": experiences,
            "total_experience_years": total_experience_years if total_experience_years > 0 else None,
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
