// -----------------------------
// 🎓 EduPredict — script global (corrigé)
// -----------------------------
document.addEventListener("DOMContentLoaded", () => {
  console.log("✅ Script JS chargé une seule fois");

  // ----- Menu mobile -----
  const menuToggle = document.getElementById("menuToggle");
  const navLinks = document.getElementById("navLinks");
  if (menuToggle && navLinks) {
    menuToggle.addEventListener("click", () => {
      navLinks.classList.toggle("active");
      menuToggle.classList.toggle("open");
    });
  }

  // ----- Delegation: suppression d’un bloc -----
  document.body.addEventListener("click", (e) => {
    if (e.target.classList.contains("remove-button")) {
      const group = e.target.closest(".course-group, .experience-group, .degree-group");
      if (group) group.remove();
    }
  });

  // ----- Fonction utilitaire pour créer les blocs dynamiques -----
  const createBlock = (type, index) => {
    if (type === "course") {
      return `
        <div class="course-group">
          <button type="button" class="remove-button" title="Supprimer ce cours">❌</button>
          <label>Nom du cours :</label>
          <input name="course_${index}_title" placeholder="Nom du cours">
          <label>Établissement :</label>
          <input name="course_${index}_school" placeholder="Établissement">
          <label>Note reçue :</label>
          <input name="course_${index}_rating" type="number" step="0.1" min="0" max="5" placeholder="4.5">
        </div>`;
    }
    if (type === "experience") {
      return `
        <div class="experience-group">
          <button type="button" class="remove-button" title="Supprimer cette expérience">❌</button>
          <label>Expérience :</label>
          <input name="experience_${index}_description" placeholder="Nouvelle expérience">
          <label>Durée :</label>
          <input name="experience_${index}_duration" placeholder="2 ans, 6 mois, 1 an et demi...">
        </div>`;
    }
    if (type === "degree") {
      return `
        <div class="degree-group">
          <button type="button" class="remove-button" title="Supprimer ce diplôme">❌</button>
          <label>Niveau :</label>
          <select name="degree_${index}_level">
            <option value="">-- Sélectionner un niveau --</option>
            <option value="Aucun">Aucun</option>
            <option value="Certificat">Certificat</option>
            <option value="Licence">Licence</option>
            <option value="Master">Master</option>
            <option value="Doctorat">Doctorat</option>
          </select>
          <label>Domaine :</label>
          <input name="degree_${index}_field" placeholder="Informatique, Mathématiques...">
        </div>`;
    }
  };

  // ----- Delegation: ajout dynamique -----
  document.body.addEventListener("click", (e) => {
    const id = e.target.id;
    if (id === "addCourse" || id === "addExperience" || id === "addDegree") {
      const type = id.replace("add", "").toLowerCase(); // "course" / "experience" / "degree"
      const container = document.getElementById(`${type}-container`);
      if (!container) return;

      const nextIndex = container.querySelectorAll(`.${type}-group`).length + 1;
      container.insertAdjacentHTML("beforeend", createBlock(type, nextIndex));
    }
  });
});
