const essayInput = document.getElementById("essayInput");
const scoreButton = document.getElementById("scoreButton");
const statusPill = document.getElementById("statusPill");
const wordCount = document.getElementById("wordCount");
const resultModal = document.getElementById("resultModal");
const closeModal = document.getElementById("closeModal");
const totalScore = document.getElementById("totalScore");
const ruleId = document.getElementById("ruleId");
const dimensionGrid = document.getElementById("dimensionGrid");

function setStatus(text, mode = "") {
  statusPill.textContent = text;
  statusPill.className = `score-pill ${mode}`.trim();
}

function updateWordCount() {
  const text = essayInput.value.trim();
  const count = text ? text.split(/\s+/).filter(Boolean).length : 0;
  wordCount.textContent = `${count} ${count === 1 ? "word" : "words"}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderList(items) {
  const values = Array.isArray(items) ? items : [];
  if (!values.length) return "<p>No details returned.</p>";
  return `<ul>${values.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function renderResults(data) {
  totalScore.textContent = `${data.total_score ?? "-"} / 12`;
  ruleId.textContent = data.rule_id ? `Rule ${data.rule_id}` : "Rule -";
  dimensionGrid.innerHTML = (data.dimension_reports || [])
    .map((item) => {
      const score = Number(item.score || 0);
      const width = Math.max(0, Math.min(100, (score / 3) * 100));
      return `
        <article class="dimension-card">
          <div class="dimension-title-row">
            <h3 class="dimension-title">${escapeHtml(item.dimension)}</h3>
            <div class="dimension-score">${score} / 3</div>
          </div>
          <div class="bar"><span style="width: ${width}%"></span></div>
          <section>
            <h4>Analysis</h4>
            <p>${escapeHtml(item.analysis)}</p>
          </section>
          <section>
            <h4>Evidence</h4>
            ${renderList(item.evidence)}
          </section>
          <section>
            <h4>Suggestions</h4>
            ${renderList(item.suggestions)}
          </section>
        </article>
      `;
    })
    .join("");
  resultModal.classList.remove("hidden");
  resultModal.setAttribute("aria-hidden", "false");
}

async function scoreEssay() {
  const essay = essayInput.value.trim();
  if (!essay) {
    setStatus("Essay required", "error");
    essayInput.focus();
    return;
  }

  scoreButton.disabled = true;
  setStatus("Scoring", "busy");

  try {
    const response = await fetch("/api/v1/essay/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ essay }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || "Scoring failed.");
    }
    renderResults(payload);
    setStatus("Complete");
  } catch (error) {
    setStatus("Failed", "error");
    totalScore.textContent = "- / 12";
    ruleId.textContent = "Rule -";
    dimensionGrid.innerHTML = `
      <article class="dimension-card">
        <h3 class="dimension-title">Scoring Error</h3>
        <p>${escapeHtml(error.message || "Scoring failed.")}</p>
      </article>
    `;
    resultModal.classList.remove("hidden");
    resultModal.setAttribute("aria-hidden", "false");
  } finally {
    scoreButton.disabled = false;
  }
}

function closeResults() {
  resultModal.classList.add("hidden");
  resultModal.setAttribute("aria-hidden", "true");
}

essayInput.addEventListener("input", updateWordCount);
scoreButton.addEventListener("click", scoreEssay);
closeModal.addEventListener("click", closeResults);
resultModal.addEventListener("click", (event) => {
  if (event.target === resultModal) closeResults();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !resultModal.classList.contains("hidden")) {
    closeResults();
  }
});

updateWordCount();
