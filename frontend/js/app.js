const API_BASE = "http://localhost:8000";

const analyzeBtn = document.getElementById("analyzeBtn");
const complaintText = document.getElementById("complaintText");
const errorMsg = document.getElementById("errorMsg");
const resultCard = document.getElementById("resultCard");

analyzeBtn.addEventListener("click", async () => {
  const text = complaintText.value.trim();
  errorMsg.textContent = "";

  if (text.length < 5) {
    errorMsg.textContent = "Please enter a complaint of at least a few words.";
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const res = await fetch(`${API_BASE}/api/complaints`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (data.error) {
      errorMsg.textContent = data.error;
      return;
    }

    renderResult(data);
    // Stash the last result so similarity.html can read it.
    localStorage.setItem("lastComplaintResult", JSON.stringify({ text, ...data }));
  } catch (err) {
    errorMsg.textContent = "Could not reach the API. Is the backend running on port 8000?";
    console.error(err);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Complaint";
  }
});

function renderResult(data) {
  resultCard.style.display = "block";

  document.getElementById("resCategory").textContent = data.category || "-";
  document.getElementById("resDepartment").textContent = data.department || "-";

  const urgencyEl = document.getElementById("resUrgency");
  const urgencyLevel = data.urgency?.urgency || "-";
  urgencyEl.textContent = urgencyLevel;
  urgencyEl.className = `badge ${urgencyLevel}`;

  document.getElementById("resLocation").textContent =
    (data.entities?.locations || []).join(", ") || "Not detected";
  document.getElementById("resDuration").textContent =
    data.entities?.duration || "Not detected";
  document.getElementById("resSentiment").textContent = data.sentiment?.label || "-";

  document.getElementById("resSummary").textContent = data.summary || "-";
  document.getElementById("resKeyPhrases").textContent =
    (data.key_phrases || []).join(", ") || "-";
}
