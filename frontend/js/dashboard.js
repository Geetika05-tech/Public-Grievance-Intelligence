const API_BASE = "http://localhost:8000";
let categoryChart = null;

async function loadDashboard() {
  try {
    const res = await fetch(`${API_BASE}/api/dashboard`);
    const data = await res.json();

    document.getElementById("statTotal").textContent = data.total_complaints ?? 0;
    document.getElementById("statHigh").textContent = data.by_urgency?.High ?? 0;
    document.getElementById("statMedium").textContent = data.by_urgency?.Medium ?? 0;
    document.getElementById("statLow").textContent = data.by_urgency?.Low ?? 0;

    renderCategoryChart(data.by_category || {});
  } catch (err) {
    console.error("Could not load dashboard:", err);
  }
}

function renderCategoryChart(byCategory) {
  const ctx = document.getElementById("categoryChart");
  const labels = Object.keys(byCategory);
  const values = Object.values(byCategory);

  if (categoryChart) categoryChart.destroy();

  categoryChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels.length ? labels : ["No data yet"],
      datasets: [{
        label: "Complaints",
        data: values.length ? values : [0],
        backgroundColor: "#38bdf8",
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#334155" }, beginAtZero: true },
      },
    },
  });
}

async function loadEmergingIssues() {
  const listEl = document.getElementById("issuesList");
  try {
    const res = await fetch(`${API_BASE}/api/emerging-issues`);
    const data = await res.json();
    const issues = data.issues || [];

    if (issues.length === 0) {
      listEl.innerHTML = '<div class="empty">No emerging issues detected yet.</div>';
      return;
    }

    listEl.innerHTML = issues
      .map(
        (i) => `
        <div class="issue">
          <span>⚠ ${i.issue} — ${i.current_count} complaints (${i.current_week})</span>
          <strong>${i.growth_pct !== null ? "+" + i.growth_pct + "%" : "new"}</strong>
        </div>`
      )
      .join("");
  } catch (err) {
    listEl.innerHTML = '<div class="error">Could not reach the API.</div>';
  }
}

document.getElementById("refreshIssuesBtn").addEventListener("click", loadEmergingIssues);

document.getElementById("reclusterBtn").addEventListener("click", async () => {
  const statusEl = document.getElementById("clusterStatus");
  statusEl.textContent = "Running clustering...";
  try {
    const res = await fetch(`${API_BASE}/api/cluster`, { method: "POST" });
    const data = await res.json();
    if (data.error) {
      statusEl.textContent = data.error;
    } else {
      statusEl.textContent = "Done — clusters: " + Object.values(data.clusters).join(", ");
      loadEmergingIssues();
    }
  } catch (err) {
    statusEl.textContent = "Could not reach the API.";
  }
});

loadDashboard();
loadEmergingIssues();
