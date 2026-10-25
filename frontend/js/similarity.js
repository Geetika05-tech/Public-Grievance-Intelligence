const stored = localStorage.getItem("lastComplaintResult");

const yourComplaintEl = document.getElementById("yourComplaint");
const similarListEl = document.getElementById("similarList");
const clusterNameEl = document.getElementById("clusterName");
const relatedCountEl = document.getElementById("relatedCount");

if (stored) {
  const data = JSON.parse(stored);
  yourComplaintEl.textContent = `"${data.text}"`;

  const similar = data.similar_complaints || [];
  if (similar.length === 0) {
    similarListEl.innerHTML = '<div class="empty">No similar complaints on file yet — submit a few more to build history.</div>';
  } else {
    similarListEl.innerHTML = similar
      .map(
        (s) => `
        <div class="similar-item">
          <span>#${s.index} — ${s.text}</span>
          <strong>${Math.round(s.similarity * 100)}%</strong>
        </div>`
      )
      .join("");
  }

  clusterNameEl.textContent = data.category || "-";
  relatedCountEl.textContent = similar.length;
} else {
  similarListEl.innerHTML = '<div class="empty">Submit a complaint on the first screen first.</div>';
}
