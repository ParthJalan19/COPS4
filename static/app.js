// State
let cameraStream = null;
let isProcessing = false;
let processInterval = null;
let isScanPaused = false; // When a match is found, scan stops and holds the result

// DOM Elements
const webcam = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const placeholder = document.getElementById("camera-placeholder");
const btnStartCamera = document.getElementById("btn-start-camera");
const btnToggleCamera = document.getElementById("btn-toggle-camera");
const btnResumeScan = document.getElementById("btn-resume-scan");
const scanHoldBanner = document.getElementById("scan-hold-banner");
const chkForceScan = document.getElementById("chk-force-scan");

const currentDbName = document.getElementById("current-db-name");
const motionBadge = document.getElementById("motion-badge");
const motionText = document.getElementById("motion-text");
const statMotionVal = document.getElementById("stat-motion-val");

const matchAlertBox = document.getElementById("match-alert-box");
const matchName = document.getElementById("match-name");
const matchInfo = document.getElementById("match-info");
const matchConfidence = document.getElementById("match-confidence");

const dbStatusBadge = document.getElementById("db-status-badge");
const dbStatusText = document.getElementById("db-status-text");
const dbWarningBanner = document.getElementById("db-warning-banner");
const dbWarningText = document.getElementById("db-warning-text");

const uploadForm = document.getElementById("upload-form");
const personName = document.getElementById("person-name");
const personDetails = document.getElementById("person-details");
const personFile = document.getElementById("person-file");
const dropzoneContent = document.getElementById("dropzone-content");
const imagePreview = document.getElementById("image-preview");
const btnSubmitText = document.getElementById("btn-submit-text");

const personsCount = document.getElementById("persons-count");
const personsTbody = document.getElementById("persons-tbody");
const btnRefreshPersons = document.getElementById("btn-refresh-persons");

const logsTbody = document.getElementById("logs-tbody");
const btnRefreshAlerts = document.getElementById("btn-refresh-alerts");
const btnClearLogs = document.getElementById("btn-clear-logs");

// Modal Popup Elements
const alertModal = document.getElementById("alert-modal");
const btnModalClose = document.getElementById("btn-modal-close");
const btnModalResume = document.getElementById("btn-modal-resume");
const modalLiveImg = document.getElementById("modal-live-img");
const modalRegisteredImg = document.getElementById("modal-registered-img");
const modalPersonName = document.getElementById("modal-person-name");
const modalConfidence = document.getElementById("modal-confidence");
const modalDetails = document.getElementById("modal-details");

const btnDismissDbBanner = document.getElementById("btn-dismiss-db-banner");
if (btnDismissDbBanner) {
  btnDismissDbBanner.addEventListener("click", () => {
    dbWarningBanner.classList.add("hidden");
    sessionStorage.setItem("dismissed_db_banner", "true");
  });
}

// Check Database Status
async function checkStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const db = data.database;

    if (currentDbName) {
      currentDbName.textContent = data.database_name || "cops_new";
    }

    if (db.connected && db.provider.includes("MongoDB Atlas")) {
      dbStatusBadge.className = "status-pill status-connected";
      dbStatusText.textContent = `MongoDB Atlas (${data.database_name})`;
      dbWarningBanner.classList.add("hidden");
    } else {
      dbStatusBadge.className = "status-pill status-connected";
      dbStatusText.textContent = `Active: ${data.database_name}.watchlist`;
      if (sessionStorage.getItem("dismissed_db_banner") !== "true") {
        dbWarningBanner.classList.remove("hidden");
        dbWarningText.innerHTML = `<strong>Local Storage Active:</strong> Table <code>${data.database_name}.watchlist</code> is running locally. To connect to MongoDB Atlas Cloud, verify your user password in <code>.env</code>.`;
      } else {
        dbWarningBanner.classList.add("hidden");
      }
    }
  } catch (err) {
    dbStatusBadge.className = "status-pill status-warning";
    dbStatusText.textContent = "Server Offline";
  }
}

// Camera Control
async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });
    webcam.srcObject = cameraStream;
    placeholder.style.display = "none";
    btnToggleCamera.textContent = "Stop Camera";
    btnToggleCamera.disabled = false;
    isScanPaused = false;
    scanHoldBanner.classList.add("hidden");
    btnResumeScan.classList.add("hidden");

    webcam.onloadedmetadata = () => {
      overlay.width = webcam.videoWidth;
      overlay.height = webcam.videoHeight;
      startProcessingLoop();
    };
  } catch (err) {
    alert("Could not access webcam: " + err.message);
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  if (processInterval) {
    clearInterval(processInterval);
    processInterval = null;
  }
  webcam.srcObject = null;
  placeholder.style.display = "flex";
  btnToggleCamera.textContent = "Start Camera";
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  matchAlertBox.classList.add("hidden");
  scanHoldBanner.classList.add("hidden");
  btnResumeScan.classList.add("hidden");
  motionBadge.className = "motion-pill motion-idle";
  motionText.textContent = "Camera Off";
  isScanPaused = false;
}

function resumeScan() {
  isScanPaused = false;
  alertModal.classList.add("hidden");
  scanHoldBanner.classList.add("hidden");
  btnResumeScan.classList.add("hidden");
  motionBadge.className = "motion-pill motion-idle";
  motionText.textContent = "Scanning Resumed";
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

btnStartCamera.addEventListener("click", startCamera);
btnToggleCamera.addEventListener("click", () => {
  if (cameraStream) stopCamera();
  else startCamera();
});
btnResumeScan.addEventListener("click", resumeScan);
btnModalResume.addEventListener("click", resumeScan);
btnModalClose.addEventListener("click", () => {
  alertModal.classList.add("hidden");
});

// Real-time Detection Loop
function startProcessingLoop() {
  if (processInterval) clearInterval(processInterval);

  const offCanvas = document.createElement("canvas");
  const offCtx = offCanvas.getContext("2d");

  processInterval = setInterval(async () => {
    // If scan is paused because a target has been identified, HOLD the scan and do not send more requests
    if (!cameraStream || isProcessing || isScanPaused || webcam.videoWidth === 0) return;

    isProcessing = true;
    try {
      offCanvas.width = webcam.videoWidth;
      offCanvas.height = webcam.videoHeight;
      offCtx.drawImage(webcam, 0, 0, offCanvas.width, offCanvas.height);

      const dataUrl = offCanvas.toDataURL("image/jpeg", 0.75);

      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: dataUrl,
          force_recognition: chkForceScan.checked
        })
      });

      if (!res.ok) throw new Error("Detection failed");
      const result = await res.json();
      renderDetectionResults(result, dataUrl);
    } catch (err) {
      console.error("Frame processing error:", err);
    } finally {
      isProcessing = false;
    }
  }, 250);
}

// Render Results & Trigger Popup Alert
function renderDetectionResults(data, capturedDataUrl) {
  // Motion Pill
  if (data.motion_detected) {
    motionBadge.className = "motion-pill motion-active";
    motionText.textContent = "Motion Detected - Scanning";
  } else {
    motionBadge.className = "motion-pill motion-idle";
    motionText.textContent = chkForceScan.checked ? "Continuous Scan Active" : "Waiting for Movement";
  }

  statMotionVal.textContent = Math.round(data.motion_score || 0) + " px";

  ctx.clearRect(0, 0, overlay.width, overlay.height);

  let identifiedTarget = null;

  if (data.faces && data.faces.length > 0) {
    for (const face of data.faces) {
      const [x, y, w, h] = face.bbox;
      const isMatch = face.is_match;

      // Adjust mirrored coordinates
      const mirroredX = overlay.width - x - w;

      ctx.save();
      ctx.lineWidth = 3;
      ctx.strokeStyle = isMatch ? "#f85149" : "#d29922";
      ctx.fillStyle = isMatch ? "#da3633" : "#d29922";

      // Draw bounding box
      ctx.strokeRect(mirroredX, y, w, h);

      const labelText = isMatch
        ? `IDENTIFIED: ${face.name} (Similar Face: ${face.confidence}%)`
        : "Unrecognized Person";

      ctx.font = "bold 13px Inter, sans-serif";
      const textWidth = ctx.measureText(labelText).width;
      const tagY = y > 26 ? y - 26 : y;

      ctx.fillRect(mirroredX, tagY, textWidth + 16, 24);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(labelText, mirroredX + 8, tagY + 17);
      ctx.restore();

      if (isMatch) {
        identifiedTarget = face;
      }
    }
  }

  // When a person with similar face structure is identified:
  if (identifiedTarget) {
    // 1. Stop the scan and show the scan part continuously
    isScanPaused = true;
    scanHoldBanner.classList.remove("hidden");
    btnResumeScan.classList.remove("hidden");

    motionBadge.className = "motion-pill motion-active";
    motionText.textContent = "🎯 TARGET IDENTIFIED - SCAN HELD";

    // 2. Update Banner
    matchName.textContent = identifiedTarget.name;
    matchInfo.textContent = identifiedTarget.info || identifiedTarget.details || "Similar Face Structure Matched";
    matchConfidence.textContent = identifiedTarget.confidence + "%";
    matchAlertBox.classList.remove("hidden");

    // 3. Show Popup Alert Modal (No noise, pure visual modal)
    modalLiveImg.src = identifiedTarget.snapshot || capturedDataUrl;
    modalRegisteredImg.src = identifiedTarget.registered_image || "/static/placeholder.png";
    modalPersonName.textContent = identifiedTarget.name;
    modalConfidence.textContent = `${identifiedTarget.confidence}% Facial Similarity`;
    modalDetails.textContent = identifiedTarget.info || identifiedTarget.details || "Watchlist Target";
    alertModal.classList.remove("hidden");

    // Refresh alerts table
    loadAlerts();
  } else {
    matchAlertBox.classList.add("hidden");
  }
}

// File dropzone preview
personFile.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      imagePreview.src = event.target.result;
      imagePreview.classList.remove("hidden");
      dropzoneContent.style.display = "none";
    };
    reader.readAsDataURL(file);
  }
});

// Upload Player to Watchlist Form
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const name = personName.value.trim();
  const info = personDetails.value.trim();
  const file = personFile.files[0];

  if (!name || !file) {
    alert("Please provide both player name and a clear photo.");
    return;
  }

  btnSubmitText.textContent = "Extracting biometrics & adding to watchlist...";
  uploadForm.querySelector("button[type='submit']").disabled = true;

  const formData = new FormData();
  formData.append("name", name);
  formData.append("info", info);
  formData.append("file", file);

  try {
    const res = await fetch("/api/watchlist", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Failed to add player to watchlist");
    }

    alert(`Player '${name}' successfully added to cops_new.watchlist table!`);
    uploadForm.reset();
    imagePreview.classList.add("hidden");
    dropzoneContent.style.display = "block";
    loadWatchlist();
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    btnSubmitText.textContent = "Add Player to Watchlist Table";
    uploadForm.querySelector("button[type='submit']").disabled = false;
  }
});

// Load Watchlist Table
async function loadWatchlist() {
  try {
    const res = await fetch("/api/watchlist");
    const players = await res.json();

    personsCount.textContent = players.length;

    if (!players || players.length === 0) {
      personsTbody.innerHTML = `
        <tr>
          <td colspan="5" class="table-empty">No players on watchlist yet. Add someone above.</td>
        </tr>
      `;
      return;
    }

    personsTbody.innerHTML = players.map(p => `
      <tr>
        <td>
          <img src="${p.image || '/static/placeholder.png'}" class="table-thumb" alt="${escapeHtml(p.name)}">
        </td>
        <td class="name-cell">${escapeHtml(p.name)}</td>
        <td class="details-cell">${escapeHtml(p.info || '--')}</td>
        <td><span class="badge-watched">WATCHED</span></td>
        <td>
          <button class="btn btn-danger" onclick="deleteFromWatchlist('${p.id}')">Remove</button>
        </td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("Failed to load watchlist:", err);
  }
}

// Delete Player from Watchlist Table
window.deleteFromWatchlist = async function(id) {
  if (!confirm("Are you sure you want to remove this player from the watchlist table?")) return;

  try {
    const res = await fetch(`/api/watchlist/${id}`, { method: "DELETE" });
    if (res.ok) {
      loadWatchlist();
    } else {
      alert("Failed to remove player");
    }
  } catch (err) {
    alert("Error: " + err.message);
  }
};

// Load Generated Alerts Table
async function loadAlerts() {
  try {
    const res = await fetch("/api/alerts?limit=25");
    const alerts = await res.json();

    if (!alerts || alerts.length === 0) {
      logsTbody.innerHTML = `
        <tr>
          <td colspan="6" class="table-empty">No alerts triggered yet. Movement + watched player detection will generate alerts here.</td>
        </tr>
      `;
      return;
    }

    logsTbody.innerHTML = alerts.map(a => `
      <tr>
        <td class="time-cell">${escapeHtml(a.timestamp)}</td>
        <td>
          ${a.snapshot ? `<img src="${a.snapshot}" class="table-thumb" alt="snap">` : '--'}
        </td>
        <td class="name-cell" style="color: #ff7b72;">🚨 ${escapeHtml(a.player_name)}</td>
        <td><span class="badge-confidence">${a.confidence}%</span></td>
        <td class="details-cell">${escapeHtml(a.info || '--')}</td>
        <td><span class="badge-alert">WATCHLIST ALERT</span></td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("Failed to load alerts:", err);
  }
}

// Clear Alerts
btnClearLogs.addEventListener("click", async () => {
  if (!confirm("Clear all generated alerts from table?")) return;
  try {
    await fetch("/api/alerts", { method: "DELETE" });
    loadAlerts();
  } catch (e) {
    alert("Error clearing alerts: " + e.message);
  }
});

btnRefreshPersons.addEventListener("click", loadWatchlist);
btnRefreshAlerts.addEventListener("click", loadAlerts);

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Initialize
checkStatus();
loadWatchlist();
loadAlerts();
setInterval(checkStatus, 15000);
