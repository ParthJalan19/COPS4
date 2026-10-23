// Surveillance Wall State
let activeCameras = {}; // camera_id -> camera object
let currentFocusCam = null;
let currentGridLayout = "2x2";
let networkIp = "127.0.0.1";
let serverPort = 8000;
let currentWeaponThreshold = 0.18;

// DOM Elements
const cameraWallGrid = document.getElementById("camera-wall-grid");
const activeCamsCount = document.getElementById("active-cams-count");
const gridBtns = document.querySelectorAll(".grid-btn");

const matchAlertBox = document.getElementById("match-alert-box");
const matchName = document.getElementById("match-name");
const matchCamName = document.getElementById("match-cam-name");
const matchInfo = document.getElementById("match-info");
const matchConfidence = document.getElementById("match-confidence");

const dbStatusBadge = document.getElementById("db-status-badge");
const dbStatusText = document.getElementById("db-status-text");
const dbWarningBanner = document.getElementById("db-warning-banner");
const dbWarningText = document.getElementById("db-warning-text");
const btnDismissDbBanner = document.getElementById("btn-dismiss-db-banner");

// Modals
const alertModal = document.getElementById("alert-modal");
const btnModalClose = document.getElementById("btn-modal-close");
const btnModalResume = document.getElementById("btn-modal-resume");
const modalLiveImg = document.getElementById("modal-live-img");
const modalRegisteredImg = document.getElementById("modal-registered-img");
const modalPersonName = document.getElementById("modal-person-name");
const modalCameraName = document.getElementById("modal-camera-name");
const modalConfidence = document.getElementById("modal-confidence");
const modalDetails = document.getElementById("modal-details");

// Weapon Alert Modal
const weaponAlertModal = document.getElementById("weapon-alert-modal");
const modalWeaponImg = document.getElementById("modal-weapon-img");
const modalWeaponType = document.getElementById("modal-weapon-type");
const modalWeaponConfidence = document.getElementById("modal-weapon-confidence");
const modalWeaponCam = document.getElementById("modal-weapon-cam");
const btnCloseWeaponModal = document.getElementById("btn-close-weapon-modal");
const btnResumeWeapon = document.getElementById("btn-resume-weapon");

const addCameraModal = document.getElementById("add-camera-modal");
const btnOpenAddCamera = document.getElementById("btn-open-add-camera");
const btnCloseAddCamera = document.getElementById("btn-close-add-camera");
const tabBtns = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

const inputMobileUrl = document.getElementById("input-mobile-url");
const mobileQrCode = document.getElementById("mobile-qr-code");
const mobileCamNameInput = document.getElementById("mobile-cam-name");
const cctvNameInput = document.getElementById("cctv-name");
const cctvUrlInput = document.getElementById("cctv-url");
const selectWebcamDevice = document.getElementById("select-webcam-device");
const webcamCamNameInput = document.getElementById("webcam-cam-name");

// Watchlist Form
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

// Logs Table
const logsTbody = document.getElementById("logs-tbody");
const btnRefreshAlerts = document.getElementById("btn-refresh-alerts");
const btnClearLogs = document.getElementById("btn-clear-logs");

// ---------------- INITIALIZATION ----------------
function setupEventListeners() {
  // Global event listeners hook
}

async function initApp() {
  await fetchNetworkInfo();
  await checkStatus();
  await loadWatchlist();
  await loadAlerts();
  await enumerateWebcams();
  await loadSavedCameras();
  setupEventListeners();
  setInterval(checkStatus, 15000);
}

// Fetch network IP for mobile connections
async function fetchNetworkInfo() {
  try {
    const res = await fetch("/api/network-ip");
    const data = await res.json();
    networkIp = data.ip;
    serverPort = data.port;
    updateMobileLink();
  } catch (e) {
    console.warn("Could not fetch network IP:", e);
  }
}

function updateMobileLink() {
  const camId = "mobile_" + Math.random().toString(36).substring(2, 7);
  const mobileUrl = `http://${networkIp}:${serverPort}/mobile?cam_id=${camId}&name=Mobile+Phone+Camera`;
  inputMobileUrl.value = mobileUrl;
  inputMobileUrl.dataset.camId = camId;

  // Render QR Code using Google Chart QR API
  mobileQrCode.innerHTML = `<img src="https://api.qrserver.com/v1/create-qr-code/?size=100x100&data=${encodeURIComponent(mobileUrl)}" alt="Mobile QR Code">`;
}

function copyMobileUrl() {
  inputMobileUrl.select();
  navigator.clipboard.writeText(inputMobileUrl.value);
  alert("Mobile stream link copied! Open it on your phone browser.");
}

// Check Database Status
async function checkStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const db = data.database;

    const currentDbName = document.getElementById("current-db-name");
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
        dbWarningText.innerHTML = `<strong>Local Storage Active:</strong> Table <code>${data.database_name}.watchlist</code> is active.`;
      } else {
        dbWarningBanner.classList.add("hidden");
      }
    }
  } catch (err) {
    dbStatusBadge.className = "status-pill status-warning";
    dbStatusText.textContent = "Server Offline";
  }
}

// ---------------- CAMERA WALL MANAGEMENT ----------------

// Setup Grid Layout buttons
gridBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    gridBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    setGridLayout(btn.dataset.grid);
  });
});

function setGridLayout(layout) {
  currentGridLayout = layout;
  cameraWallGrid.className = `camera-wall-grid grid-${layout}`;
  if (layout !== "1x1" && currentFocusCam) {
    const tile = document.getElementById(`tile-${currentFocusCam}`);
    if (tile) tile.classList.remove("tile-focused");
    currentFocusCam = null;
  }
}

function toggleFocusCamera(camId) {
  const tile = document.getElementById(`tile-${camId}`);
  if (!tile) return;

  if (currentFocusCam === camId) {
    tile.classList.remove("tile-focused");
    currentFocusCam = null;
    setGridLayout(currentGridLayout);
  } else {
    document.querySelectorAll(".camera-tile").forEach(t => t.classList.remove("tile-focused"));
    tile.classList.add("tile-focused");
    currentFocusCam = camId;
  }
}

// Load saved cameras from backend
async function loadSavedCameras() {
  try {
    const res = await fetch("/api/cameras");
    const cams = await res.json();
    for (const cam of cams) {
      if (cam.id !== "cam_default") {
        createCameraTile(cam);
      }
    }
    updateCameraCount();
  } catch (e) {
    console.warn("Could not load cameras:", e);
  }
}

function updateCameraCount() {
  const count = document.querySelectorAll(".camera-tile").length;
  activeCamsCount.textContent = `${count} Active Feed${count > 1 ? "s" : ""}`;
}

// ---------------- PRIMARY WEBCAM LOGIC ----------------
let primaryStream = null;
let primaryInterval = null;
let primaryProcessing = false;

async function startPrimaryWebcam() {
  const video = document.getElementById("webcam-cam_default");
  const overlay = document.getElementById("overlay-cam_default");
  const placeholder = document.getElementById("placeholder-cam_default");
  const btnToggle = document.getElementById("btn-toggle-cam_default");

  try {
    primaryStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    video.srcObject = primaryStream;
    placeholder.style.display = "none";
    btnToggle.textContent = "Stop Webcam";

    video.onloadedmetadata = () => {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      startWebcamLoop("cam_default", "Laptop / Primary Webcam", video, overlay);
    };
  } catch (err) {
    alert("Could not access primary webcam: " + err.message);
  }
}

function stopPrimaryWebcam() {
  if (primaryStream) {
    primaryStream.getTracks().forEach(t => t.stop());
    primaryStream = null;
  }
  if (primaryInterval) {
    clearInterval(primaryInterval);
    primaryInterval = null;
  }
  const video = document.getElementById("webcam-cam_default");
  const placeholder = document.getElementById("placeholder-cam_default");
  const btnToggle = document.getElementById("btn-toggle-cam_default");
  const overlay = document.getElementById("overlay-cam_default");
  const ctx = overlay.getContext("2d");

  video.srcObject = null;
  placeholder.style.display = "flex";
  btnToggle.textContent = "Start Webcam";
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function togglePrimaryWebcam() {
  if (primaryStream) stopPrimaryWebcam();
  else startPrimaryWebcam();
}

function startWebcamLoop(camId, camName, videoEl, overlayEl) {
  if (primaryInterval) clearInterval(primaryInterval);
  const off = document.createElement("canvas");
  const offCtx = off.getContext("2d");
  const ctx = overlayEl.getContext("2d");

  primaryInterval = setInterval(async () => {
    const tile = document.getElementById(`tile-${camId}`);
    if (!primaryStream || primaryProcessing || tile.dataset.paused === "true" || !videoEl.videoWidth) return;

    primaryProcessing = true;
    try {
      off.width = videoEl.videoWidth;
      off.height = videoEl.videoHeight;
      offCtx.drawImage(videoEl, 0, 0, off.width, off.height);
      const dataUrl = off.toDataURL("image/jpeg", 0.75);

      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: dataUrl,
          camera_id: camId,
          camera_name: camName,
          force_recognition: true,
          weapon_threshold: currentWeaponThreshold
        })
      });

      if (res.ok) {
        const result = await res.json();
        handleDetectionResult(camId, camName, result, dataUrl, overlayEl, ctx);
      }
    } catch (e) {
      console.warn("Webcam loop error:", e);
    } finally {
      primaryProcessing = false;
    }
  }, 250);
}

// Synthesized Threat Tone (No external audio file required)
function playThreatTone() {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;
    const audioCtx = new AudioContextClass();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(880, audioCtx.currentTime); // High pitch alarm
    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.28);
    gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.28);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.29);
  } catch (e) {
    // Browser may require user interaction first
  }
}

// ---------------- GENERIC DETECTION RENDERER ----------------
function handleDetectionResult(camId, camName, data, capturedDataUrl, overlayEl, ctx) {
  const motionBadge = document.getElementById(`motion-badge-${camId}`);
  const statEl = document.getElementById(`stat-${camId}`);
  const holdBanner = document.getElementById(`hold-banner-${camId}`);
  const btnResume = document.getElementById(`btn-resume-${camId}`);
  const tile = document.getElementById(`tile-${camId}`);
  const isVideoFeed = tile && tile.querySelector("video") !== null;

  // Motion Pill
  if (motionBadge) {
    if (data.motion_detected) {
      motionBadge.className = "motion-pill motion-active";
      motionBadge.querySelector(".motion-text").textContent = "Motion";
    } else {
      motionBadge.className = "motion-pill motion-idle";
      motionBadge.querySelector(".motion-text").textContent = "Waiting";
    }
  }

  if (statEl) {
    statEl.textContent = Math.round(data.motion_score || 0) + " px motion";
  }

  ctx.clearRect(0, 0, overlayEl.width, overlayEl.height);

  let identifiedTarget = null;
  let identifiedWeapon = null;

  // 1. Draw Faces (Watchlist Biometrics)
  if (data.faces && data.faces.length > 0) {
    for (const face of data.faces) {
      const [x, y, w, h] = face.bbox;
      const isMatch = face.is_match;
      const boxX = isVideoFeed ? (overlayEl.width - x - w) : x;

      ctx.save();
      ctx.lineWidth = 3;
      ctx.strokeStyle = isMatch ? "#f85149" : "#d29922";
      ctx.fillStyle = isMatch ? "#da3633" : "#d29922";
      ctx.strokeRect(boxX, y, w, h);

      const labelText = isMatch
        ? `IDENTIFIED: ${face.name} (${face.confidence}%)`
        : "Unrecognized Person";

      ctx.font = "bold 13px Inter, sans-serif";
      const textWidth = ctx.measureText(labelText).width;
      const tagY = y > 26 ? y - 26 : y;
      ctx.fillRect(boxX, tagY, textWidth + 16, 24);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(labelText, boxX + 8, tagY + 17);
      ctx.restore();

      if (isMatch) identifiedTarget = face;
    }
  }

  // 2. Draw Weapons (Knives, Firearms / Pistols)
  if (data.weapons && data.weapons.length > 0) {
    for (const weapon of data.weapons) {
      const [x, y, w, h] = weapon.bbox;
      const boxX = isVideoFeed ? (overlayEl.width - x - w) : x;

      ctx.save();
      // Glowing Neon Crimson Threat Box
      ctx.shadowColor = "#ff0055";
      ctx.shadowBlur = 12;
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = "#ff1744";
      ctx.strokeRect(boxX, y, w, h);

      // Tech HUD corner brackets
      const corner = Math.min(18, w * 0.25, h * 0.25);
      ctx.lineWidth = 4;
      ctx.strokeStyle = "#ffffff";
      ctx.beginPath();
      // top-left
      ctx.moveTo(boxX, y + corner); ctx.lineTo(boxX, y); ctx.lineTo(boxX + corner, y);
      // top-right
      ctx.moveTo(boxX + w - corner, y); ctx.lineTo(boxX + w, y); ctx.lineTo(boxX + w, y + corner);
      // bottom-left
      ctx.moveTo(boxX, y + h - corner); ctx.lineTo(boxX, y + h); ctx.lineTo(boxX + corner, y + h);
      // bottom-right
      ctx.moveTo(boxX + w - corner, y + h); ctx.lineTo(boxX + w, y + h); ctx.lineTo(boxX + w, y + h - corner);
      ctx.stroke();

      // Weapon Label Tag
      const icon = weapon.class === "knife" ? "🔪" : "🔫";
      const tagText = `${icon} ARMED: ${weapon.label.toUpperCase()} (${weapon.confidence}%)`;

      ctx.font = "bold 13px Inter, sans-serif";
      const textWidth = ctx.measureText(tagText).width;
      const tagY = y > 26 ? y - 26 : y;

      ctx.fillStyle = "#d50000";
      ctx.fillRect(boxX, tagY, textWidth + 18, 24);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(tagText, boxX + 8, tagY + 17);
      ctx.restore();

      identifiedWeapon = weapon;
    }
  }

  // Trigger Immediate Weapon Threat Alert
  if (identifiedWeapon) {
    playThreatTone();
    const icon = identifiedWeapon.class === "knife" ? "🔪" : "🔫";

    // Pause this camera feed so the threat is held on screen
    tile.dataset.paused = "true";
    if (holdBanner) {
      holdBanner.innerHTML = `<span>🚨 ARMED THREAT: ${icon} ${identifiedWeapon.label.toUpperCase()}</span>`;
      holdBanner.classList.remove("hidden");
    }
    if (btnResume) btnResume.classList.remove("hidden");

    // Open the Critical Weapon Alert Modal
    if (modalWeaponImg) modalWeaponImg.src = capturedDataUrl;
    if (modalWeaponType) modalWeaponType.textContent = `${icon} ${identifiedWeapon.label.toUpperCase()}`;
    if (modalWeaponConfidence) modalWeaponConfidence.textContent = `${identifiedWeapon.confidence}% Visual Match`;
    if (modalWeaponCam) modalWeaponCam.textContent = camName;
    if (weaponAlertModal) weaponAlertModal.classList.remove("hidden");

    // Update bottom match alert box
    matchName.innerHTML = `<span style="color: #ff1744; font-weight: 800;">${icon} CRITICAL WEAPON DETECTED: ${identifiedWeapon.label.toUpperCase()}</span>`;
    matchCamName.textContent = camName;
    matchInfo.textContent = `Armed Threat: Visual Weapon Detection confirmed at ${identifiedWeapon.confidence}% confidence`;
    matchConfidence.textContent = identifiedWeapon.confidence + "%";
    matchAlertBox.classList.remove("hidden");

    loadAlerts();
  }

  // Trigger Watchlist Player Target Alert
  if (identifiedTarget) {
    playThreatTone();
    // Pause only this camera feed and hold view
    tile.dataset.paused = "true";
    if (holdBanner) holdBanner.classList.remove("hidden");
    if (btnResume) btnResume.classList.remove("hidden");

    // Show Popup Modal with Camera Source
    modalLiveImg.src = identifiedTarget.snapshot || capturedDataUrl;
    modalRegisteredImg.src = identifiedTarget.registered_image || "/static/placeholder.png";
    modalPersonName.textContent = identifiedTarget.name;
    modalCameraName.textContent = camName;
    modalConfidence.textContent = `${identifiedTarget.confidence}% Facial Similarity`;
    modalDetails.textContent = identifiedTarget.info || identifiedTarget.details || "Watchlist Target";
    alertModal.classList.remove("hidden");

    // Update bottom match alert box
    matchName.textContent = identifiedTarget.name;
    matchCamName.textContent = camName;
    matchInfo.textContent = identifiedTarget.info || "Similar Face Structure Matched";
    matchConfidence.textContent = identifiedTarget.confidence + "%";
    matchAlertBox.classList.remove("hidden");

    loadAlerts();
  }
}

function resumeCameraScan(camId) {
  const tile = document.getElementById(`tile-${camId}`);
  if (!tile) return;
  tile.dataset.paused = "false";

  const holdBanner = document.getElementById(`hold-banner-${camId}`);
  const btnResume = document.getElementById(`btn-resume-${camId}`);
  const overlay = document.getElementById(`overlay-${camId}`);

  if (holdBanner) holdBanner.classList.add("hidden");
  if (btnResume) btnResume.classList.add("hidden");
  if (overlay) {
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }
}

btnModalResume.addEventListener("click", () => {
  alertModal.classList.add("hidden");
  // Resume all paused cameras
  document.querySelectorAll(".camera-tile").forEach(t => {
    resumeCameraScan(t.dataset.camId);
  });
  matchAlertBox.classList.add("hidden");
});

btnModalClose.addEventListener("click", () => {
  alertModal.classList.add("hidden");
});

if (btnResumeWeapon) {
  btnResumeWeapon.addEventListener("click", () => {
    weaponAlertModal.classList.add("hidden");
    document.querySelectorAll(".camera-tile").forEach(t => {
      resumeCameraScan(t.dataset.camId);
    });
    matchAlertBox.classList.add("hidden");
  });
}

if (btnCloseWeaponModal) {
  btnCloseWeaponModal.addEventListener("click", () => {
    weaponAlertModal.classList.add("hidden");
  });
}

// ---------------- ADD NEW CAMERA TILE (DYNAMIC) ----------------

function createCameraTile(cam) {
  const existing = document.getElementById(`tile-${cam.id}`);
  if (existing) return;

  const tile = document.createElement("div");
  tile.className = "camera-tile";
  tile.id = `tile-${cam.id}`;
  tile.dataset.camId = cam.id;
  tile.dataset.camType = cam.type;

  let icon = "📹";
  if (cam.type === "mobile") icon = "📱";
  if (cam.type === "webcam") icon = "💻";

  let mediaHtml = "";
  if (cam.type === "cctv_ip") {
    mediaHtml = `<img id="stream-${cam.id}" src="/api/stream/${cam.id}" alt="${cam.name}">`;
  } else if (cam.type === "mobile") {
    mediaHtml = `
      <img id="mobile-stream-${cam.id}" src="/api/mobile/frame/${cam.id}" alt="${escapeHtml(cam.name)}">
      <div id="mobile-notice-${cam.id}" class="camera-placeholder" style="display: none;">
        <span>📱 Waiting for mobile stream...</span>
      </div>
    `;
  } else {
    mediaHtml = `<video id="video-${cam.id}" autoplay playsinline muted></video>`;
  }

  tile.innerHTML = `
    <div class="tile-header">
      <div class="tile-title">
        <span class="tile-icon">${icon}</span>
        <span class="tile-name">${escapeHtml(cam.name)}</span>
      </div>
      <div class="tile-badges">
        <div class="motion-pill motion-idle" id="motion-badge-${cam.id}">
          <span class="motion-dot"></span>
          <span class="motion-text">Live</span>
        </div>
        <button class="btn-tile-focus" title="Fullscreen Focus" onclick="toggleFocusCamera('${cam.id}')">⛶</button>
        <button class="btn-tile-remove" title="Remove Feed" onclick="removeCameraFeed('${cam.id}')">&times;</button>
      </div>
    </div>

    <div class="tile-video-wrapper">
      ${mediaHtml}
      <canvas id="overlay-${cam.id}"></canvas>
      <div id="hold-banner-${cam.id}" class="scan-hold-banner hidden">
        <span>🎯 TARGET IDENTIFIED ON THIS FEED</span>
      </div>
    </div>

    <div class="tile-footer">
      <span style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase;">${cam.type}</span>
      <button class="btn btn-sm btn-primary hidden" id="btn-resume-${cam.id}" onclick="resumeCameraScan('${cam.id}')">Resume Scan</button>
      <span class="tile-stat" id="stat-${cam.id}">Monitoring</span>
    </div>
  `;

  cameraWallGrid.appendChild(tile);
  updateCameraCount();

  // Setup stream loop for Mobile / CCTV
  if (cam.type === "mobile") {
    startMobileTileWatcher(cam.id, cam.name);
  } else if (cam.type === "cctv_ip") {
    startCctvTileWatcher(cam.id, cam.name);
  } else if (cam.type === "webcam" && cam.device_id) {
    startAdditionalWebcam(cam.id, cam.name, cam.device_id);
  }
}

// Watch mobile frame updates on tile
function startMobileTileWatcher(camId, camName) {
  const imgEl = document.getElementById(`mobile-stream-${camId}`);
  const overlay = document.getElementById(`overlay-${camId}`);
  if (!imgEl || !overlay) return;

  const interval = setInterval(() => {
    const tile = document.getElementById(`tile-${camId}`);
    if (!tile) {
      clearInterval(interval);
      return;
    }
    // Update image src timestamp to fetch latest frame
    imgEl.src = `/api/mobile/frame/${camId}?t=` + Date.now();
  }, 250);
}

// Watch CCTV stream for face detection
function startCctvTileWatcher(camId, camName) {
  const overlay = document.getElementById(`overlay-${camId}`);
  if (!overlay) return;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const interval = setInterval(async () => {
    const tile = document.getElementById(`tile-${camId}`);
    if (!tile) {
      clearInterval(interval);
      return;
    }
    if (tile.dataset.paused === "true") return;

    // Fetch snapshot of CCTV frame for AI recognition
    try {
      const res = await fetch(`/api/mobile/frame/${camId}`);
      if (res.ok) {
        const blob = await res.blob();
        const reader = new FileReader();
        reader.onload = async (e) => {
          const dataUrl = e.target.result;
          const detRes = await fetch("/api/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              image: dataUrl,
              camera_id: camId,
              camera_name: camName,
              force_recognition: true,
              weapon_threshold: currentWeaponThreshold
            })
          });
          if (detRes.ok) {
            const data = await detRes.json();
            handleDetectionResult(camId, camName, data, dataUrl, overlay, overlay.getContext("2d"));
          }
        };
        reader.readAsDataURL(blob);
      }
    } catch (e) {}
  }, 400);
}

// Additional USB webcam
async function startAdditionalWebcam(camId, camName, deviceId) {
  const video = document.getElementById(`video-${camId}`);
  const overlay = document.getElementById(`overlay-${camId}`);
  if (!video || !overlay) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
    });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      startWebcamLoop(camId, camName, video, overlay);
    };
  } catch (e) {
    console.warn("Failed to open USB camera:", e);
  }
}

// Remove camera feed
async function removeCameraFeed(camId) {
  if (!confirm("Remove this camera feed from the wall?")) return;
  const tile = document.getElementById(`tile-${camId}`);
  if (tile) tile.remove();
  updateCameraCount();
  try {
    await fetch(`/api/cameras/${camId}`, { method: "DELETE" });
  } catch (e) {}
}

// ---------------- ADD CAMERA MODAL CONTROLS ----------------
btnOpenAddCamera.addEventListener("click", () => {
  updateMobileLink();
  addCameraModal.classList.remove("hidden");
});

btnCloseAddCamera.addEventListener("click", () => {
  addCameraModal.classList.add("hidden");
});

// Tab Switching
tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    tabBtns.forEach(b => b.classList.remove("active"));
    tabContents.forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

// Add Mobile Feed
async function addMobileCameraFeed() {
  const camId = inputMobileUrl.dataset.camId || ("mobile_" + Date.now());
  const name = mobileCamNameInput.value.trim() || "Mobile Phone Camera";

  const camObj = { id: camId, name: name, type: "mobile" };
  createCameraTile(camObj);

  await fetch("/api/cameras", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(camObj)
  });

  addCameraModal.classList.add("hidden");
  alert(`Mobile feed tile '${name}' added! Open the link on your phone to start streaming.`);
}

// Add CCTV Stream
async function addCctvCameraFeed() {
  const name = cctvNameInput.value.trim();
  const url = cctvUrlInput.value.trim();

  if (!name || !url) {
    alert("Please provide both camera name and RTSP/HTTP stream URL.");
    return;
  }

  const camId = "cctv_" + Date.now();
  const camObj = { id: camId, name: name, type: "cctv_ip", stream_url: url };
  createCameraTile(camObj);

  await fetch("/api/cameras", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(camObj)
  });

  addCameraModal.classList.add("hidden");
  cctvNameInput.value = "";
  cctvUrlInput.value = "";
}

// Enumerate connected webcams
async function enumerateWebcams() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === "videoinput");
    selectWebcamDevice.innerHTML = videoDevices.map((d, i) =>
      `<option value="${d.deviceId}">${d.label || `Camera ${i + 1}`}</option>`
    ).join("");
  } catch (e) {
    console.warn("Could not list video devices:", e);
  }
}

// Add USB Webcam Feed
async function addUsbWebcamFeed() {
  const deviceId = selectWebcamDevice.value;
  const name = webcamCamNameInput.value.trim() || selectWebcamDevice.options[selectWebcamDevice.selectedIndex]?.text || "USB Webcam";

  const camId = "usb_" + Date.now();
  const camObj = { id: camId, name: name, type: "webcam", device_id: deviceId };
  createCameraTile(camObj);

  await fetch("/api/cameras", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(camObj)
  });

  addCameraModal.classList.add("hidden");
}

// ---------------- WATCHLIST MANAGEMENT ----------------
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = personName.value.trim();
  const info = personDetails.value.trim();
  const file = personFile.files[0];

  if (!name || !file) {
    alert("Please provide both player name and a clear photo.");
    return;
  }

  btnSubmitText.textContent = "Analyzing biometrics & adding to watchlist...";
  uploadForm.querySelector("button[type='submit']").disabled = true;

  const formData = new FormData();
  formData.append("name", name);
  formData.append("info", info);
  formData.append("file", file);

  try {
    const res = await fetch("/api/watchlist", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to add player to watchlist");

    alert(`Player '${name}' added to cops_new.watchlist in MongoDB Atlas!`);
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

async function loadWatchlist() {
  try {
    const res = await fetch("/api/watchlist");
    const players = await res.json();
    personsCount.textContent = players.length;

    if (!players || players.length === 0) {
      personsTbody.innerHTML = `<tr><td colspan="5" class="table-empty">No players on watchlist yet. Add someone above.</td></tr>`;
      return;
    }

    personsTbody.innerHTML = players.map(p => `
      <tr>
        <td><img src="${p.image || '/static/placeholder.png'}" class="table-thumb" alt="${escapeHtml(p.name)}"></td>
        <td class="name-cell">${escapeHtml(p.name)}</td>
        <td class="details-cell">${escapeHtml(p.info || '--')}</td>
        <td><span class="badge-watched">WATCHED</span></td>
        <td><button class="btn btn-danger" onclick="deleteFromWatchlist('${p.id}')">Remove</button></td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("Failed to load watchlist:", err);
  }
}

window.deleteFromWatchlist = async function(id) {
  if (!confirm("Are you sure you want to remove this player from the watchlist table?")) return;
  try {
    const res = await fetch(`/api/watchlist/${id}`, { method: "DELETE" });
    if (res.ok) loadWatchlist();
  } catch (err) {
    alert("Error: " + err.message);
  }
};

// ---------------- ALERTS TABLE ----------------
async function loadAlerts() {
  try {
    const res = await fetch("/api/alerts?limit=25");
    const alerts = await res.json();

    if (!alerts || alerts.length === 0) {
      logsTbody.innerHTML = `<tr><td colspan="7" class="table-empty">No alerts triggered yet. Movement + watched player detection will generate alerts here.</td></tr>`;
      return;
    }

    logsTbody.innerHTML = alerts.map(a => {
      const isWeapon = a.alert_type === "WEAPON_DETECTED";
      const icon = a.weapon_type === "knife" ? "🔪" : "🔫";
      const badgeClass = isWeapon ? "badge-weapon-alert" : "badge-alert";
      const badgeText = isWeapon ? `${icon} ARMED WEAPON` : "WATCHLIST MATCH";
      const nameColor = isWeapon ? "#ff1744" : "#ff7b72";
      const rowClass = isWeapon ? "row-weapon-threat" : "";

      return `
      <tr class="${rowClass}">
        <td class="time-cell">${escapeHtml(a.timestamp)}</td>
        <td>${a.snapshot ? `<img src="${a.snapshot}" class="table-thumb" alt="snap">` : '--'}</td>
        <td><span style="font-weight: 600; color: #58a6ff;">${escapeHtml(a.camera_name || 'Main Camera')}</span></td>
        <td class="name-cell" style="color: ${nameColor}; font-weight: 700;">${isWeapon ? icon : '🚨'} ${escapeHtml(a.player_name)}</td>
        <td><span class="badge-confidence" style="${isWeapon ? 'background: rgba(255, 23, 68, 0.2); color: #ff1744; border-color: #ff1744;' : ''}">${a.confidence}%</span></td>
        <td class="details-cell">${escapeHtml(a.info || '--')}</td>
        <td><span class="${badgeClass}">${badgeText}</span></td>
      </tr>
      `;
    }).join("");
  } catch (err) {
    console.error("Failed to load alerts:", err);
  }
}

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

if (btnDismissDbBanner) {
  btnDismissDbBanner.addEventListener("click", () => {
    dbWarningBanner.classList.add("hidden");
    sessionStorage.setItem("dismissed_db_banner", "true");
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

// ---------------- WEAPON SENSITIVITY & VERIFICATION TEST MODAL ----------------
const selectWeaponSensitivity = document.getElementById("select-weapon-sensitivity");
const weaponTestModal = document.getElementById("weapon-test-modal");
const btnOpenWeaponTest = document.getElementById("btn-open-weapon-test");
const btnCloseWeaponTest = document.getElementById("btn-close-weapon-test");
const inputTestWeaponFile = document.getElementById("input-test-weapon-file");
const weaponTestImg = document.getElementById("weapon-test-img");
const weaponTestCanvas = document.getElementById("weapon-test-canvas");
const weaponTestPlaceholder = document.getElementById("weapon-test-placeholder");
const weaponTestSummary = document.getElementById("weapon-test-summary");

if (selectWeaponSensitivity) {
  selectWeaponSensitivity.addEventListener("change", (e) => {
    currentWeaponThreshold = parseFloat(e.target.value) || 0.28;
    console.log("Weapon detection threshold updated to:", currentWeaponThreshold);
  });
}

if (btnOpenWeaponTest) {
  btnOpenWeaponTest.addEventListener("click", () => {
    weaponTestModal.classList.remove("hidden");
  });
}

if (btnCloseWeaponTest) {
  btnCloseWeaponTest.addEventListener("click", () => {
    weaponTestModal.classList.add("hidden");
  });
}

window.testSampleWeapon = async function(sampleUrl) {
  weaponTestPlaceholder.textContent = "Loading real weapon sample & running AI detection...";
  weaponTestPlaceholder.style.display = "block";
  weaponTestImg.style.display = "none";
  weaponTestSummary.textContent = "";

  try {
    const res = await fetch(sampleUrl);
    const blob = await res.blob();
    const reader = new FileReader();
    reader.onload = async (e) => {
      const dataUrl = e.target.result;
      await runWeaponVerification(dataUrl);
    };
    reader.readAsDataURL(blob);
  } catch (err) {
    weaponTestSummary.textContent = "Error loading sample: " + err.message;
  }
};

if (inputTestWeaponFile) {
  inputTestWeaponFile.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async (ev) => {
        await runWeaponVerification(ev.target.result);
      };
      reader.readAsDataURL(file);
    }
  });
}

async function runWeaponVerification(dataUrl) {
  weaponTestPlaceholder.textContent = "Running Dual-Model YOLOv8 Threat Ensemble...";
  weaponTestPlaceholder.style.display = "block";
  weaponTestImg.style.display = "none";

  try {
    const res = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: dataUrl,
        camera_id: "test_verification",
        camera_name: "Real Weapon Image Test",
        force_recognition: false,
        weapon_threshold: currentWeaponThreshold
      })
    });

    const data = await res.json();
    weaponTestPlaceholder.style.display = "none";
    weaponTestImg.src = dataUrl;
    weaponTestImg.style.display = "block";

    weaponTestImg.onload = () => {
      weaponTestCanvas.width = weaponTestImg.clientWidth || 400;
      weaponTestCanvas.height = weaponTestImg.clientHeight || 300;
      const ctx = weaponTestCanvas.getContext("2d");
      ctx.clearRect(0, 0, weaponTestCanvas.width, weaponTestCanvas.height);

      const scaleX = weaponTestCanvas.width / weaponTestImg.naturalWidth;
      const scaleY = weaponTestCanvas.height / weaponTestImg.naturalHeight;

      if (data.weapons && data.weapons.length > 0) {
        for (const w of data.weapons) {
          const [bx, by, bw, bh] = w.bbox;
          const sx = bx * scaleX;
          const sy = by * scaleY;
          const sw = bw * scaleX;
          const sh = bh * scaleY;

          ctx.save();
          ctx.shadowColor = "#ff0055";
          ctx.shadowBlur = 10;
          ctx.lineWidth = 3.5;
          ctx.strokeStyle = "#ff1744";
          ctx.strokeRect(sx, sy, sw, sh);

          const icon = w.class === "knife" ? "🔪" : "🔫";
          const tag = `${icon} ${w.label.toUpperCase()} (${w.confidence}%)`;
          ctx.fillStyle = "#d50000";
          ctx.font = "bold 13px Inter, sans-serif";
          ctx.fillRect(sx, sy > 26 ? sy - 26 : sy, ctx.measureText(tag).width + 16, 24);
          ctx.fillStyle = "#fff";
          ctx.fillText(tag, sx + 8, sy > 26 ? sy - 9 : sy + 17);
          ctx.restore();
        }

        const weaponSummaryText = data.weapons.map(w => `${w.label} (${w.confidence}%)`).join(", ");
        weaponTestSummary.innerHTML = `<span style="color: #ff1744;">🚨 DETECTED: ${weaponSummaryText}</span> (Logged in Atlas alerts)`;
        loadAlerts();
      } else {
        weaponTestSummary.innerHTML = `<span style="color: #e3b341;">ℹ️ No weapon detected above ${Math.round(currentWeaponThreshold * 100)}% threshold. Try switching Sensitivity to High (22%).</span>`;
      }
    };
  } catch (err) {
    weaponTestSummary.textContent = "Test error: " + err.message;
  }
}

// Boot app
initApp();
