/**
 * Popup script – Deepdetect AI Chrome Extension
 */

"use strict";

const imageList = document.getElementById("imageList");
const statusBar = document.getElementById("statusBar");
const scanBtn = document.getElementById("scanBtn");
const clearBtn = document.getElementById("clearBtn");

// State
let mediaUrls = { images: [], videos: [] };
let analysing = new Set();

// ──────────────────────────────────────────────────────────────────────────
// Initialise
// ──────────────────────────────────────────────────────────────────────────

async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;

  try {
    const response = await chrome.tabs.sendMessage(tab.id, { type: "GET_MEDIA" });
    mediaUrls = response || { images: [], videos: [] };
    renderImageList(mediaUrls.images);
    setStatus(`Found ${mediaUrls.images.length} image(s), ${mediaUrls.videos.length} video(s).`);
  } catch {
    setStatus("Cannot scan — reload the page and try again.");
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Render
// ──────────────────────────────────────────────────────────────────────────

function renderImageList(urls) {
  if (urls.length === 0) {
    imageList.innerHTML = '<div class="empty">No images found.</div>';
    return;
  }
  imageList.innerHTML = "";
  urls.slice(0, 20).forEach((url) => {
    const item = document.createElement("div");
    item.className = "media-item";
    item.dataset.url = url;

    const urlSpan = document.createElement("span");
    urlSpan.className = "url";
    urlSpan.textContent = url;
    urlSpan.title = url;

    const btn = document.createElement("button");
    btn.className = "analyse-btn";
    btn.textContent = "Analyse";
    btn.addEventListener("click", () => analyseItem(url, "image", item, btn));

    item.appendChild(urlSpan);
    item.appendChild(btn);
    imageList.appendChild(item);
  });
}

// ──────────────────────────────────────────────────────────────────────────
// Analysis
// ──────────────────────────────────────────────────────────────────────────

async function analyseItem(url, mediaType, itemEl, btnEl) {
  if (analysing.has(url)) return;
  analysing.add(url);
  btnEl.disabled = true;
  btnEl.textContent = "…";
  setStatus(`Analyzing ${truncate(url, 40)}…`);

  const response = await sendToBackground({ type: "ANALYSE_URL", mediaType, url });

  analysing.delete(url);
  btnEl.disabled = false;

  if (response.ok) {
    const { prediction, confidence } = response.result;
    const badge = document.createElement("span");
    badge.className = `result-badge ${prediction}`;
    badge.textContent = `${prediction.toUpperCase()} ${Math.round(confidence * 100)}%`;
    btnEl.replaceWith(badge);

    if (prediction === "fake") {
      markPageElement(url, confidence);
    }
    setStatus(`Result: ${prediction} (${Math.round(confidence * 100)}%)`);
  } else {
    btnEl.textContent = "Retry";
    setStatus(`Error: ${response.error}`);
  }
}

async function markPageElement(url, confidence) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) {
    chrome.tabs.sendMessage(tab.id, { type: "MARK_FAKE", url, confidence });
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Button handlers
// ──────────────────────────────────────────────────────────────────────────

scanBtn.addEventListener("click", init);

clearBtn.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) {
    chrome.tabs.sendMessage(tab.id, { type: "CLEAR_MARKS" });
    setStatus("Marks cleared.");
  }
});

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

function sendToBackground(message) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(message, (response) => {
      resolve(response || { ok: false, error: "No response" });
    });
  });
}

function setStatus(msg) {
  statusBar.textContent = msg;
}

function truncate(str, len) {
  return str.length > len ? str.slice(0, len) + "…" : str;
}

// Run on open
init();
