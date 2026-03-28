/**
 * Background service worker – Deepdetect AI Chrome Extension
 *
 * Handles URL-based prediction requests from the popup by fetching the
 * media URL and forwarding the bytes to the FastAPI backend.
 */

"use strict";

const API_BASE = "http://localhost:8000";

// ──────────────────────────────────────────────────────────────────────────
// Message handler
// ──────────────────────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "ANALYSE_URL") {
    analyseMediaUrl(message.mediaType, message.url)
      .then((result) => sendResponse({ ok: true, result }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true; // keep the message channel open for async response
  }
});

// ──────────────────────────────────────────────────────────────────────────
// Core logic
// ──────────────────────────────────────────────────────────────────────────

async function analyseMediaUrl(mediaType, url) {
  // Download the media resource
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch media (${response.status})`);
  }
  const blob = await response.blob();

  // Determine the correct endpoint
  const endpoint = `${API_BASE}/predict/${mediaType}`;

  const form = new FormData();
  form.append("file", blob, extractFilename(url));

  const apiResponse = await fetch(endpoint, { method: "POST", body: form });
  if (!apiResponse.ok) {
    const text = await apiResponse.text();
    throw new Error(`Backend error (${apiResponse.status}): ${text}`);
  }
  return apiResponse.json();
}

function extractFilename(url) {
  try {
    const parts = new URL(url).pathname.split("/");
    return parts[parts.length - 1] || "media";
  } catch {
    return "media";
  }
}
