/**
 * Content script – Deepdetect AI Chrome Extension
 *
 * Scans the current page for images and videos, collects their URLs,
 * and reports them to the popup/background on request.
 */

(function () {
  "use strict";

  const BADGE_CLASS = "deepdetect-badge";

  // ──────────────────────────────────────────────────────────────────────────
  // Media discovery
  // ──────────────────────────────────────────────────────────────────────────

  function collectMediaUrls() {
    const images = Array.from(document.querySelectorAll("img"))
      .map((el) => el.src)
      .filter(isValidMediaUrl);

    const videos = Array.from(document.querySelectorAll("video source, video[src]"))
      .map((el) => el.src || el.getAttribute("src"))
      .filter(isValidMediaUrl);

    return { images, videos };
  }

  function isValidMediaUrl(url) {
    return (
      typeof url === "string" &&
      url.length > 0 &&
      (url.startsWith("http://") || url.startsWith("https://"))
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Badge overlay (marks fake media on the page)
  // ──────────────────────────────────────────────────────────────────────────

  function injectBadgeStyles() {
    if (document.getElementById("deepdetect-styles")) return;
    const style = document.createElement("style");
    style.id = "deepdetect-styles";
    style.textContent = `
      .${BADGE_CLASS} {
        position: absolute;
        top: 4px;
        left: 4px;
        background: rgba(220, 38, 38, 0.9);
        color: #fff;
        font-size: 11px;
        font-weight: bold;
        padding: 2px 6px;
        border-radius: 4px;
        z-index: 99999;
        pointer-events: none;
        font-family: sans-serif;
        letter-spacing: 0.04em;
      }
      .deepdetect-highlighted {
        outline: 3px solid #dc2626 !important;
        outline-offset: 2px;
      }
    `;
    document.head.appendChild(style);
  }

  function markElementAsFake(el, confidence) {
    injectBadgeStyles();

    // Ensure parent is positioned so the badge overlay works
    const parent = el.parentElement;
    if (parent && getComputedStyle(parent).position === "static") {
      parent.style.position = "relative";
    }

    el.classList.add("deepdetect-highlighted");

    const badge = document.createElement("span");
    badge.className = BADGE_CLASS;
    badge.textContent = `⚠ FAKE ${Math.round(confidence * 100)}%`;
    el.insertAdjacentElement("beforebegin", badge);
  }

  function clearBadges() {
    document.querySelectorAll(`.${BADGE_CLASS}`).forEach((b) => b.remove());
    document.querySelectorAll(".deepdetect-highlighted").forEach((el) => {
      el.classList.remove("deepdetect-highlighted");
    });
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Message handling
  // ──────────────────────────────────────────────────────────────────────────

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "GET_MEDIA") {
      sendResponse(collectMediaUrls());
      return true;
    }

    if (message.type === "MARK_FAKE") {
      const { url, confidence } = message;
      const imgs = Array.from(document.querySelectorAll("img")).filter(
        (el) => el.src === url
      );
      imgs.forEach((el) => markElementAsFake(el, confidence));
      sendResponse({ marked: imgs.length });
      return true;
    }

    if (message.type === "CLEAR_MARKS") {
      clearBadges();
      sendResponse({ cleared: true });
      return true;
    }
  });
})();
