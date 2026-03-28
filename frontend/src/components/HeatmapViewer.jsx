import React, { useState } from "react";

/**
 * Renders a Grad-CAM or spectrogram heatmap overlay image.
 *
 * Props:
 *   heatmapUrl : base64 data URI (from backend) or empty string
 *   label      : string – caption shown below the image
 */
export default function HeatmapViewer({ heatmapUrl, label }) {
  const [enlarged, setEnlarged] = useState(false);

  if (!heatmapUrl) {
    return (
      <div className="rounded-xl border border-gray-700 bg-gray-900 p-6 text-center text-gray-500">
        No heatmap available
      </div>
    );
  }

  return (
    <>
      <div
        className="rounded-xl overflow-hidden border border-gray-700 cursor-zoom-in"
        onClick={() => setEnlarged(true)}
      >
        <img
          src={heatmapUrl}
          alt="Grad-CAM heatmap"
          className="w-full object-contain max-h-64"
        />
        {label && (
          <p className="text-xs text-center text-gray-500 py-2 bg-gray-900">{label}</p>
        )}
      </div>

      {/* Lightbox */}
      {enlarged && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-50"
          onClick={() => setEnlarged(false)}
        >
          <img
            src={heatmapUrl}
            alt="Grad-CAM heatmap (enlarged)"
            className="max-w-3xl max-h-screen rounded-xl shadow-2xl"
          />
        </div>
      )}
    </>
  );
}
