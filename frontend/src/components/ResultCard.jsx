import React from "react";

/**
 * Displays the prediction result card.
 *
 * Props:
 *   result : { prediction, confidence, raw_score, filename, timestamp, frame_count? }
 */
export default function ResultCard({ result }) {
  if (!result) return null;

  const isFake = result.prediction === "fake";
  const pct = Math.round(result.confidence * 100);

  return (
    <div
      className={`rounded-2xl border p-6 space-y-4 ${
        isFake
          ? "border-red-600 bg-red-950/30"
          : "border-green-600 bg-green-950/30"
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-bold">
          {isFake ? (
            <span className="text-red-400">⚠️ Fake Detected</span>
          ) : (
            <span className="text-green-400">✅ Authentic</span>
          )}
        </h3>
        <span
          className={`text-3xl font-extrabold ${
            isFake ? "text-red-400" : "text-green-400"
          }`}
        >
          {pct}%
        </span>
      </div>

      {/* Confidence bar */}
      <div>
        <p className="text-xs text-gray-400 mb-1">Confidence</p>
        <div className="w-full bg-gray-800 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isFake ? "bg-red-500" : "bg-green-500"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Metadata */}
      <div className="text-sm text-gray-400 space-y-1">
        {result.filename && (
          <p>
            <span className="text-gray-500">File:</span> {result.filename}
          </p>
        )}
        {result.frame_count && (
          <p>
            <span className="text-gray-500">Frames analyzed:</span>{" "}
            {result.frame_count}
          </p>
        )}
        {result.timestamp && (
          <p>
            <span className="text-gray-500">Time:</span>{" "}
            {new Date(result.timestamp).toLocaleString()}
          </p>
        )}
      </div>
    </div>
  );
}
