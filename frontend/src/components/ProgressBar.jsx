import React from "react";

/**
 * Animated progress bar.
 * Props: value (0-100 number)
 */
export default function ProgressBar({ value }) {
  return (
    <div className="w-full bg-gray-800 rounded-full h-2.5 overflow-hidden">
      <div
        className="h-full bg-gradient-to-r from-brand-500 to-indigo-400 rounded-full transition-all duration-300"
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  );
}
