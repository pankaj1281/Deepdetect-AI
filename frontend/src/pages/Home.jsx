import React, { useState } from "react";
import FileUploader from "../components/FileUploader";
import ResultCard from "../components/ResultCard";
import HeatmapViewer from "../components/HeatmapViewer";
import ProgressBar from "../components/ProgressBar";
import { predictImage, predictVideo, predictAudio } from "../api";

const TABS = [
  { id: "image", label: "Image", icon: "🖼️" },
  { id: "video", label: "Video", icon: "🎬" },
  { id: "audio", label: "Audio", icon: "🎵" },
];

export default function Home() {
  const [tab, setTab] = useState("image");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleTabChange = (id) => {
    setTab(id);
    setFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
  };

  const handleFile = (f) => {
    setFile(f);
    setResult(null);
    setError(null);
  };

  const handleDetect = async () => {
    if (!file) return;
    setLoading(true);
    setProgress(0);
    setError(null);
    setResult(null);

    const onProgress = (loaded, total) =>
      setProgress(Math.round((loaded / total) * 90));

    try {
      let data;
      if (tab === "image") data = await predictImage(file, onProgress);
      else if (tab === "video") data = await predictVideo(file, onProgress);
      else data = await predictAudio(file, onProgress);
      setProgress(100);
      setResult(data);
    } catch (err) {
      setError(
        err?.response?.data?.detail || "Detection failed. Is the backend running?"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-extrabold text-white">
          AI Deepfake Detector
        </h1>
        <p className="text-gray-400 text-lg">
          Upload an image, video, or audio clip to check for synthetic / deepfake content.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex justify-center gap-3">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => handleTabChange(t.id)}
            className={`px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center gap-2 transition-all
              ${tab === t.id
                ? "bg-brand-600 text-white shadow-lg shadow-brand-900/40"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white"
              }`}
          >
            <span>{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>

      {/* Uploader */}
      <FileUploader mediaType={tab} onFile={handleFile} disabled={loading} />

      {/* Selected file info */}
      {file && (
        <div className="flex items-center justify-between bg-gray-900 rounded-xl px-4 py-3">
          <span className="text-sm text-gray-300 truncate">{file.name}</span>
          <span className="text-xs text-gray-500 ml-4 shrink-0">
            {(file.size / 1024).toFixed(1)} KB
          </span>
        </div>
      )}

      {/* Upload progress */}
      {loading && (
        <div className="space-y-1">
          <ProgressBar value={progress} />
          <p className="text-xs text-gray-500 text-right">{progress}%</p>
        </div>
      )}

      {/* Detect button */}
      <button
        onClick={handleDetect}
        disabled={!file || loading}
        className="w-full py-3 rounded-xl font-bold text-white bg-brand-600 hover:bg-brand-700
          disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-lg"
      >
        {loading ? "Analyzing…" : "Detect Deepfake"}
      </button>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-700 text-red-300 rounded-xl px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-4">
          <ResultCard result={result} />
          {result.heatmap_url && (
            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-2">
                Explainability — Grad-CAM Heatmap
              </h3>
              <HeatmapViewer
                heatmapUrl={result.heatmap_url}
                label="Red/yellow regions indicate manipulated areas"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
