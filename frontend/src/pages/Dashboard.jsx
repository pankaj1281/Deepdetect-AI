import React, { useEffect, useState } from "react";
import { fetchResults } from "../api";

function Badge({ prediction }) {
  const isFake = prediction === "fake";
  return (
    <span
      className={`text-xs font-bold px-2 py-1 rounded-full ${
        isFake ? "bg-red-900 text-red-300" : "bg-green-900 text-green-300"
      }`}
    >
      {prediction.toUpperCase()}
    </span>
  );
}

function MediaIcon({ type }) {
  const icons = { image: "🖼️", video: "🎬", audio: "🎵" };
  return <span>{icons[type] || "📁"}</span>;
}

export default function Dashboard() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchResults(50);
      setResults(data.results || []);
    } catch (err) {
      setError("Failed to load results. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-extrabold text-white">Detection Dashboard</h1>
        <button
          onClick={load}
          disabled={loading}
          className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-sm rounded-xl text-gray-300
            disabled:opacity-40"
        >
          {loading ? "Loading…" : "↻ Refresh"}
        </button>
      </div>

      {/* Stats row */}
      {results.length > 0 && (
        <div className="grid grid-cols-3 gap-4">
          {[
            {
              label: "Total Analyzed",
              value: results.length,
              color: "text-white",
            },
            {
              label: "Fake Detected",
              value: results.filter((r) => r.prediction === "fake").length,
              color: "text-red-400",
            },
            {
              label: "Authentic",
              value: results.filter((r) => r.prediction === "real").length,
              color: "text-green-400",
            },
          ].map((stat) => (
            <div
              key={stat.label}
              className="bg-gray-900 rounded-2xl p-5 border border-gray-800 text-center"
            >
              <p className={`text-3xl font-extrabold ${stat.color}`}>{stat.value}</p>
              <p className="text-xs text-gray-500 mt-1">{stat.label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-700 text-red-300 rounded-xl px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {/* Results table */}
      {results.length === 0 && !loading && !error && (
        <p className="text-gray-500 text-center py-16">
          No results yet. Upload media on the Detector page.
        </p>
      )}

      {results.length > 0 && (
        <div className="overflow-x-auto rounded-2xl border border-gray-800">
          <table className="w-full text-sm">
            <thead className="bg-gray-900 text-gray-400 uppercase text-xs">
              <tr>
                <th className="px-4 py-3 text-left">Type</th>
                <th className="px-4 py-3 text-left">File</th>
                <th className="px-4 py-3 text-left">Result</th>
                <th className="px-4 py-3 text-right">Confidence</th>
                <th className="px-4 py-3 text-left">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {results.map((r, idx) => (
                <tr key={idx} className="hover:bg-gray-900/50 transition-colors">
                  <td className="px-4 py-3">
                    <MediaIcon type={r.media_type} />
                  </td>
                  <td className="px-4 py-3 text-gray-300 truncate max-w-xs">
                    {r.filename || "—"}
                  </td>
                  <td className="px-4 py-3">
                    <Badge prediction={r.prediction} />
                  </td>
                  <td className="px-4 py-3 text-right text-gray-300">
                    {Math.round(r.confidence * 100)}%
                  </td>
                  <td className="px-4 py-3 text-gray-500 text-xs">
                    {r.timestamp
                      ? new Date(r.timestamp).toLocaleString()
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
