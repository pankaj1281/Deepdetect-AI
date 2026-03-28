import React, { useCallback } from "react";
import { useDropzone } from "react-dropzone";

const ACCEPT_MAP = {
  image: { "image/*": [".jpg", ".jpeg", ".png", ".webp", ".bmp"] },
  video: { "video/*": [".mp4", ".avi", ".mov", ".mkv"] },
  audio: { "audio/*": [".wav", ".mp3", ".flac", ".ogg"] },
};

/**
 * Drag-and-drop file uploader for a given media type.
 *
 * Props:
 *   mediaType  : "image" | "video" | "audio"
 *   onFile     : (File) => void
 *   disabled   : bool
 */
export default function FileUploader({ mediaType, onFile, disabled }) {
  const onDrop = useCallback(
    (accepted) => {
      if (accepted.length > 0) onFile(accepted[0]);
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT_MAP[mediaType] || {},
    maxFiles: 1,
    disabled,
  });

  const icons = {
    image: "🖼️",
    video: "🎬",
    audio: "🎵",
  };

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer select-none
        transition-all duration-200
        ${isDragActive ? "border-brand-500 bg-brand-900/20" : "border-gray-700 hover:border-brand-600"}
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <input {...getInputProps()} />
      <div className="text-5xl mb-3">{icons[mediaType]}</div>
      {isDragActive ? (
        <p className="text-brand-400 font-medium">Drop the file here…</p>
      ) : (
        <>
          <p className="text-gray-300 font-medium">
            Drag & drop a {mediaType} file, or{" "}
            <span className="text-brand-400 underline">click to browse</span>
          </p>
          <p className="text-gray-500 text-sm mt-1">
            Accepted:{" "}
            {Object.values(ACCEPT_MAP[mediaType] || {})
              .flat()
              .join(", ")}
          </p>
        </>
      )}
    </div>
  );
}
