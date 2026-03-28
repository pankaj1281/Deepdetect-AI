import axios from "axios";

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: BASE_URL });

/**
 * Predict whether an image is real or fake.
 * @param {File} file
 * @param {function} onProgress  - optional (loaded, total) callback
 */
export async function predictImage(file, onProgress) {
  const form = new FormData();
  form.append("file", file);
  const res = await api.post("/predict/image", form, {
    onUploadProgress: onProgress
      ? (e) => onProgress(e.loaded, e.total)
      : undefined,
  });
  return res.data;
}

/**
 * Predict whether a video is real or fake.
 * @param {File} file
 * @param {function} onProgress
 */
export async function predictVideo(file, onProgress) {
  const form = new FormData();
  form.append("file", file);
  const res = await api.post("/predict/video", form, {
    onUploadProgress: onProgress
      ? (e) => onProgress(e.loaded, e.total)
      : undefined,
  });
  return res.data;
}

/**
 * Predict whether audio is real or fake.
 * @param {File} file
 * @param {function} onProgress
 */
export async function predictAudio(file, onProgress) {
  const form = new FormData();
  form.append("file", file);
  const res = await api.post("/predict/audio", form, {
    onUploadProgress: onProgress
      ? (e) => onProgress(e.loaded, e.total)
      : undefined,
  });
  return res.data;
}

/**
 * Fetch recent detection results from the backend.
 * @param {number} limit
 */
export async function fetchResults(limit = 20) {
  const res = await api.get("/results", { params: { limit } });
  return res.data;
}
