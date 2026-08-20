const API = import.meta.env.VITE_API_BASE_URL || "/api";

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) }
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body?.error?.message || `HTTP ${response.status}`);
  return body;
}

async function download(path, payload) {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body?.error?.message || `HTTP ${response.status}`);
  }
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="([^"]+)"/i);
  return { blob: await response.blob(), filename: match?.[1] || "submission.zip" };
}

export const api = {
  health: () => request("/health/ready"),
  catalog: () => request("/catalog/object-labels"),
  rewrite: (query, requestId) => request("/query/rewrite", { method: "POST", body: JSON.stringify({ query, request_id: requestId }) }),
  search: (payload) => request("/search", { method: "POST", body: JSON.stringify(payload) }),
  trake: (payload) => request("/trake", { method: "POST", body: JSON.stringify(payload) }),
  vqa: (payload) => request("/vqa", { method: "POST", body: JSON.stringify(payload) }),
  packageSubmission: (payload) => download("/submission/package", payload),
  imageUrl: (frameId) => `${API}/media/keyframes/${encodeURIComponent(frameId)}`,
  videoUrl: (videoId) => `${API}/media/videos/${encodeURIComponent(videoId)}`,
  neighbors: (frameId) => request(`/media/keyframes/${encodeURIComponent(frameId)}/neighbors?before=2&after=2`)
};
