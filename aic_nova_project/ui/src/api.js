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

export const api = {
  health: () => request("/health/ready"),
  catalog: () => request("/catalog/object-labels"),
  rewrite: (query, requestId) => request("/query/rewrite", { method: "POST", body: JSON.stringify({ query, request_id: requestId }) }),
  search: (payload) => request("/search", { method: "POST", body: JSON.stringify(payload) }),
  trake: (payload) => request("/trake", { method: "POST", body: JSON.stringify(payload) }),
  vqa: (payload) => request("/vqa", { method: "POST", body: JSON.stringify(payload) }),
  imageUrl: (frameId) => `${API}/media/keyframes/${encodeURIComponent(frameId)}`,
  videoUrl: (videoId) => `${API}/media/videos/${encodeURIComponent(videoId)}`,
  neighbors: (frameId) => request(`/media/keyframes/${encodeURIComponent(frameId)}/neighbors?before=2&after=2`)
};
