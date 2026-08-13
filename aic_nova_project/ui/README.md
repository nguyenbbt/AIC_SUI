# AIC Nova operator UI

The UI is a React/Vite operator console for t-KIS, v-KIS, TRAKE and VQA. Both
KIS variants call the same `/search` pipeline; v-KIS only changes how the human
obtains the text description.

## Run locally

Start the Online API on port 8000, then use Node 20.19 or newer:

To review every screen without Offline data or model services, start the
strictly isolated demo API from the project root:

```powershell
python -m uvicorn retrieval_api.demo:app --host 127.0.0.1 --port 8000
```

Then, in another terminal:

```powershell
cd ui
npm install
npm run dev
```

Vite proxies `/api` to `http://localhost:8000`. Set `VITE_API_BASE_URL` only
when the UI and API are served behind a different gateway. Production builds:

```powershell
npm run build
```

## Implemented operator workflow

- t-KIS and v-KIS text input, optional single Vietnamese q1 rewrite, and
  seven-branch selection. q0 remains the operator's original text; q2 is not
  part of the active contract.
- Object autocomplete from the active SQLite object vocabulary, with explicit
  COCO-80 fallback, counts `1+`/`2+`/`3+`, and soft/hard behavior. Unknown
  labels are rejected and hard filtering is disabled unless the SQLite catalog
  (or the isolated demo fixture) is active.
- Ranked keyframe grid, diagnostics, adjacent keyframes, video seek preview.
- Selection tray capped at 100 and logical KIS export using BTC
  `frame_id = source_frame_idx`.
- Ordered-event TRAKE editor, DANTE timeline and logical TRAKE export.
- VQA answer type, grounded evidence display, explicit cited-frame selection,
  and logical VQA export.

Production must serve the UI and `/api` through the same gateway origin, or set
`VITE_API_BASE_URL` and configure an explicit trusted-origin policy at that
gateway. The API intentionally does not enable permissive CORS.

The organizer PDF does not freeze the upload transport or file delimiter. The
downloads are therefore deliberately named `logical-submission.json`; adapt
them only after the organizer publishes the final submission transport.
