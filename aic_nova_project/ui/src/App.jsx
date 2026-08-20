import { useEffect, useMemo, useState } from "react";
import { api } from "./api.js";

const BRANCHES = ["visual_dense", "ocr_dense", "ocr_bm25", "asr_dense", "asr_bm25", "summary_dense", "summary_bm25"];
const uid = () => (globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`);

export default function App() {
  const [mode, setMode] = useState("kis_text");
  const [query, setQuery] = useState("");
  const [paraphrases, setParaphrases] = useState([]);
  const [rewriteMeta, setRewriteMeta] = useState(null);
  const [events, setEvents] = useState(["", ""]);
  const [answerType, setAnswerType] = useState("short_text");
  const [constraints, setConstraints] = useState([]);
  const [labels, setLabels] = useState([]);
  const [catalogSource, setCatalogSource] = useState("loading");
  const [enabledBranches, setEnabledBranches] = useState(BRANCHES);
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);
  const [inspect, setInspect] = useState(null);
  const [queryFilename, setQueryFilename] = useState("");
  const [submissionFiles, setSubmissionFiles] = useState([]);
  const [archiveName, setArchiveName] = useState("submission");
  const [vqaSubmission, setVqaSubmission] = useState(null);

  useEffect(() => {
    api.catalog().then((x) => { setLabels(x.labels); setCatalogSource(x.source); }).catch(() => setCatalogSource("unavailable"));
    let active = true;
    const refreshHealth = () => api.health()
      .then((value) => { if (active) setHealth(value); })
      .catch(() => { if (active) setHealth({ status: "unavailable", checks: {} }); });
    refreshHealth();
    const timer = globalThis.setInterval(refreshHealth, 10_000);
    return () => { active = false; globalThis.clearInterval(timer); };
  }, []);

  const candidates = result?.candidates || [];
  const selectionKeys = useMemo(() => new Set(selected.map((x) => `${x.video_id}:${x.source_frame_idx}`)), [selected]);
  const submissionMode = mode.startsWith("kis") ? "kis" : mode === "vqa" ? "qa" : "trake";

  async function rewrite() {
    if (!query.trim()) return;
    setBusy(true); setError("");
    try { const x = await api.rewrite(query.trim(), `rw-${uid()}`); setParaphrases((x.paraphrases || []).slice(0, 1)); setRewriteMeta(x); }
    catch (e) { setError(e.message); setParaphrases([]); setRewriteMeta(null); }
    finally { setBusy(false); }
  }

  async function run() {
    const labelMap = new Map(labels.map((item) => [item.label.toLocaleLowerCase(), item.label]));
    const normalizedConstraints = constraints.map(({ id, ...item }) => ({
      ...item,
      label: labelMap.get(item.label.trim().toLocaleLowerCase()) || item.label.trim(),
    }));
    const invalidLabels = normalizedConstraints.filter((item) => !labelMap.has(item.label.toLocaleLowerCase()));
    const hardAllowed = catalogSource === "sqlite" || catalogSource === "demo_fixture";
    if (mode.startsWith("kis") && invalidLabels.length) {
      setError(`Object không có trong catalog Offline: ${invalidLabels.map((x) => x.label || "(trống)").join(", ")}`);
      return;
    }
    if (mode.startsWith("kis") && !hardAllowed && normalizedConstraints.some((x) => x.filter_mode === "hard")) {
      setError("Hard filter chỉ được dùng khi object catalog được đọc từ SQLite Offline.");
      return;
    }
    if (mode.startsWith("kis") && !enabledBranches.length) {
      setError("Phải bật ít nhất một retrieval branch.");
      return;
    }
    const currentQuery = query.trim();
    const currentParaphrases = rewriteMeta?.original_text === currentQuery ? paraphrases.slice(0, 1) : [];
    setBusy(true); setError(""); setResult(null);
    if (mode.startsWith("kis")) setSelected([]);
    if (mode === "vqa") setVqaSubmission(null);
    try {
      if (mode.startsWith("kis")) {
        setResult(await api.search({
          query: currentQuery, mode, paraphrases: currentParaphrases,
          object_constraints: normalizedConstraints,
          enabled_branches: enabledBranches, include_diagnostics: true,
          query_id: `kis-${uid()}`
        }));
      } else if (mode === "trake") {
        const eventTexts = events.map((x) => x.trim()).filter(Boolean);
        setResult(await api.trake({ query_id: `trake-${uid()}`, event_texts: eventTexts, top_k_videos: 5 }));
      } else {
        setResult(await api.vqa({ question_id: `vqa-${uid()}`, question: query.trim(), answer_type: answerType }));
      }
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  function changeQuery(value) {
    setQuery(value);
    setParaphrases([]);
    setRewriteMeta(null);
    setResult(null);
    setSelected([]);
    setVqaSubmission(null);
  }

  function toggleCandidate(candidate) {
    const key = `${candidate.video_id}:${candidate.source_frame_idx}`;
    setSelected((old) => selectionKeys.has(key) ? old.filter((x) => `${x.video_id}:${x.source_frame_idx}` !== key) : [...old, candidate].slice(0, 100));
  }

  function saveCurrentSubmission() {
    setError("");
    try {
      const filename = validateQueryFilename(queryFilename, submissionMode);
      let file;
      if (mode.startsWith("kis")) {
        if (!selected.length) throw new Error("Hãy chọn ít nhất một frame KIS trước khi lưu.");
        file = { mode: "kis", query_filename: filename, rows: selected.map((x) => ({ video_id: x.video_id, frame_id: x.source_frame_idx })) };
      } else if (mode === "trake") {
        const eventCount = events.map((x) => x.trim()).filter(Boolean).length;
        const rows = result?.results?.map((x) => ({ video_id: x.video_id, frame_ids: x.sequence.map((frame) => frame.source_frame_idx) })) || [];
        if (!rows.length) throw new Error("Chưa có kết quả TRAKE để lưu.");
        if (rows.some((row) => row.frame_ids.length !== eventCount)) throw new Error("TRAKE không có đúng một frame cho mỗi event.");
        file = { mode: "trake", query_filename: filename, event_count: eventCount, rows };
      } else {
        if (!vqaSubmission) throw new Error("Hãy chọn evidence frame và nhập answer Q&A trước khi lưu.");
        file = { mode: "qa", query_filename: filename, rows: [vqaSubmission] };
      }
      setSubmissionFiles((old) => [...old.filter((x) => csvName(x.query_filename).toLowerCase() !== csvName(filename).toLowerCase()), file]);
    } catch (e) { setError(e.message); }
  }

  async function downloadSubmissionPackage() {
    setError(""); setBusy(true);
    try {
      if (!/^[A-Za-z0-9_-]+$/.test(archiveName)) throw new Error("Tên ZIP chỉ được chứa chữ ASCII, số, gạch dưới hoặc gạch nối.");
      if (!submissionFiles.length) throw new Error("Gói submission chưa có file query nào.");
      const packaged = await api.packageSubmission({ archive_name: archiveName, files: submissionFiles });
      saveBlob(packaged.blob, packaged.filename);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  const valid = mode === "trake" ? events.filter((x) => x.trim()).length >= 2 : Boolean(query.trim());
  return <div className="app">
    <header><div><span className="eyebrow">AIC 2026 · operator console</span><h1>NOVA Retrieval</h1></div><Status health={health} /></header>
    <nav>{[["kis_text","t-KIS"],["kis_video","v-KIS (ngoài 3 mode sơ tuyển)"],["trake","TRAKE"],["vqa","Q&A (VQA)"]].map(([id,label]) => <button key={id} className={mode===id?"active":""} onClick={() => { setMode(id); setResult(null); setSelected([]); setParaphrases([]); setRewriteMeta(null); setVqaSubmission(null); }}>{label}</button>)}</nav>
    <main>
      <section className="control panel">
        <label>Tên file query BTC</label>
        <input value={queryFilename} onChange={(e)=>setQueryFilename(e.target.value)} placeholder={submissionMode === "kis" ? "query-1-kis.txt" : submissionMode === "qa" ? "query-3-qa.txt" : "query-4-trake.txt"} />
        <p className="hint">Nhập đúng tên file .txt BTC cung cấp; hệ thống sẽ đổi sang .csv.</p>
        {mode === "trake" ? <TRAKEEditor events={events} setEvents={(next) => { setEvents(next); setResult(null); }} /> : <>
          <label>{mode === "vqa" ? "Câu hỏi" : mode === "kis_video" ? "Mô tả clip đang xem" : "Mô tả cần tìm"}</label>
          <textarea value={query} onChange={(e)=>changeQuery(e.target.value)} placeholder={mode === "kis_video" ? "Bạn xem video BTC chiếu rồi tự mô tả cảnh ở đây…" : "Nhập truy vấn bằng tiếng Việt hoặc tiếng Anh…"} />
          {mode.startsWith("kis") && <div className="rewrite"><button className="secondary" onClick={rewrite} title={health?.checks?.["rewrite.enabled"]==="true"?"GPT rewrite đã sẵn sàng":"Bật QUERY_REWRITE_ENABLED trên API"} disabled={busy || !query.trim() || health?.checks?.["rewrite.enabled"]!=="true"}>LLM rewrite</button>{rewriteMeta?.model_id&&<small>{rewriteMeta.model_id}</small>}{paraphrases.map((x)=><span key="q1">q1: {x}</span>)}{rewriteMeta?.warnings?.map((x)=><span className="warning" key={x}>{x}</span>)}</div>}
          {mode === "vqa" && <select value={answerType} onChange={(e)=>setAnswerType(e.target.value)}><option value="short_text">Short text</option><option value="yes_no">Yes / No</option><option value="number">Number</option><option value="list">List</option></select>}
        </>}
        {mode.startsWith("kis") && <ObjectConstraints labels={labels} source={catalogSource} items={constraints} setItems={setConstraints} />}
        {mode.startsWith("kis") && <BranchPicker enabled={enabledBranches} setEnabled={setEnabledBranches} />}
        <button className="run" onClick={run} disabled={busy || !valid}>{busy ? "Đang xử lý…" : "Tìm kiếm"}</button>
        {error && <p className="error">{error}</p>}
      </section>
      <section className="results">
        <ResultSummary mode={mode} result={result} />
        {mode.startsWith("kis") && <div className="grid">{candidates.map((c,i)=><FrameCard key={c.frame_id} candidate={c} rank={i+1} selected={selectionKeys.has(`${c.video_id}:${c.source_frame_idx}`)} toggle={()=>toggleCandidate(c)} inspect={()=>setInspect(c)} />)}</div>}
        {mode === "trake" && <TRAKEResults result={result} inspect={setInspect} />}
        {mode === "vqa" && <VQAResult result={result} inspect={setInspect} onSubmissionChange={setVqaSubmission} />}
      </section>
    </main>
    <SubmissionBuilder files={submissionFiles} setFiles={setSubmissionFiles} archiveName={archiveName} setArchiveName={setArchiveName} saveCurrent={saveCurrentSubmission} downloadPackage={downloadSubmissionPackage} busy={busy} />
    {mode.startsWith("kis") && <aside className="tray"><strong>Selection {selected.length}/100</strong><span>BTC frame_id = source_frame_idx</span><button onClick={saveCurrentSubmission} disabled={!selected.length}>Lưu query vào gói</button><button className="ghost" onClick={()=>setSelected([])} disabled={!selected.length}>Xóa</button></aside>}
    {inspect && <Inspector item={inspect} close={()=>setInspect(null)} />}
  </div>;
}

function Status({health}) { const ok=health?.status==="ready"||health?.status==="healthy"; const demo=health?.checks?.demo==="true";return <div className={`status ${ok?"ok":"bad"}`}><i />{health?.status||"checking"}{demo&&" · DEMO"}</div>; }
function TRAKEEditor({events,setEvents}) { const move=(i,d)=>{const next=[...events];[next[i],next[i+d]]=[next[i+d],next[i]];setEvents(next)};return <div><label>Chuỗi sự kiện theo đúng thứ tự</label>{events.map((x,i)=><div className="event" key={i}><b>{i+1}</b><input value={x} onChange={(e)=>setEvents(events.map((v,j)=>j===i?e.target.value:v))} placeholder={`Sự kiện ${i+1}`} /><span><button className="ghost" disabled={i===0} onClick={()=>move(i,-1)}>↑</button><button className="ghost" disabled={i===events.length-1} onClick={()=>move(i,1)}>↓</button><button className="ghost" onClick={()=>events.length>2&&setEvents(events.filter((_,j)=>j!==i))}>×</button></span></div>)}<button className="secondary" onClick={()=>setEvents([...events,""])}>+ Thêm sự kiện</button></div>; }
function ObjectConstraints({labels,source,items,setItems}) { const known=new Set(labels.map(x=>x.label.toLocaleLowerCase()));const hardAllowed=source==="sqlite"||source==="demo_fixture";return <details open><summary>Object constraints <small>{source}</small></summary><p className="hint">Chỉ nhận đúng nhãn trong catalog Offline. Hard filter chỉ mở khi catalog SQLite sẵn sàng.</p><datalist id="object-labels">{labels.map((x)=><option key={x.label} value={x.label}>{x.detection_count}</option>)}</datalist>{items.map((x)=>{const invalid=!known.has(x.label.trim().toLocaleLowerCase());return <div className={`constraint ${invalid?"invalid":""}`} key={x.id}><input list="object-labels" value={x.label} aria-invalid={invalid} onChange={(e)=>setItems(items.map(v=>v.id===x.id?{...v,label:e.target.value}:v))} placeholder="Chọn object"/><select value={x.count} onChange={(e)=>setItems(items.map(v=>v.id===x.id?{...v,count:Number(e.target.value)}:v))}><option value={1}>1+</option><option value={2}>2+</option><option value={3}>3+</option></select><select value={x.filter_mode} onChange={(e)=>setItems(items.map(v=>v.id===x.id?{...v,filter_mode:e.target.value}:v))}><option value="soft">Soft boost</option><option value="hard" disabled={!hardAllowed}>Hard filter</option></select><button className="ghost" onClick={()=>setItems(items.filter(v=>v.id!==x.id))}>×</button></div>})}<button className="secondary" disabled={!labels.length} onClick={()=>setItems([...items,{id:uid(),label:labels[0]?.label||"",count_operator:"gte",count:1,min_confidence:.5,position:null,filter_mode:"soft"}])}>+ Object</button></details>; }
function BranchPicker({enabled,setEnabled}) { return <details><summary>7 retrieval branches <small>{enabled.length}/7</small></summary><div className="branches">{BRANCHES.map((b)=><label key={b}><input type="checkbox" checked={enabled.includes(b)} onChange={()=>setEnabled(enabled.includes(b)?enabled.filter(x=>x!==b):[...enabled,b])}/>{b}</label>)}</div></details>; }
function ResultSummary({mode,result}) { if(!result)return <div className="empty">Kết quả sẽ xuất hiện ở đây.</div>; const n=mode.startsWith("kis")?result.candidates?.length:mode==="trake"?result.results?.length:result.result?.evidence?.length; return <div className="result-head"><div><span className="eyebrow">{mode}</span><h2>{n||0} kết quả / evidence</h2></div>{result.diagnostics&&<details><summary>Diagnostics</summary><pre>{JSON.stringify(result.diagnostics,null,2)}</pre></details>}</div>; }
function FrameCard({candidate:c,rank,selected,toggle,inspect}) { return <article className={`card ${selected?"chosen":""}`}><div className="thumb" onClick={inspect}><img src={api.imageUrl(c.frame_id)} alt={c.frame_id}/><em>#{rank}</em><span>{c.timestamp_sec.toFixed(2)}s</span></div><div className="meta"><strong>{c.video_id}</strong><code>BTC frame {c.source_frame_idx}</code><small>{c.frame_id}</small><div className="score"><i style={{width:`${Math.max(2,Math.min(100,c.final_score*100))}%`}}/>score {c.final_score.toFixed(4)}</div><button onClick={toggle}>{selected?"Đã chọn":"Chọn nộp"}</button></div></article>; }
function TRAKEResults({result,inspect}) { return <div className="sequences">{result?.results?.map((r,i)=><article key={`${r.video_id}-${i}`}><h3>#{i+1} {r.video_id} <small>{r.score.toFixed(4)}</small></h3><div className="timeline">{r.sequence.map((x,j)=><div key={x.event_id} onClick={()=>inspect(x)}><img src={api.imageUrl(x.frame_id)} alt={x.frame_id}/><b>{j+1}</b><span>{x.timestamp_sec.toFixed(2)}s · frame {x.source_frame_idx}</span></div>)}</div></article>)}</div>; }
function VQAResult({result,inspect,onSubmissionChange}) { const r=result?.result;const [answer,setAnswer]=useState("");const [selectedEvidenceId,setSelectedEvidenceId]=useState(null);const responseAnswer=r?.response?.answer;const evidenceKey=r?.response?.evidence_ids?.join("|")||"";useEffect(()=>setAnswer(responseAnswer||""),[responseAnswer]);useEffect(()=>setSelectedEvidenceId(null),[evidenceKey]);const cited=new Set(r?.response?.evidence_ids||[]);const images=r?.evidence?.filter(x=>x.evidence_type==="image")||[];const selectedEvidence=images.find(x=>x.evidence_id===selectedEvidenceId&&cited.has(x.evidence_id));const row=selectedEvidence&&answer.trim()&&answer.length<=100?{video_id:selectedEvidence.video_id,frame_id:selectedEvidence.source_frame_idx,answer}:null;useEffect(()=>onSubmissionChange(row),[onSubmissionChange,selectedEvidenceId,answer,evidenceKey]);if(!r)return null;return <article className="answer"><span className={`pill ${r.response.status}`}>{r.response.status}</span><h2>{r.response.answer||"Không đủ evidence"}</h2><p>Confidence: {r.response.confidence}</p>{r.response.status==="answered"&&<label>Đáp án nộp (tối đa 100 ký tự)<input value={answer} maxLength={100} onChange={e=>setAnswer(e.target.value)}/><small>{answer.length}/100</small></label>}{r.response.status==="answered"&&<p className="hint">Chọn rõ một frame được VLM trích dẫn làm frame nộp. Khoảng trắng đầu/cuối được giữ nguyên theo quy định BTC.</p>}<div className="evidence">{images.map(x=><div className={`vqa-evidence ${x.evidence_id===selectedEvidenceId?"chosen":""} ${!cited.has(x.evidence_id)?"uncited":""}`} key={x.evidence_id}><button disabled={!cited.has(x.evidence_id)} onClick={()=>setSelectedEvidenceId(x.evidence_id)}><img src={api.imageUrl(x.frame_id)} alt={x.frame_id}/><span>{x.evidence_id} · frame {x.source_frame_idx}</span></button><button className="ghost" onClick={()=>inspect(x)}>Xem</button></div>)}</div>{row&&<p className="hint">Evidence đã chọn: {row.video_id}, frame {row.frame_id}</p>}<details><summary>Grounded evidence</summary><pre>{JSON.stringify(r.evidence,null,2)}</pre></details></article>; }
function Inspector({item,close}) { const frameId=item.frame_id;const [neighbors,setNeighbors]=useState([]);useEffect(()=>{api.neighbors(frameId).then(x=>setNeighbors(x.frames)).catch(()=>setNeighbors([]));},[frameId]);return <div className="overlay" onClick={close}><div className="modal" onClick={e=>e.stopPropagation()}><button className="close" onClick={close}>×</button><div><img className="hero" src={api.imageUrl(frameId)} alt={frameId}/><div className="neighbor-strip">{neighbors.map(x=><img className={x.frame_id===frameId?"current":""} key={x.frame_id} src={api.imageUrl(x.frame_id)} alt={x.frame_id}/>)}</div></div><div><h2>{item.video_id}</h2><p>timestamp {Number(item.timestamp_sec).toFixed(3)}s · BTC frame {item.source_frame_idx}</p><video src={`${api.videoUrl(item.video_id)}#t=${Math.max(0,Number(item.timestamp_sec)-2)}`} controls preload="metadata"/><pre>{JSON.stringify(item,null,2)}</pre></div></div></div>; }
function SubmissionBuilder({files,setFiles,archiveName,setArchiveName,saveCurrent,downloadPackage,busy}) { return <section className="submission-builder panel"><div><span className="eyebrow">BTC preliminary submission</span><h2>Gói nộp bài</h2><p className="hint">Mỗi query là một CSV UTF-8 không header. ZIP luôn chứa thư mục submission/.</p></div><button className="secondary" onClick={saveCurrent} disabled={busy}>Lưu kết quả query hiện tại</button><div className="package-files">{files.length?files.map((file)=><span key={csvName(file.query_filename)}><code>{csvName(file.query_filename)}</code> · {file.rows.length} dòng<button className="ghost" onClick={()=>setFiles(files.filter(x=>x!==file))}>×</button></span>):<small>Chưa có file nào.</small>}</div><label>Tên ZIP<input value={archiveName} onChange={e=>setArchiveName(e.target.value)} placeholder="team_ABC_round1"/></label><button className="run" onClick={downloadPackage} disabled={busy||!files.length}>{busy?"Đang tạo ZIP…":"Tải ZIP đúng chuẩn BTC"}</button></section>; }
function validateQueryFilename(value,mode) { const name=value.trim();const match=name.match(/^[A-Za-z0-9][A-Za-z0-9_-]*-(kis|qa|trake)(?:\.(txt|csv))?$/i);if(!match)throw new Error("Tên query phải giống query-1-kis.txt, query-3-qa.txt hoặc query-4-trake.txt.");if(match[1].toLowerCase()!==mode)throw new Error("Hậu tố tên query không khớp mode đang chọn.");return name; }
function csvName(value) { return `${value.replace(/\.(txt|csv)$/i,"")}.csv`; }
function saveBlob(blob,filename) { const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),0); }
