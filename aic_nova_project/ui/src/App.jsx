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

  useEffect(() => {
    api.catalog().then((x) => { setLabels(x.labels); setCatalogSource(x.source); }).catch(() => setCatalogSource("unavailable"));
    api.health().then(setHealth).catch(() => setHealth({ status: "unavailable", checks: {} }));
  }, []);

  const candidates = result?.candidates || [];
  const selectionKeys = useMemo(() => new Set(selected.map((x) => `${x.video_id}:${x.source_frame_idx}`)), [selected]);

  async function rewrite() {
    if (!query.trim()) return;
    setBusy(true); setError("");
    try { const x = await api.rewrite(query.trim(), `rw-${uid()}`); setParaphrases(x.paraphrases); setRewriteMeta(x); }
    catch (e) { setError(e.message); setParaphrases([]); setRewriteMeta(null); }
    finally { setBusy(false); }
  }

  async function run() {
    setBusy(true); setError(""); setResult(null);
    try {
      if (mode.startsWith("kis")) {
        setResult(await api.search({
          query: query.trim(), mode, paraphrases,
          object_constraints: constraints.map(({ id, ...item }) => item),
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

  function toggleCandidate(candidate) {
    const key = `${candidate.video_id}:${candidate.source_frame_idx}`;
    setSelected((old) => selectionKeys.has(key) ? old.filter((x) => `${x.video_id}:${x.source_frame_idx}` !== key) : [...old, candidate].slice(0, 100));
  }

  function downloadSubmission() {
    const logicalRows = selected.map((x) => ({ video_id: x.video_id, frame_id: x.source_frame_idx }));
    downloadJson(logicalRows, "aic-kis-logical-submission.json");
  }

  const valid = mode === "trake" ? events.filter((x) => x.trim()).length >= 2 : Boolean(query.trim());
  return <div className="app">
    <header><div><span className="eyebrow">AIC 2026 · operator console</span><h1>NOVA Retrieval</h1></div><Status health={health} /></header>
    <nav>{[["kis_text","t-KIS"],["kis_video","v-KIS"],["trake","TRAKE"],["vqa","VQA"]].map(([id,label]) => <button key={id} className={mode===id?"active":""} onClick={() => { setMode(id); setResult(null); }}>{label}</button>)}</nav>
    <main>
      <section className="control panel">
        {mode === "trake" ? <TRAKEEditor events={events} setEvents={setEvents} /> : <>
          <label>{mode === "vqa" ? "Câu hỏi" : mode === "kis_video" ? "Mô tả clip đang xem" : "Mô tả cần tìm"}</label>
          <textarea value={query} onChange={(e)=>setQuery(e.target.value)} placeholder={mode === "kis_video" ? "Bạn xem video BTC chiếu rồi tự mô tả cảnh ở đây…" : "Nhập truy vấn bằng tiếng Việt hoặc tiếng Anh…"} />
          {mode.startsWith("kis") && <div className="rewrite"><button className="secondary" onClick={rewrite} title={health?.checks?.["rewrite.enabled"]==="true"?"GPT rewrite đã sẵn sàng":"Bật QUERY_REWRITE_ENABLED trên API"} disabled={busy || !query.trim() || health?.checks?.["rewrite.enabled"]!=="true"}>LLM rewrite</button>{rewriteMeta?.model_id&&<small>{rewriteMeta.model_id}</small>}{paraphrases.map((x,i)=><span key={i}>q{i+1}: {x}</span>)}</div>}
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
        {mode === "vqa" && <VQAResult result={result} inspect={setInspect} />}
      </section>
    </main>
    {mode.startsWith("kis") && <aside className="tray"><strong>Selection {selected.length}/100</strong><span>BTC frame_id = source_frame_idx</span><button onClick={downloadSubmission} disabled={!selected.length}>Xuất logical JSON</button><button className="ghost" onClick={()=>setSelected([])} disabled={!selected.length}>Xóa</button></aside>}
    {inspect && <Inspector item={inspect} close={()=>setInspect(null)} />}
  </div>;
}

function Status({health}) { const ok=health?.status==="ready"||health?.status==="healthy"; const demo=health?.checks?.demo==="true";return <div className={`status ${ok?"ok":"bad"}`}><i />{health?.status||"checking"}{demo&&" · DEMO"}</div>; }
function TRAKEEditor({events,setEvents}) { const move=(i,d)=>{const next=[...events];[next[i],next[i+d]]=[next[i+d],next[i]];setEvents(next)};return <div><label>Chuỗi sự kiện theo đúng thứ tự</label>{events.map((x,i)=><div className="event" key={i}><b>{i+1}</b><input value={x} onChange={(e)=>setEvents(events.map((v,j)=>j===i?e.target.value:v))} placeholder={`Sự kiện ${i+1}`} /><span><button className="ghost" disabled={i===0} onClick={()=>move(i,-1)}>↑</button><button className="ghost" disabled={i===events.length-1} onClick={()=>move(i,1)}>↓</button><button className="ghost" onClick={()=>events.length>2&&setEvents(events.filter((_,j)=>j!==i))}>×</button></span></div>)}<button className="secondary" onClick={()=>setEvents([...events,""])}>+ Thêm sự kiện</button></div>; }
function ObjectConstraints({labels,source,items,setItems}) { return <details open><summary>Object constraints <small>{source}</small></summary><datalist id="object-labels">{labels.map((x)=><option key={x.label} value={x.label}>{x.detection_count}</option>)}</datalist>{items.map((x)=><div className="constraint" key={x.id}><input list="object-labels" value={x.label} onChange={(e)=>setItems(items.map(v=>v.id===x.id?{...v,label:e.target.value}:v))} placeholder="object"/><select value={x.count} onChange={(e)=>setItems(items.map(v=>v.id===x.id?{...v,count:Number(e.target.value)}:v))}><option value={1}>1+</option><option value={2}>2+</option><option value={3}>3+</option></select><select value={x.filter_mode} onChange={(e)=>setItems(items.map(v=>v.id===x.id?{...v,filter_mode:e.target.value}:v))}><option value="soft">Soft boost</option><option value="hard">Hard filter</option></select><button className="ghost" onClick={()=>setItems(items.filter(v=>v.id!==x.id))}>×</button></div>)}<button className="secondary" onClick={()=>setItems([...items,{id:uid(),label:"person",count_operator:"gte",count:1,min_confidence:.5,position:null,filter_mode:"soft"}])}>+ Object</button></details>; }
function BranchPicker({enabled,setEnabled}) { return <details><summary>7 retrieval branches <small>{enabled.length}/7</small></summary><div className="branches">{BRANCHES.map((b)=><label key={b}><input type="checkbox" checked={enabled.includes(b)} onChange={()=>setEnabled(enabled.includes(b)?enabled.filter(x=>x!==b):[...enabled,b])}/>{b}</label>)}</div></details>; }
function ResultSummary({mode,result}) { if(!result)return <div className="empty">Kết quả sẽ xuất hiện ở đây.</div>; const n=mode.startsWith("kis")?result.candidates?.length:mode==="trake"?result.results?.length:result.result?.evidence?.length; return <div className="result-head"><div><span className="eyebrow">{mode}</span><h2>{n||0} kết quả / evidence</h2></div>{result.diagnostics&&<details><summary>Diagnostics</summary><pre>{JSON.stringify(result.diagnostics,null,2)}</pre></details>}</div>; }
function FrameCard({candidate:c,rank,selected,toggle,inspect}) { return <article className={`card ${selected?"chosen":""}`}><div className="thumb" onClick={inspect}><img src={api.imageUrl(c.frame_id)} alt={c.frame_id}/><em>#{rank}</em><span>{c.timestamp_sec.toFixed(2)}s</span></div><div className="meta"><strong>{c.video_id}</strong><code>BTC frame {c.source_frame_idx}</code><small>{c.frame_id}</small><div className="score"><i style={{width:`${Math.max(2,Math.min(100,c.final_score*100))}%`}}/>score {c.final_score.toFixed(4)}</div><button onClick={toggle}>{selected?"Đã chọn":"Chọn nộp"}</button></div></article>; }
function TRAKEResults({result,inspect}) { const rows=result?.results?.map(r=>({video_id:r.video_id,frame_ids:r.sequence.map(x=>x.source_frame_idx)}))||[];return <div className="sequences">{!!rows.length&&<button className="secondary" onClick={()=>downloadJson(rows,"aic-trake-logical-submission.json")}>Xuất TRAKE logical JSON</button>}{result?.results?.map((r,i)=><article key={`${r.video_id}-${i}`}><h3>#{i+1} {r.video_id} <small>{r.score.toFixed(4)}</small></h3><div className="timeline">{r.sequence.map((x,j)=><div key={x.event_id} onClick={()=>inspect(x)}><img src={api.imageUrl(x.frame_id)} alt={x.frame_id}/><b>{j+1}</b><span>{x.timestamp_sec.toFixed(2)}s · frame {x.source_frame_idx}</span></div>)}</div></article>)}</div>; }
function VQAResult({result,inspect}) { const r=result?.result;const [answer,setAnswer]=useState("");useEffect(()=>setAnswer(r?.response?.answer||""),[r?.response?.answer]);if(!r)return null;const grounded=r.evidence.find(x=>x.evidence_type==="image"&&r.response.evidence_ids.includes(x.evidence_id));const row=grounded&&answer.trim()?{video_id:grounded.video_id,frame_id:grounded.source_frame_idx,answer:answer.trim()}:null;return <article className="answer"><span className={`pill ${r.response.status}`}>{r.response.status}</span><h2>{r.response.answer||"Không đủ evidence"}</h2><p>Confidence: {r.response.confidence}</p>{r.response.status==="answered"&&<label>Đáp án nộp (có thể chỉnh)<input value={answer} onChange={e=>setAnswer(e.target.value)}/></label>}{row&&<button className="secondary" onClick={()=>downloadJson(row,"aic-vqa-logical-submission.json")}>Xuất VQA logical JSON</button>}<div className="evidence">{r.evidence.filter(x=>x.evidence_type==="image").map(x=><img key={x.evidence_id} src={api.imageUrl(x.frame_id)} alt={x.frame_id} onClick={()=>inspect(x)}/>)}</div><details><summary>Grounded evidence</summary><pre>{JSON.stringify(r.evidence,null,2)}</pre></details></article>; }
function Inspector({item,close}) { const frameId=item.frame_id;const [neighbors,setNeighbors]=useState([]);useEffect(()=>{api.neighbors(frameId).then(x=>setNeighbors(x.frames)).catch(()=>setNeighbors([]));},[frameId]);return <div className="overlay" onClick={close}><div className="modal" onClick={e=>e.stopPropagation()}><button className="close" onClick={close}>×</button><div><img className="hero" src={api.imageUrl(frameId)} alt={frameId}/><div className="neighbor-strip">{neighbors.map(x=><img className={x.frame_id===frameId?"current":""} key={x.frame_id} src={api.imageUrl(x.frame_id)} alt={x.frame_id}/>)}</div></div><div><h2>{item.video_id}</h2><p>timestamp {Number(item.timestamp_sec).toFixed(3)}s · BTC frame {item.source_frame_idx}</p><video src={`${api.videoUrl(item.video_id)}#t=${Math.max(0,Number(item.timestamp_sec)-2)}`} controls preload="metadata"/><pre>{JSON.stringify(item,null,2)}</pre></div></div></div>; }
function downloadJson(value,filename) { const blob=new Blob([JSON.stringify(value,null,2)],{type:"application/json"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=filename;link.click();URL.revokeObjectURL(link.href); }
