import { useEffect, useRef, useState } from "react";
import { get, q, raw, useApi } from "../api";
import { Code } from "../parts/lazy";
import { Copy, fmtBytes, Json, KV, Q, Section, Table, Text, useParam } from "../ui";

const CODE = /\.(py|json|ya?ml|toml|sh|md|cfg|ini|html|css|ts|tsx|js)$/i;

/** Follows a growing file: polls for new bytes every 2 s and keeps the view at the bottom. */
function Follow({ path }: { path: string }) {
  const [text, setText] = useState("");
  const [info, setInfo] = useState<any>(null);
  const end = useRef(0);
  const box = useRef<HTMLPreElement>(null);
  useEffect(() => {
    let stop = false;
    end.current = 0;
    setText("");
    const poll = async () => {
      try {
        const r = await get(`/api/tail?${q({ path, offset: end.current === 0 ? -200000 : end.current })}`);
        if (stop) return;
        if (r.size < end.current) { end.current = 0; setText(""); }       // truncated or replaced: read its tail again
        else if (r.end > end.current) { setText((t) => (t + r.text).slice(-2_000_000)); end.current = r.end; }
        setInfo(r);
      } catch (e) { setInfo({ error: String(e) }); }
      if (!stop) setTimeout(poll, 2000);
    };
    poll();
    return () => { stop = true; };
  }, [path]);
  useEffect(() => { box.current?.scrollTo(0, box.current.scrollHeight); }, [text]);
  return (
    <div>
      <div className="muted small">following · {info?.size !== undefined ? `${fmtBytes(info.size)} · modified ${info.modified}` : ""} {info?.error ?? ""} · the last 200 KB, then every new byte</div>
      <pre ref={box} className="follow wrap">{text}</pre>
    </div>
  );
}

function Preview({ path }: { path: string }) {
  const query = useApi(`/api/preview?${q({ path })}`);
  const [follow, setFollow] = useState(false);
  useEffect(() => setFollow(false), [path]);
  return (
    <Q query={query} what="file">{(p) => (
      <div>
        <KV items={[["path", <span className="mono">{p.path}</span>], ["size", fmtBytes(p.bytes)], ["bytes", p.bytes], ["modified", p.modified], ["kind", p.kind],
          ["open", <span className="row gap"><a href={raw(path)} target="_blank">raw</a><Copy text={path} label="copy path" /></span>]]} />
        {p.kind === "text" && <label className="small"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow (tail -f)</label>}
        {p.kind === "text" && (follow ? <Follow path={path} /> : CODE.test(path) && (p.text?.length ?? 0) < 1_500_000 ? <Code text={p.text} path={path} height={window.innerHeight - 330} /> : <Text text={p.text} lines={2000} />)}
        {p.kind === "video" && <video src={raw(path)} controls muted className="clip-video" />}
        {p.kind === "image" && <img src={raw(path)} className="preview-img" />}
        {p.kind === "npz" && <Table rows={p.arrays} filter={false} cols={[{ key: "name" }, { key: "shape", value: (r: any) => `(${r.shape.join(", ")})` }, { key: "dtype" }]} />}
        {p.kind === "binary" && <div className="muted">Binary file: metadata only.</div>}
        {p.kind !== "text" && <Section title="record" collapsed><Json value={p} /></Section>}
      </div>
    )}</Q>
  );
}

export default function Files() {
  const [path, setPath] = useParam("path");
  const isFile = useApi(path ? `/api/preview?${q({ path })}` : null);
  const folder = isFile.data && isFile.data.kind !== "dir" ? path.split("/").slice(0, -1).join("/") : path;
  const list = useApi(`/api/files?${q({ path: folder })}`);
  const parts = folder ? folder.split("/") : [];
  return (
    <div className="page">
      <div className="crumbs">
        <button className="btn small" onClick={() => setPath(null)}>run folder</button>
        {parts.map((p, i) => <span key={i}> / <button className="btn small" onClick={() => setPath(parts.slice(0, i + 1).join("/"))}>{p}</button></span>)}
        {isFile.data && isFile.data.kind !== "dir" && <span> / <b>{path.split("/").pop()}</b></span>}
      </div>
      <div className="files-layout">
        <Section title={`/${folder}`}>
          <Q query={list} what="folder">{(rows: any[]) => (
            <Table rows={rows} rowKey={(r: any) => r.name} selected={path.split("/").pop()} maxHeight={window.innerHeight - 240}
              onRowClick={(r: any) => setPath(folder ? `${folder}/${r.name}` : r.name)} cols={[
                { key: "name", width: "minmax(160px, 3fr)", render: (r: any) => <span>{r.type === "dir" ? "📁 " : ""}{r.name}</span> },
                { key: "bytes", label: "size", width: "90px", value: (r: any) => r.bytes, render: (r: any) => fmtBytes(r.bytes) }, { key: "modified", width: "150px" }]} />
          )}</Q>
        </Section>
        <Section title={isFile.data && isFile.data.kind !== "dir" ? path.split("/").pop() : "Preview"}>
          {isFile.data && isFile.data.kind !== "dir" ? <Preview path={path} /> : <div className="muted">Pick a file. Text and JSON under 2 MB are shown in full; larger text shows its first and last 256 KB; follow tails a growing log.</div>}
        </Section>
      </div>
    </div>
  );
}
