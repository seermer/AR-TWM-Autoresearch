import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { q, raw, useApi } from "../../api";
import { Badge, Chart, Chips, Drawer, Empty, fmtBytes, fmtDur, Json, KV, LazyVideo, Markdown, PALETTE, Q, Section, Table, useParam } from "../../ui";

function CameraPath({ path }: { path: { x: number[]; y: number[]; z: number[] } }) {
  const [gl, setGl] = useState(false);
  useEffect(() => { import("echarts-gl").then(() => setGl(true)); }, []);
  const pts = path.x.map((x, i) => [x, path.y[i], path.z[i]]);
  return (
    <div className="grid2">
      {gl ? <Chart height={340} option={{
        tooltip: {}, visualMap: { show: false, dimension: 3, min: 0, max: pts.length - 1, inRange: { color: ["#5b8ff9", "#f6bd16", "#e8684a"] } },
        xAxis3D: { name: "x" }, yAxis3D: { name: "y" }, zAxis3D: { name: "z" }, grid3D: { viewControl: { autoRotate: false } },
        series: [{ type: "line3D", data: pts.map((p, i) => [...p, i]), lineStyle: { width: 3 } },
          { type: "scatter3D", data: [[...pts[0], 0]], symbolSize: 10, itemStyle: { color: "#30a46c" } }],
      }} /> : <div className="muted pad">Loading 3D…</div>}
      <Chart height={340} option={{
        tooltip: { formatter: (p: any) => `frame ${p.dataIndex}<br/>x ${p.data[0].toFixed(3)}, z ${p.data[1].toFixed(3)}` },
        title: { text: "top-down (x vs z), green = first frame", textStyle: { fontSize: 11 } },
        grid: { left: 50, right: 20, top: 30, bottom: 30 },
        xAxis: { type: "value", scale: true, name: "x" }, yAxis: { type: "value", scale: true, name: "z" },
        series: [{ type: "line", data: pts.map((p) => [p[0], p[2]]), symbolSize: 3, lineStyle: { color: PALETTE[0] } },
          { type: "scatter", data: [[pts[0][0], pts[0][2]]], symbolSize: 10, itemStyle: { color: "#30a46c" } }],
      }} />
    </div>
  );
}

/** The clip with its caption segments and annotation turns on the video's own time axis. */
function ClipVideo({ clip }: { clip: any }) {
  const video = useRef<HTMLVideoElement>(null);
  const [t, setT] = useState(0);
  const duration = clip.metadata?.duration || 1;
  const segments = clip.caption?.segments ?? [];
  const turns = clip.annotation?.turns ?? [];
  const seek = (s: number) => { if (video.current) { video.current.currentTime = s; video.current.play().catch(() => {}); } };
  const active = segments.findIndex((s: any) => t >= s.time_range_s[0] && t < s.time_range_s[1]);
  return (
    <div>
      <video ref={video} src={raw(clip.video)} controls muted playsInline className="clip-video" onTimeUpdate={(e) => setT(e.currentTarget.currentTime)} />
      <div className="seg-track" title="caption segments">
        {segments.map((s: any, i: number) => (
          <div key={i} className={`seg ${i === active ? "active" : ""}`} onClick={() => seek(s.time_range_s[0])}
            style={{ left: `${(s.time_range_s[0] / duration) * 100}%`, width: `${((s.time_range_s[1] - s.time_range_s[0]) / duration) * 100}%`, background: PALETTE[i % PALETTE.length] }}>
            {i + 1}
          </div>
        ))}
        {turns.map((u: any, i: number) => <div key={`t${i}`} className="turn-mark" style={{ left: `${(u.start_s / duration) * 100}%` }} title={`turn ${i + 1} at ${u.start_s}s: ${JSON.stringify(u)}`} onClick={() => seek(u.start_s)} />)}
        <div className="playhead" style={{ left: `${(t / duration) * 100}%` }} />
      </div>
      <div className="muted small">{t.toFixed(2)} s of {duration.toFixed(2)} s · coloured bars = caption segments (click to seek) · white ticks = annotation turns</div>
      <Table rows={segments.map((s: any, i: number) => ({ "#": i + 1, from: s.time_range_s[0], to: s.time_range_s[1], prompt: s.prompt }))} filter={false} maxHeight={260}
        rowClass={(r: any) => (r["#"] - 1 === active ? "selected" : "")} onRowClick={(r: any) => seek(r.from)}
        cols={[{ key: "#", width: "40px" }, { key: "from", width: "70px" }, { key: "to", width: "70px" }, { key: "prompt", width: "minmax(200px, 1fr)" }]} />
      {turns.length > 0 && <Table rows={turns.map((u: any, i: number) => ({ "#": i + 1, ...u }))} filter={false} maxHeight={200}
        cols={[{ key: "#", width: "40px" }, ...Object.keys(turns[0]).map((k) => ({ key: k }))]} />}
    </div>
  );
}

function ClipDetail({ id, onClose }: { id: string | null; onClose: () => void }) {
  const query = useApi(id ? `/api/clip/${id}` : null);
  return (
    <Drawer open={!!id} onClose={onClose} title={`clip ${id?.slice(0, 16)}…`} wide>
      <Q query={query} what="clip">{(c) => (
        <div>
          <KV items={[["clip id", <span className="mono">{c.clip_id}</span>], ["derived from", `${c.derived_from?.kind}: ${c.derived_from?.name}`], ["ingested by", <Link to={`/node/${c.ingested_by}/data`}>{c.ingested_by}</Link>],
            ["frames", c.metadata?.frames], ["fps", c.metadata?.fps], ["size", c.metadata?.width && `${c.metadata.width} × ${c.metadata.height}`], ["duration", fmtDur(c.metadata?.duration)],
            ["turns", c.metadata?.turns], ["warnings", c.warnings?.length ? c.warnings.join("; ") : "none"],
            ["leakage", c.leakage ? `${c.leakage.matches?.length ?? 0} matches / ${c.leakage.near?.length ?? 0} near` : "no record"],
            ["files", <span className="row gap"><Link to={`/files?${q({ path: c.video })}`}>video</Link><Link to={`/files?${q({ path: c.pose })}`}>pose</Link></span>]]} />
          <ClipVideo clip={c} />
          {c.annotation?.scene_prompt && <Section title="scene prompt (annotation)"><Markdown text={c.annotation.scene_prompt} /></Section>}
          <Section title="camera path (pose)">{c.camera_path ? <CameraPath path={c.camera_path} /> : <div className="muted">no readable pose</div>}</Section>
          <Section title="caption"><Json value={c.caption} open={3} /></Section>
          <Section title="annotation"><Json value={c.annotation} open={3} /></Section>
          <Section title="leakage check"><Json value={c.leakage} open={3} /></Section>
          <Section title="clip record" collapsed><Json value={c} open={1} /></Section>
        </div>
      )}</Q>
    </Drawer>
  );
}

function Gallery({ id, dataset, onOpen }: { id: string; dataset: string; onOpen: (c: string) => void }) {
  const query = useApi(`/api/clips/${encodeURIComponent(id)}/${encodeURIComponent(dataset)}`);
  const [filter, setFilter] = useState("");
  const [view, setView] = useState<"grid" | "table">("grid");
  return (
    <Q query={query} what="clips">{(rows: any[]) => {
      const shown = rows.filter((r) => !filter || JSON.stringify(r).toLowerCase().includes(filter.toLowerCase()));
      return (
        <div>
          <div className="row gap">
            <input className="search" placeholder="filter clips (origin, id, warnings…)" value={filter} onChange={(e) => setFilter(e.target.value)} />
            <span className="muted small">{shown.length} of {rows.length} clips</span>
            <Chips items={["grid", "table"] as const} value={view} onChange={setView as any} />
          </div>
          {view === "grid" ? (
            <div className="gallery">
              {shown.map((c) => (
                <div key={c.clip_id} className="clip-card" onClick={() => onOpen(c.clip_id)}>
                  <LazyVideo src={raw(c.video)} controls={false} />
                  <div className="small clip-meta">
                    <div className="ellipsis" title={c.origin}>{c.origin}</div>
                    <div className="muted">{c.metadata?.frames} frames · {fmtDur(c.metadata?.duration)} · {c.metadata?.turns} turns</div>
                    <div>{c.warnings?.length ? <Badge tone="warn">{c.warnings.length} warnings</Badge> : null}
                      {c.leakage && (c.leakage.matches?.length || c.leakage.near?.length) ? <Badge tone="bad">leakage {c.leakage.matches?.length}/{c.leakage.near?.length}</Badge> : null}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <Table rows={shown} rowKey={(r: any) => r.clip_id} onRowClick={(r: any) => onOpen(r.clip_id)} maxHeight={600} cols={[
              { key: "clip_id", width: "160px", mono: true, render: (r: any) => r.clip_id.slice(0, 16) }, { key: "origin", width: "minmax(200px, 2fr)" },
              { key: "frames", value: (r: any) => r.metadata?.frames }, { key: "duration", value: (r: any) => r.metadata?.duration }, { key: "turns", value: (r: any) => r.metadata?.turns },
              { key: "warnings", value: (r: any) => r.warnings?.join("; ") },
              { key: "leakage", value: (r: any) => r.leakage ? `${r.leakage.matches?.length ?? 0} matches / ${r.leakage.near?.length ?? 0} near` : "" }]} />
          )}
        </div>
      );
    }}</Q>
  );
}

export default function DataTab({ id }: { id: string }) {
  const query = useApi(`/api/data/${encodeURIComponent(id)}`);
  const [dataset, setDataset] = useParam("dataset", "");
  const [clip, setClip] = useParam("clip", "");
  return (
    <Q query={query} what="training data">{(d) => {
      const ds = dataset || d.datasets[0]?.dataset;
      return (
        <div>
          {d.commit ? (
            <Section title="Data commit" note={d.commit.commit_id}>
              <KV items={[["message", d.commit.message], ["parent commit", d.commit.parent_commit], ["node", d.commit.node_id], ["attempt", d.commit.attempt]]} />
              <Section title="manifest" collapsed><Json value={(() => { try { return JSON.parse(d.commit.manifest); } catch { return d.commit.manifest; } })()} open={2} /></Section>
            </Section>
          ) : <div className="note">No data commit (baseline, or not built yet).</div>}
          {d.datasets.length > 0 && (
            <Section title="Clips">
              <Chips items={d.datasets.map((x: any) => x.dataset)} value={ds} onChange={setDataset} render={(k) => `${k} (${d.datasets.find((x: any) => x.dataset === k).clips})`} />
              {ds && <Gallery id={id} dataset={ds} onOpen={setClip} />}
            </Section>
          )}
          <Section title={`Every ingest candidate (${d.ingests.length})`}>
            {d.ingests.length ? <Table rows={d.ingests} maxHeight={500} cols={[
              { key: "time", width: "150px" }, { key: "phase", width: "110px" }, { key: "attempt", width: "60px" },
              { key: "outcome", width: "90px", render: (r: any) => <Badge>{r.outcome}</Badge> },
              { key: "clip_id", width: "130px", render: (r: any) => r.clip_id ? <button className="btn tiny" onClick={() => setClip(r.clip_id)}>{r.clip_id.slice(0, 12)}</button> : "" },
              { key: "reasons", width: "minmax(180px, 2fr)" }, { key: "derived_from", width: "minmax(140px, 1fr)" },
              { key: "video", width: "minmax(180px, 2fr)", mono: true, render: (r: any) => r.host_video ? <Link to={`/files?${q({ path: r.host_video })}`}>{r.video}</Link> : r.video },
              { key: "annotation", width: "minmax(140px, 1fr)", mono: true }, { key: "pose", width: "minmax(140px, 1fr)", mono: true }]} /> : <Empty>No ingest calls.</Empty>}
          </Section>
          <Section title={`Kept staging files (${d.staging.length})`} collapsed={!d.staging.length}>
            <Table rows={d.staging} maxHeight={400} cols={[{ key: "path", width: "minmax(300px, 4fr)", mono: true, render: (r: any) => <Link to={`/files?${q({ path: r.path })}`}>{r.path}</Link> },
              { key: "bytes", width: "100px", render: (r: any) => fmtBytes(r.bytes) }]} />
          </Section>
          <ClipDetail id={clip || null} onClose={() => setClip(null)} />
        </div>
      );
    }}</Q>
  );
}
