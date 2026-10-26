import { useMemo, useState } from "react";
import { raw, useApi } from "../../api";
import { Chart, Chips, deltaClass, Empty, fmtDelta, Json, KV, LazyVideo, Q, Section, SyncedVideos, Table, useParam } from "../../ui";

const JUDGED = ["event_edit_adherence", "subject_action_adherence", "perspective_switch_adherence", "causal_fidelity"];

function JudgeTable({ mine, theirs, other }: { mine: any[]; theirs: any[]; other: string }) {
  const key = (r: any) => `${r.turn}\u0000${r.question}`;
  const b = new Map(theirs.map((r) => [key(r), r]));
  const rows = [...new Map([...mine, ...theirs].map((r) => [key(r), r])).keys()].map((k) => {
    const x = mine.find((r) => key(r) === k), y = b.get(k);
    const r = x ?? y;
    return { turn: r.turn, question: r.question, mine: x, theirs: y };
  });
  const cell = (r: any) => !r ? "" : <span className={r.passed === true ? "pos b" : r.passed === false ? "neg b" : ""}>{r.answer}</span>;
  return <Table rows={rows} maxHeight={500} cols={[{ key: "turn", width: "minmax(180px, 2fr)" }, { key: "question", width: "minmax(160px, 1.5fr)" },
    { key: "this node", width: "110px", value: (r: any) => r.mine?.answer, render: (r: any) => cell(r.mine) },
    { key: other || "compared", width: "110px", value: (r: any) => r.theirs?.answer, render: (r: any) => cell(r.theirs) }]}
    empty="No judge answers (these metrics were not judged for this case)." />;
}

function CaseDetail({ id, other, caseId }: { id: string; other: string; caseId: string }) {
  const query = useApi(`/api/eval/${encodeURIComponent(id)}/${encodeURIComponent(caseId)}?other=${encodeURIComponent(other)}`);
  return (
    <Q query={query} what="case">{(d) => {
      const a = d.node, b = d.other;
      if (!a) return <Empty>No case {caseId} for {id}.</Empty>;
      const metrics = [...new Set([...Object.keys(a.scores), ...Object.keys(b?.scores ?? {})])];
      return (
        <div>
          <SyncedVideos items={[{ label: <b>{id}</b>, src: raw(a.video) }, ...(other ? [{ label: <b>{other}</b>, src: b ? raw(b.video) : null }] : [])]} />
          <KV items={[["perspective", a.perspective], ["actions", (a.actions ?? []).join(" → ")]]} />
          <div className="grid2">
            <Section title="Case scores">
              <Table rows={metrics.map((m) => ({ metric: m, mine: a.scores[m], theirs: b?.scores?.[m] }))} filter={false} maxHeight={600} cols={[
                { key: "metric", width: "minmax(170px, 2fr)" }, { key: "mine", label: id }, { key: "theirs", label: other || "compared" },
                { key: "delta", label: "Δ", value: (r: any) => typeof r.mine === "number" && typeof r.theirs === "number" ? r.mine - r.theirs : null,
                  render: (r: any) => { const v = typeof r.mine === "number" && typeof r.theirs === "number" ? r.mine - r.theirs : null; return <span className={deltaClass(v)}>{fmtDelta(v)}</span>; } }]} />
            </Section>
            <Section title="Judge answers" note="green = the expected answer, red = not">
              <JudgeTable mine={a.judge} theirs={b?.judge ?? []} other={other} />
            </Section>
          </div>
          <Section title={`Prompt schedule (${a.prompt_schedule?.length ?? 0})`}>
            <Table rows={a.prompt_schedule ?? []} filter maxHeight={420} cols={[{ key: "round", width: "60px" }, { key: "turn", width: "60px" }, { key: "prompt", width: "minmax(300px, 1fr)" }]} />
          </Section>
        </div>
      );
    }}</Q>
  );
}

export default function EvalTab({ id }: { id: string }) {
  const node = useApi(`/api/node/${encodeURIComponent(id)}`);
  const nodes = useApi("/api/overview");
  const parent = node.data?.node?.parent_id ?? "";
  const [other, setOther] = useParam("other", parent || (id !== "root" ? "root" : ""));
  const [caseId, setCase] = useParam("case", "");
  const [mode, setMode] = useState<"score" | "delta">("score");
  const [view, setView] = useState<"heatmap" | "videos">("heatmap");
  const mine = useApi(`/api/eval/${encodeURIComponent(id)}`);
  const theirs = useApi(other ? `/api/eval/${encodeURIComponent(other)}` : null);
  const ids: string[] = (nodes.data?.nodes ?? []).map((n: any) => n.node);
  return (
    <Q query={mine} what="eval">{(ev) => {
      if (!ev.cases.length) return <div>{ev.aggregates ? <Json value={ev.aggregates} /> : <Empty>No eval for {id} yet.</Empty>}</div>;
      const theirCases = Object.fromEntries((theirs.data?.cases ?? []).map((c: any) => [c.case, c]));
      const metrics = [...new Set(ev.cases.flatMap((c: any) => Object.keys(c.scores)))] as string[];
      const sel = caseId || ev.cases[0].case;
      const agg = ev.aggregates ?? {}, agg2 = theirs.data?.aggregates ?? {};
      const aggRows = [...Object.entries(agg.dimensions ?? {}).map(([k, v]) => ({ group: "dimension", name: k, mine: v, theirs: agg2.dimensions?.[k] })),
        ...Object.entries(agg.metrics ?? {}).map(([k, v]) => ({ group: "metric", name: k, mine: v, theirs: agg2.metrics?.[k] }))];
      return (
        <div>
          <div className="row gap">
            <span>compare with</span>
            <select value={other} onChange={(e) => setOther(e.target.value)}>
              <option value="">(none)</option>
              {ids.filter((n) => n !== id).map((n) => <option key={n} value={n}>{n}{n === parent ? " (parent)" : ""}</option>)}
            </select>
          </div>
          <Section title="Aggregates" note={other ? `${id} vs ${other}` : id}>
            <Table rows={aggRows} filter maxHeight={520} cols={[{ key: "group", width: "90px" }, { key: "name", width: "minmax(180px, 2fr)" }, { key: "mine", label: id },
              { key: "theirs", label: other || "compared" },
              { key: "delta", label: "Δ", value: (r: any) => typeof r.mine === "number" && typeof r.theirs === "number" ? r.mine - r.theirs : null,
                render: (r: any) => { const v = typeof r.mine === "number" && typeof r.theirs === "number" ? r.mine - r.theirs : null; return <span className={deltaClass(v)}>{fmtDelta(v)}</span>; } }]} />
            <Section title="aggregates.json" collapsed><Json value={ev.aggregates} open={2} /></Section>
          </Section>
          <Section title={`Cases (${ev.cases.length})`} actions={<><Chips items={["heatmap", "videos"] as const} value={view} onChange={setView as any} />
            {view === "heatmap" && other && <Chips items={["score", "delta"] as const} value={mode} onChange={setMode as any} render={(m) => m === "delta" ? `Δ vs ${other}` : "score"} />}</>}>
            {view === "heatmap" ? <Heatmap cases={ev.cases} metrics={metrics} theirs={theirCases} mode={other ? mode : "score"} onCase={setCase} selected={sel} />
              : <div className="gallery">{ev.cases.map((c: any) => (
                  <div key={c.case} className={`clip-card ${c.case === sel ? "selected" : ""}`} onClick={() => setCase(c.case)}>
                    <LazyVideo src={raw(c.video)} controls={false} />
                    <div className="small clip-meta"><b>case {c.case}</b> · {c.perspective}<div className="muted ellipsis">{(c.actions ?? []).join(" → ")}</div></div>
                  </div>))}</div>}
          </Section>
          <Section title={`Case ${sel}`} id="case">
            <Chips items={ev.cases.map((c: any) => c.case)} value={sel} onChange={setCase} />
            <CaseDetail id={id} other={other} caseId={sel} />
          </Section>
        </div>
      );
    }}</Q>
  );
}

function Heatmap({ cases, metrics, theirs, mode, onCase, selected }: { cases: any[]; metrics: string[]; theirs: Record<string, any>; mode: "score" | "delta"; onCase: (c: string) => void; selected: string }) {
  const ordered = [...JUDGED.filter((m) => metrics.includes(m)), ...metrics.filter((m) => !JUDGED.includes(m))];
  const option = useMemo(() => {
    const data: any[] = [];
    cases.forEach((c, y) => ordered.forEach((m, x) => {
      const v = c.scores[m], o = theirs[c.case]?.scores?.[m];
      const val = mode === "delta" ? (typeof v === "number" && typeof o === "number" ? v - o : null) : (typeof v === "number" ? v : null);
      data.push([x, y, val]);
    }));
    const vals = data.map((d) => d[2]).filter((v) => v !== null) as number[];
    const ext = mode === "delta" ? Math.max(0.01, ...vals.map(Math.abs)) : 1;
    return {
      tooltip: { formatter: (p: any) => `case ${cases[p.data[1]].case}<br/>${ordered[p.data[0]]}<br/>${mode === "delta" ? "Δ " : ""}${p.data[2] === null ? "—" : p.data[2].toFixed(4)}` },
      grid: { left: 70, right: 80, top: 10, bottom: 150 },
      xAxis: { type: "category", data: ordered, axisLabel: { rotate: 50, fontSize: 10 }, splitArea: { show: true } },
      yAxis: { type: "category", data: cases.map((c) => `case ${c.case}`), inverse: true, axisLabel: { fontSize: 10 } },
      visualMap: mode === "delta" ? { min: -ext, max: ext, calculable: true, orient: "vertical", right: 0, top: "middle", inRange: { color: ["#e5484d", "#2a2a2a", "#30a46c"] } }
        : { min: Math.min(0, ...vals), max: Math.max(1, ...vals), calculable: true, orient: "vertical", right: 0, top: "middle", inRange: { color: ["#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"] } },
      dataZoom: cases.length > 30 ? [{ type: "slider", yAxisIndex: 0, right: 60, width: 12, startValue: 0, endValue: 29 }, { type: "inside", yAxisIndex: 0 }] : [],
      series: [{ type: "heatmap", data, emphasis: { itemStyle: { borderColor: "#fff", borderWidth: 1 } }, label: { show: cases.length <= 20 && ordered.length <= 20, fontSize: 9, formatter: (p: any) => p.data[2] === null ? "" : p.data[2].toFixed(2) } }],
    };
  }, [cases, ordered.join(), theirs, mode]);
  return (
    <div>
      <div className="muted small">click a cell for that case · selected: case {selected}</div>
      <Chart option={option} height={Math.min(900, 170 + Math.min(cases.length, 30) * 20)} onClick={(p) => { onCase(cases[p.data[1]].case); document.getElementById("case")?.scrollIntoView({ behavior: "smooth" }); }} />
    </div>
  );
}
