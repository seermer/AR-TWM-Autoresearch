import * as echarts from "echarts";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { q, useApi } from "../api";
import { Chart, fmtDur, fmtTs, PALETTE, Q } from "../ui";
import { EventDrawer } from "./Event";

const ORDER = ["node", "phase", "sandbox", "llm", "kernel tools", "gpu jobs", "train", "concat_lora", "render", "wbench", "subprocess"];
const rank = (lane: string) => {
  const i = ORDER.findIndex((o) => lane === o || lane.startsWith(o + " ·"));
  return i === -1 ? ORDER.length : i;
};
const BAD = "#ff2d2d", WARN = "#f5a524", GREY = "#8b8d98";
const LANES = PALETTE.filter((c) => !["#e8684a", "#ff99c3", "#ff9d4d", "#f6bd16"].includes(c));

type Span = { lane: string; label: string; start: number; end: number; status: string; node?: string; phase?: string; attempt?: number; seq?: number; inferred?: boolean; [k: string]: any };

function pack(spans: Span[]) {
  const lanes = [...new Set(spans.map((s) => s.lane))].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  const rows: string[] = [];
  const placed: [Span, number][] = [];
  for (const lane of lanes) {
    const ends: number[] = [];
    for (const s of spans.filter((x) => x.lane === lane).sort((a, b) => a.start - b.start)) {
      let r = ends.findIndex((e) => e <= s.start);
      if (r === -1) { r = ends.length; ends.push(s.end); } else ends[r] = s.end;
      placed.push([s, rows.length + r]);
    }
    for (let r = 0; r < ends.length; r++) rows.push(r === 0 ? lane : `${lane} ·${r + 1}`);
  }
  return { rows, placed, lanes };
}

/** Every recorded span (start/end pair) and mark of the run, or of one node, on one time axis. */
export function Timeline({ node, height = 520 }: { node?: string; height?: number }) {
  const query = useApi(`/api/timeline?${q({ node })}`);
  return <Q query={query} what="timeline">{(d) => <TimelineChart data={d} node={node} height={height} />}</Q>;
}

function TimelineChart({ data, node, height }: { data: any; node?: string; height: number }) {
  const navigate = useNavigate();
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [seq, setSeq] = useState<number | null>(null);
  const now = Date.now() / 1000;
  const spans: Span[] = useMemo(() => {
    const out: Span[] = data.spans.map((s: any) => ({ ...s, end: s.status === "running" ? Math.max(now, s.end ?? now) : s.end ?? s.start }));
    if (!node) {
      const byNode = new Map<string, Span[]>();
      for (const s of out) if (s.node && s.node !== "run" && s.node !== "gpu") byNode.set(s.node, [...(byNode.get(s.node) ?? []), s]);
      for (const [n, ss] of byNode) out.push({ lane: "node", label: n, node: n, start: Math.min(...ss.map((s) => s.start)), end: Math.max(...ss.map((s) => s.end)),
        status: ss.some((s) => s.status === "running") ? "running" : "ok" });
    }
    return out;
  }, [data, node]);
  const allLanes = useMemo(() => [...new Set(spans.map((s) => s.lane))].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b)), [spans]);
  const visible = spans.filter((s) => !hidden.has(s.lane));
  const { rows, placed } = useMemo(() => pack(visible), [visible.length, hidden, spans]);
  const marks = data.marks;
  const option = useMemo(() => {
    const color = (s: Span) => s.status === "error" ? BAD : ["cancelled", "canceled"].includes(s.status) ? WARN : s.status === "no end recorded" ? GREY
      : LANES[allLanes.indexOf(s.lane) % LANES.length];
    const yRows = ["marks", ...rows];
    return {
      tooltip: {
        confine: true,
        formatter: (p: any) => {
          const s: Span | undefined = p.data?.span;
          const m = p.data?.mark;
          if (m) return `<b>${m.type}</b><br/>${m.label ?? ""}<br/>${fmtTs(m.ts)}<br/>${m.node ?? ""} · ${m.phase ?? ""} · ${m.attempt ?? ""}<br/><i>click: open the event</i>`;
          if (!s) return "";
          const extra = Object.entries(s).filter(([k, v]) => !["lane", "label", "start", "end", "status", "node", "phase", "attempt", "seq", "end_seq", "inferred"].includes(k) && v !== null && v !== undefined)
            .map(([k, v]) => `${k}: ${v}`).join("<br/>");
          return `<b>${s.label}</b> <span style="opacity:.7">(${s.lane})</span><br/>${s.node ?? ""} · ${s.phase ?? ""} · attempt ${s.attempt ?? ""}<br/>`
            + `${fmtTs(s.start)} → ${s.status === "running" ? "now" : fmtTs(s.end)}<br/>${fmtDur(s.end - s.start)} · <b>${s.status}</b>`
            + `${s.inferred ? "<br/><i>start/end paired by order (inferred)</i>" : ""}${extra ? "<br/>" + extra : ""}<br/><i>click: ${s.lane === "node" ? "open the node" : "open the start event"}</i>`;
        },
      },
      grid: { left: 150, right: rows.length > 30 ? 40 : 20, top: 10, bottom: 60 },
      xAxis: { type: "time", min: data.first * 1000, max: Math.max(data.last, ...visible.map((s) => s.end)) * 1000, axisLabel: { hideOverlap: true } },
      yAxis: { type: "category", data: yRows, inverse: true, axisLabel: { fontSize: 10, width: 140, overflow: "truncate" }, splitLine: { show: true, lineStyle: { opacity: 0.15 } } },
      dataZoom: [
        { type: "inside", xAxisIndex: 0, filterMode: "weakFilter" },
        { type: "slider", xAxisIndex: 0, height: 18, bottom: 8, filterMode: "weakFilter", labelFormatter: (v: number) => fmtTs(v / 1000).slice(5) },
        ...(yRows.length > 30 ? [{ type: "slider", yAxisIndex: 0, right: 6, width: 14, startValue: 0, endValue: 29, filterMode: "weakFilter" }] : []),
      ],
      series: [
        {
          type: "custom", encode: { x: [1, 2], y: 0 },
          renderItem: (params: any, api: any) => {
            const y = api.value(0), a = api.coord([api.value(1), y]), b = api.coord([api.value(2), y]);
            const h = api.size([0, 1])[1] * 0.7;
            const shape = echarts.graphic.clipRectByRect({ x: a[0], y: a[1] - h / 2, width: Math.max(b[0] - a[0], 1.5), height: h },
              { x: params.coordSys.x, y: params.coordSys.y, width: params.coordSys.width, height: params.coordSys.height });
            const s: Span = placed[params.dataIndex][0];
            if (!shape) return null;
            const rect = { type: "rect", shape, style: { fill: color(s), opacity: s.status === "running" ? 0.55 : 0.9, stroke: s.inferred ? "#fff" : undefined, lineDash: s.inferred ? [2, 2] : undefined } };
            if (s.lane !== "node" || shape.width < 40) return rect;
            return { type: "group", children: [rect, { type: "text", style: { text: s.label, x: shape.x + 4, y: shape.y + shape.height / 2, verticalAlign: "middle", fill: "#000", font: "10px sans-serif" } }] };
          },
          data: placed.map(([s, r]) => ({ value: [r + 1, s.start * 1000, s.end * 1000], span: s })),
        },
        {
          type: "scatter", symbolSize: 9,
          data: marks.map((m: any) => ({ value: [m.ts * 1000, 0], mark: m, itemStyle: { color: m.level === "error" || m.level === "critical" ? BAD : m.level === "warning" ? WARN : "#9aa0ff" } })),
          encode: { x: 0, y: 1 },
        },
      ],
    };
  }, [placed, rows, marks, data.first, data.last, allLanes]);
  return (
    <div>
      <div className="row gap wrap small">
        <span className="muted">lanes:</span>
        {allLanes.map((l, i) => (
          <label key={l} className="lane-toggle"><input type="checkbox" checked={!hidden.has(l)} onChange={() => {
            const h = new Set(hidden); h.has(l) ? h.delete(l) : h.add(l); setHidden(h);
          }} /><span className="swatch" style={{ background: LANES[i % LANES.length] }} />{l} <span className="muted">({spans.filter((s) => s.lane === l).length})</span></label>
        ))}
        <span className="muted">· red = error, orange = cancelled, grey = no end recorded, faded = running, dashed outline = pairing inferred</span>
      </div>
      <Chart option={option} height={Math.min(height, 90 + (rows.length + 1) * 16)} onClick={(p) => {
        const s: Span | undefined = p.data?.span;
        if (s?.lane === "node") navigate(`/node/${s.node}/timeline`);
        else if (s?.seq !== undefined) setSeq(s.seq);
        else if (p.data?.mark) setSeq(p.data.mark.seq);
      }} />
      <EventDrawer seq={seq} onClose={() => setSeq(null)} />
    </div>
  );
}
