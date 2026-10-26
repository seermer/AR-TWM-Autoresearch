import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useApi } from "../api";
import { Dag } from "../parts/Dag";
import { EventDrawer } from "../parts/Event";
import { Timeline } from "../parts/Timeline";
import { Badge, Chart, deltaClass, fmtDelta, fmtTs, fmtVal, PALETTE, Q, Section, Table } from "../ui";

export function NodeTree({ nodes, height = 360, selected }: { nodes: any[]; height?: number; selected?: string }) {
  const navigate = useNavigate();
  const scored = nodes.map((n) => n.score).filter((s) => s !== null && s !== undefined);
  const best = scored.length ? Math.max(...scored) : null;
  const dag = useMemo(() => ({
    nodes: nodes.map((n) => ({
      id: n.node, width: 180, height: 78,
      className: `tree-node st-${n.status} ${n.score !== null && n.score === best ? "best" : ""}`,
      content: (
        <div>
          <div className="row gap"><b>{n.node}</b><span className="grow" /><Badge>{n.status}</Badge></div>
          <div className="mono">{n.score !== null && n.score !== undefined ? n.score.toFixed(4) : "no score"}
            {n["vs parent"] !== null && n["vs parent"] !== undefined && <span className={deltaClass(n["vs parent"])}> {fmtDelta(n["vs parent"])}</span>}</div>
          <div className="muted small">{n.duration_min ? `${n.duration_min} min` : ""} {n.score === best && best !== null ? "★ best" : ""}{n.error ? " · error" : ""}</div>
        </div>
      ),
    })),
    edges: nodes.filter((n) => n.parent).map((n) => ({ id: `${n.parent}->${n.node}`, source: n.parent, target: n.node,
      color: n["vs parent"] > 0 ? "#30a46c" : n["vs parent"] < 0 ? "#e5484d" : undefined })),
  }), [nodes, best]);
  return <Dag nodes={dag.nodes} edges={dag.edges} height={height} onNode={(id) => navigate(`/node/${id}`)} selected={selected} />;
}

function ScoreChart({ nodes }: { nodes: any[] }) {
  const navigate = useNavigate();
  const option = useMemo(() => {
    const idx = Object.fromEntries(nodes.map((n, i) => [n.node, i]));
    const pts = nodes.map((n, i) => [i, n.score]);
    let best = -Infinity;
    const bestSoFar = nodes.map((n, i) => { if (n.score !== null && n.score !== undefined) best = Math.max(best, n.score); return [i, isFinite(best) ? best : null]; });
    const links = nodes.filter((n) => n.parent && n.score !== null && nodes[idx[n.parent]]?.score !== null)
      .map((n) => ({ coords: [[idx[n.parent], nodes[idx[n.parent]].score], [idx[n.node], n.score]] }));
    return {
      tooltip: { trigger: "item", formatter: (p: any) => p.seriesName === "node" ? `<b>${nodes[p.data[0]].node}</b><br/>score ${fmtVal(p.data[1])}<br/>parent ${nodes[p.data[0]].parent || "—"}<br/>vs parent ${fmtDelta(nodes[p.data[0]]["vs parent"])}` : p.seriesName },
      legend: { top: 0 },
      grid: { left: 60, right: 20, top: 30, bottom: 40 },
      xAxis: { type: "category", data: nodes.map((n) => n.node), name: "nodes in creation order", nameLocation: "middle", nameGap: 26 },
      yAxis: { type: "value", scale: true, name: "score" },
      series: [
        { name: "parent → child", type: "lines", coordinateSystem: "cartesian2d", data: links, lineStyle: { color: "#888", width: 1, curveness: 0.2, opacity: 0.6 }, symbol: ["none", "arrow"], symbolSize: 6 },
        { name: "best so far", type: "line", step: "end", data: bestSoFar, symbol: "none", lineStyle: { type: "dashed", color: PALETTE[2] } },
        { name: "node", type: "scatter", data: pts, symbolSize: 11, itemStyle: { color: PALETTE[0] } },
      ],
    };
  }, [nodes]);
  return <Chart option={option} height={300} onClick={(p) => p.seriesName === "node" && navigate(`/node/${nodes[p.data[0]].node}`)} />;
}

export function GpuChart({ height = 420 }: { height?: number }) {
  const query = useApi("/api/gpu");
  return (
    <Q query={query} what="GPU samples">{(g) => {
      const gpus = Object.keys(g.gpus);
      if (!g.samples) return <div className="muted">No GPU samples recorded.</div>;
      const ms = (pts: [number, number][]) => pts.map(([t, v]) => [t * 1000, v]);
      const option = {
        tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
        legend: { top: 0, data: gpus.map((x) => `GPU ${x}`) },
        axisPointer: { link: [{ xAxisIndex: "all" }] },
        grid: [{ left: 60, right: 20, top: 30, height: "33%" }, { left: 60, right: 20, top: "56%", height: "33%" }],
        xAxis: [{ type: "time", gridIndex: 0, axisLabel: { show: false } }, { type: "time", gridIndex: 1 }],
        yAxis: [{ type: "value", gridIndex: 0, name: "util %", max: 100 }, { type: "value", gridIndex: 1, name: "memory GiB" }],
        dataZoom: [{ type: "inside", xAxisIndex: [0, 1] }, { type: "slider", xAxisIndex: [0, 1], height: 16, bottom: 4 }],
        series: gpus.flatMap((x, i) => [
          { name: `GPU ${x}`, type: "line", xAxisIndex: 0, yAxisIndex: 0, showSymbol: false, lineStyle: { width: 1 }, color: PALETTE[i % PALETTE.length],
            data: ms(g.gpus[x].util) },
          { name: `GPU ${x}`, type: "line", xAxisIndex: 1, yAxisIndex: 1, showSymbol: false, lineStyle: { width: 1 }, color: PALETTE[i % PALETTE.length],
            data: ms(g.gpus[x].memory_gib) },
        ]),
      };
      const note = g.buckets < g.samples ? `downsampled to ${g.buckets.toLocaleString()} time buckets, each showing its lowest and highest sample per GPU` : "every sample";
      return <><div className="muted small">{g.samples.toLocaleString()} samples, {fmtTs(g.first)} → {fmtTs(g.last)} ({note}; zoom with the wheel or the slider)</div><Chart option={option} height={height} /></>;
    }}</Q>
  );
}

export const nodeCols = [
  { key: "node", width: "110px", render: (r: any) => <Link to={`/node/${r.node}`}>{r.node}</Link> },
  { key: "parent", width: "100px", render: (r: any) => r.parent ? <Link to={`/node/${r.parent}`}>{r.parent}</Link> : "" },
  { key: "status", width: "90px", render: (r: any) => <Badge>{r.status}</Badge> },
  { key: "score", width: "80px" },
  { key: "vs parent", width: "90px", render: (r: any) => <span className={deltaClass(r["vs parent"])}>{fmtDelta(r["vs parent"])}</span> },
  { key: "depth", width: "60px" },
  { key: "attempts", width: "minmax(140px, 1fr)", mono: true },
  { key: "duration_min", label: "minutes", width: "80px" },
  { key: "created", width: "150px" },
  { key: "subtree_value", width: "100px" },
  { key: "lora_rank", label: "LoRA rank", width: "80px" },
  { key: "lora_alpha", label: "LoRA alpha", width: "80px" },
  { key: "error", width: "minmax(160px, 2fr)" },
];

export const eventCols = [
  { key: "seq", width: "70px" }, { key: "time", width: "150px" },
  { key: "node", width: "90px" }, { key: "phase", width: "120px" }, { key: "attempt", width: "60px" },
  { key: "type", width: "170px", render: (r: any) => <span className={/error|fail|reject/.test(r.type) ? "neg" : ""}>{r.type}</span> },
  { key: "component", width: "90px" }, { key: "summary", width: "minmax(200px, 3fr)", mono: true },
];

export default function Overview() {
  const query = useApi("/api/overview");
  const [seq, setSeq] = useState<number | null>(null);
  return (
    <Q query={query} what="overview">{(ov) => (
      <div className="page">
        <div className="grid2">
          <Section title="Node tree" note="click a node to open it · edge colour: better / worse than its parent">
            <NodeTree nodes={ov.nodes} />
          </Section>
          <Section title="Score by node" note="lines join each node to its parent">
            <ScoreChart nodes={ov.nodes} />
          </Section>
        </div>
        <Section title={`Nodes (${ov.nodes.length})`}>
          <Table rows={ov.nodes} cols={nodeCols} rowKey={(r: any) => r.node} />
        </Section>
        <Section title="Run timeline" note="every recorded span; click a bar for its event, a node bar for the node">
          <Timeline height={560} />
        </Section>
        <Section title="GPUs"><GpuChart /></Section>
        <div className="grid2">
          <Section title={`Alerts (${ov.alerts.length}, newest first)`}>
            <Table rows={ov.alerts} cols={[{ key: "time", width: "150px" }, { key: "level", width: "80px", render: (r: any) => <Badge>{r.level}</Badge> },
              { key: "kind", width: "110px" }, { key: "message", width: "minmax(200px, 3fr)" }]} />
          </Section>
          <Section title="Problems by kind" note="the run's last hour / total · click for the list">
            <Table rows={ov.problem_counts} cols={[{ key: "kind", render: (r: any) => <Link to={`/problems?${new URLSearchParams({ kinds: r.kind })}`}>{r.kind}</Link> },
              { key: "source", render: (r: any) => <Badge>{r.source}</Badge> }, { key: "last hour" }, { key: "total" }]} filter={false} />
          </Section>
        </div>
        <Section title="Latest 50 events" note="updates live · click one for its record and payload" actions={<Link to="/trace" className="btn small">all events</Link>}>
          <Table rows={ov.recent} cols={eventCols} rowKey={(r: any) => String(r.seq)} onRowClick={(r: any) => setSeq(r.seq)} />
        </Section>
        <EventDrawer seq={seq} onClose={() => setSeq(null)} />
      </div>
    )}</Q>
  );
}
