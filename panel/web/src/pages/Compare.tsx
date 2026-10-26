import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useApi } from "../api";
import { Chart, Chips, Empty, PALETTE, Q, Section, Table } from "../ui";
import { LossChart } from "./node/Training";

export default function Compare() {
  const matrix = useApi("/api/eval-matrix");
  const training = useApi("/api/training-compare");
  const navigate = useNavigate();
  const [mode, setMode] = useState<"score" | "delta vs parent" | "delta vs root">("score");
  return (
    <div className="page">
      <Q query={matrix} what="metrics">{(m) => {
        if (!m.nodes.length) return <Empty>No scored node yet.</Empty>;
        const by = Object.fromEntries(m.nodes.map((n: any) => [n.node, n]));
        const cols = ["score", ...m.metrics];
        const value = (n: any, k: string) => (k === "score" ? n.score : n.metrics[k]) ?? null;
        const ref = (n: any) => mode === "delta vs parent" ? by[n.parent] : mode === "delta vs root" ? by["root"] : null;
        const cell = (n: any, k: string) => { const v = value(n, k); if (mode === "score") return v; const r = ref(n); const o = r ? value(r, k) : null; return v !== null && o !== null ? v - o : null; };
        return (
          <>
            <Section title="Nodes × metrics" actions={<Chips items={["score", "delta vs parent", "delta vs root"] as const} value={mode} onChange={setMode as any} />} note="click a node to open its eval">
              <HeatmapAll nodes={m.nodes} cols={cols} cell={cell} delta={mode !== "score"} onNode={(n) => navigate(`/node/${n}/eval`)} />
              <Table rows={m.nodes} rowKey={(r: any) => r.node} maxHeight={500} cols={[{ key: "node", width: "100px" }, { key: "parent", width: "90px" },
                ...cols.map((k) => ({ key: k, width: "110px", value: (r: any) => cell(r, k) }))]} />
            </Section>
            <Section title="Metrics across nodes" note="one line per metric, nodes in creation order · click the legend to hide metrics">
              <Chart height={420} option={{
                tooltip: { trigger: "axis" }, legend: { type: "scroll", top: 0 }, grid: { left: 60, right: 20, top: 40, bottom: 40 },
                xAxis: { type: "category", data: m.nodes.map((n: any) => n.node) }, yAxis: { type: "value", scale: true },
                series: cols.map((k, i) => ({ name: k, type: "line", data: m.nodes.map((n: any) => value(n, k)), color: PALETTE[i % PALETTE.length], lineStyle: { width: k === "score" ? 3 : 1 } })),
              }} />
            </Section>
          </>
        );
      }}</Q>
      <Section title="Training loss of every node">
        <Q query={training} what="loss curves">{(curves: any[]) => curves.length ? <LossChart curves={curves.map((c) => ({ name: `${c.node} #${c.attempt}${c.score !== null ? ` (${c.score.toFixed(3)})` : ""}`, loss: c.loss }))} showLr={false} height={460} /> : <Empty>No training logged.</Empty>}</Q>
      </Section>
    </div>
  );
}

function HeatmapAll({ nodes, cols, cell, delta, onNode }: { nodes: any[]; cols: string[]; cell: (n: any, k: string) => number | null; delta: boolean; onNode: (n: string) => void }) {
  const option = useMemo(() => {
    const data = nodes.flatMap((n, y) => cols.map((k, x) => [x, y, cell(n, k)]));
    const vals = data.map((d) => d[2]).filter((v) => v !== null) as number[];
    const ext = Math.max(0.01, ...vals.map(Math.abs));
    return {
      tooltip: { formatter: (p: any) => `${nodes[p.data[1]].node}<br/>${cols[p.data[0]]}: ${p.data[2] === null ? "—" : p.data[2].toFixed(4)}` },
      grid: { left: 90, right: 80, top: 10, bottom: 150 },
      xAxis: { type: "category", data: cols, axisLabel: { rotate: 50, fontSize: 10 } },
      yAxis: { type: "category", data: nodes.map((n) => n.node), inverse: true },
      visualMap: delta ? { min: -ext, max: ext, calculable: true, right: 0, top: "middle", inRange: { color: ["#e5484d", "#2a2a2a", "#30a46c"] } }
        : { min: Math.min(...vals, 0), max: Math.max(...vals, 1), calculable: true, right: 0, top: "middle", inRange: { color: ["#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"] } },
      series: [{ type: "heatmap", data, label: { show: nodes.length <= 25, fontSize: 9, formatter: (p: any) => p.data[2] === null ? "" : p.data[2].toFixed(3) } }],
    };
  }, [nodes, cols, delta, cell]);
  return <Chart option={option} height={170 + nodes.length * 26} onClick={(p) => onNode(nodes[p.data[1]].node)} />;
}
