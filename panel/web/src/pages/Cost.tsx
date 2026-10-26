import { useMemo, useState } from "react";
import { useApi } from "../api";
import { EventDrawer } from "../parts/Event";
import { Chart, fmtDur, fmtInt, fmtTs, fmtUsd, KV, PALETTE, Q, Section, Table } from "../ui";

function Charts({ calls }: { calls: any[] }) {
  const roles = useMemo(() => [...new Set(calls.map((c) => c.role))].sort(), [calls]);
  const cumulative = useMemo(() => {
    const acc: Record<string, number> = {}, accUsd: Record<string, number> = {};
    const tok: Record<string, any[]> = {}, usd: Record<string, any[]> = {};
    for (const c of calls) {
      acc[c.role] = (acc[c.role] ?? 0) + (c.total_tokens ?? 0);
      accUsd[c.role] = (accUsd[c.role] ?? 0) + (c.usd ?? 0);
      (tok[c.role] ??= []).push([c.ts * 1000, acc[c.role]]);
      (usd[c.role] ??= []).push([c.ts * 1000, accUsd[c.role]]);
    }
    return { tok, usd };
  }, [calls]);
  const line = (series: Record<string, any[]>, name: string) => ({
    tooltip: { trigger: "axis" }, legend: { type: "scroll", top: 0 }, grid: { left: 70, right: 20, top: 50, bottom: 50 },
    xAxis: { type: "time" }, yAxis: { type: "value", name },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 14, bottom: 4 }],
    series: roles.map((r, i) => ({ name: r, type: "line", step: "end", showSymbol: false, data: series[r] ?? [], color: PALETTE[i % PALETTE.length] })),
  });
  const scatter = (key: string, name: string) => ({
    tooltip: { formatter: (p: any) => `${p.seriesName}<br/>${name}: ${fmtInt(p.data[1])}<br/>${p.data[2]} · ${p.data[3]}` },
    legend: { type: "scroll", top: 0 }, grid: { left: 70, right: 20, top: 50, bottom: 50 },
    xAxis: { type: "time" }, yAxis: { type: "value", name },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 14, bottom: 4 }],
    series: roles.map((r, i) => ({ name: r, type: "scatter", symbolSize: 4, color: PALETTE[i % PALETTE.length],
      data: calls.filter((c) => c.role === r).map((c) => [c.ts * 1000, c[key], c.node, `${c.phase}-${c.attempt}`, c.seq]) })),
  });
  return (
    <div className="grid2">
      <Section title="Tokens over time, cumulative by role"><Chart option={line(cumulative.tok, "tokens")} height={320} /></Section>
      <Section title="Spend over time, cumulative by role"><Chart option={line(cumulative.usd, "USD")} height={320} /></Section>
      <Section title="Prompt tokens per call (context size)"><Chart option={scatter("prompt_tokens", "prompt tokens")} height={320} /></Section>
      <Section title="Latency per call"><Chart option={scatter("latency_s", "seconds")} height={320} /></Section>
    </div>
  );
}

export default function Cost() {
  const query = useApi("/api/cost");
  const [seq, setSeq] = useState<number | null>(null);
  return (
    <div className="page">
      <Q query={query} what="cost">{(c) => {
        const usd = c.calls.reduce((s: number, x: any) => s + (x.usd ?? 0), 0);
        const has = c.calls.some((x: any) => x.usd !== null && x.usd !== undefined);
        return (
          <>
            <KV items={[["calls", fmtInt(c.calls.length)], ["tokens", fmtInt(c.calls.reduce((s: number, x: any) => s + (x.total_tokens ?? 0), 0))],
              ["prompt / completion", `${fmtInt(c.calls.reduce((s: number, x: any) => s + (x.prompt_tokens ?? 0), 0))} / ${fmtInt(c.calls.reduce((s: number, x: any) => s + (x.completion_tokens ?? 0), 0))}`],
              ["spend", has ? fmtUsd(usd) : "n/a (the provider reports no cost)"], ["model time", fmtDur(c.calls.reduce((s: number, x: any) => s + (x.latency_s ?? 0), 0))],
              ["non-200 responses", c.errors.length]]} />
            <Charts calls={c.calls} />
            <div className="grid2">
              <Section title="By node and phase"><Table rows={c.by_phase} cols={["node", "phase", "calls", "tokens", "usd", "latency_s", "errors"].map((key) => ({ key }))} /></Section>
              <Section title="By role" note="role = the agent's own label; ask calls excluded"><Table rows={c.by_role} cols={["role", "conversations", "calls", "tokens", "usd"].map((key) => ({ key }))} /></Section>
            </div>
            <Section title={`Non-200 responses (${c.errors.length})`} collapsed={!c.errors.length}><Table rows={c.errors} cols={["time", "node", "phase", "status"].map((key) => ({ key }))} /></Section>
            <Section title={`Every model call (${c.calls.length})`} note="click one for its event">
              <Table rows={c.calls} rowKey={(r: any) => String(r.seq)} onRowClick={(r: any) => setSeq(r.seq)} sort={["ts", "desc"]} maxHeight={600} cols={[
                { key: "ts", label: "time", width: "150px", render: (r: any) => fmtTs(r.ts) }, { key: "node", width: "90px" }, { key: "phase", width: "110px" },
                { key: "attempt", width: "60px" }, { key: "role", width: "minmax(110px, 1fr)" }, { key: "turn_index", label: "turn", width: "60px" }, { key: "model", width: "minmax(100px, 1fr)" },
                { key: "prompt_tokens", label: "in", width: "90px" }, { key: "completion_tokens", label: "out", width: "80px" }, { key: "usd", width: "80px" },
                { key: "latency_s", label: "s", width: "70px" }, { key: "status", width: "60px" }]} />
            </Section>
            <EventDrawer seq={seq} onClose={() => setSeq(null)} />
          </>
        );
      }}</Q>
    </div>
  );
}
