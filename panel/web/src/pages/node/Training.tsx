import { useMemo, useState } from "react";
import { useApi } from "../../api";
import { Code } from "../../parts/lazy";
import { Badge, Chart, Empty, KV, PALETTE, Q, Section, Table } from "../../ui";

export function ema(ys: number[], k: number) {
  if (!k) return ys;
  let s: number | null = null;
  return ys.map((y) => (s = s === null || !isFinite(s) ? y : k * s + (1 - k) * y));
}

/** Loss curves (and lr on a second axis) with smoothing and a log scale. */
export function LossChart({ curves, showLr = true, height = 380 }: { curves: { name: string; loss: any[] }[]; showLr?: boolean; height?: number }) {
  const [smooth, setSmooth] = useState(0);
  const [log, setLog] = useState(false);
  const option = useMemo(() => ({
    tooltip: { trigger: "axis" }, legend: { top: 0, type: "scroll" },
    grid: { left: 60, right: showLr ? 70 : 20, top: 30, bottom: 50 },
    xAxis: { type: "value", name: "step", nameLocation: "middle", nameGap: 26 },
    yAxis: [{ type: log ? "log" : "value", name: "loss", scale: true }, ...(showLr ? [{ type: "value", name: "lr", axisLabel: { formatter: (v: number) => v.toExponential(1) } }] : [])],
    dataZoom: [{ type: "inside" }, { type: "slider", height: 14, bottom: 4 }],
    series: curves.flatMap((c, i) => {
      const ys = ema(c.loss.map((p) => p.loss), smooth);
      return [
        ...(smooth ? [{ name: `${c.name} raw`, type: "line", showSymbol: false, data: c.loss.map((p) => [p.step, p.loss]), lineStyle: { width: 1, opacity: 0.25 }, color: PALETTE[i % PALETTE.length] }] : []),
        { name: c.name, type: "line", showSymbol: false, data: c.loss.map((p, k) => [p.step, ys[k]]), lineStyle: { width: 1.5 }, color: PALETTE[i % PALETTE.length] },
        ...(showLr ? [{ name: `${c.name} lr`, type: "line", yAxisIndex: 1, showSymbol: false, data: c.loss.map((p) => [p.step, p.lr]), lineStyle: { width: 1, type: "dashed" }, color: PALETTE[(i + 3) % PALETTE.length] }] : []),
      ];
    }),
  }), [curves, smooth, log, showLr]);
  return (
    <div>
      <div className="row gap small">
        <span>smoothing (EMA)</span><input type="range" min={0} max={0.99} step={0.01} value={smooth} onChange={(e) => setSmooth(+e.target.value)} /><span>{smooth.toFixed(2)}</span>
        <label><input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} /> log scale</label>
      </div>
      <Chart option={option} height={height} />
    </div>
  );
}

export default function TrainingTab({ id }: { id: string }) {
  const query = useApi(`/api/training/${encodeURIComponent(id)}`);
  return (
    <Q query={query} what="training">{(attempts: any[]) => attempts.length === 0 ? <Empty>No improve_recipe attempt (baseline, or not started).</Empty> : (
      <div>
        {attempts.filter((a) => a.loss.length).length > 1 && (
          <Section title="All attempts' loss"><LossChart curves={attempts.filter((a) => a.loss.length).map((a) => ({ name: `attempt ${a.attempt}`, loss: a.loss }))} showLr={false} /></Section>
        )}
        {[...attempts].reverse().map((t) => (
          <Section key={t.attempt} title={<>improve_recipe attempt {t.attempt} {t.outcome && <Badge>{t.outcome.outcome}</Badge>}</>}>
            <KV items={[["reached training", t.reached_training ? "yes" : "no: this attempt never reached training"], ["started", t.started], ["duration", t.duration_min && `${t.duration_min} min`],
              ["outcome", t.outcome?.outcome], ["detail", t.outcome?.detail], ["loss points", t.loss.length || null]]} />
            {t.loss.length > 0 && <LossChart curves={[{ name: `attempt ${t.attempt}`, loss: t.loss }]} />}
            <div className="grid2">
              <Section title={`Gate checks (${t.gates.length})`}>
                <Table rows={t.gates} filter={false} cols={[{ key: "time", width: "150px" }, { key: "result", width: "80px", render: (r: any) => <Badge>{r.result}</Badge> }, { key: "failures", width: "minmax(200px, 3fr)" }]} />
              </Section>
              <Section title="GPU during training">
                <Table rows={t.gpu} filter={false} cols={[{ key: "gpu" }, { key: "mean_util", label: "mean util %" }, { key: "peak_memory_gib", label: "peak memory GiB" }]} />
              </Section>
            </div>
            <Section title="Resolved train config" collapsed={!t.config}>{t.config ? <Code text={t.config} path="train_config.yaml" height={460} /> : <div className="muted">none</div>}</Section>
          </Section>
        ))}
      </div>
    )}</Q>
  );
}
