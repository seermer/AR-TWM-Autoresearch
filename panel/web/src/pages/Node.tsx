import { lazy, Suspense, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { q, useApi } from "../api";
import { Diff } from "../parts/lazy";
import { EventDrawer } from "../parts/Event";
import { Timeline } from "../parts/Timeline";
import { Badge, Chart, deltaClass, fmtDelta, fmtTs, Json, Markdown, PALETTE, Q, Section, Table, Tabs, Text } from "../ui";
import { eventCols } from "./Overview";

const Agents = lazy(() => import("./node/Agents"));
const CodeTab = lazy(() => import("./node/Code"));
const DataTab = lazy(() => import("./node/Data"));
const TrainingTab = lazy(() => import("./node/Training"));
const EvalTab = lazy(() => import("./node/Eval"));

const TABS = ["summary", "timeline", "agents", "code", "data", "training", "eval", "events"];

function Summary({ d, id }: { d: any; id: string }) {
  const n = d.node;
  const metricsChart = useMemo(() => ({
    tooltip: { trigger: "axis" }, legend: { top: 0 },
    grid: { left: 180, right: 20, top: 30, bottom: 20 },
    yAxis: { type: "category", data: d.metrics.map((m: any) => m.metric), inverse: true, axisLabel: { fontSize: 10 } },
    xAxis: { type: "value", scale: true },
    series: ["node", "parent", "root"].map((k, i) => ({ name: k === "node" ? id : k, type: "bar", data: d.metrics.map((m: any) => m[k]), itemStyle: { color: PALETTE[i] } })),
  }), [d]);
  const timings = Object.entries(d.timings ?? {}).filter(([k, v]) => k !== "total_s" && typeof v === "number") as [string, number][];
  return (
    <div>
      {d.baseline && <div className="note">Baseline: no parent, edit, recipe, rationale or training. Only its eval, lineage and events apply.</div>}
      {n.error && <div className="err">Error: {n.error}</div>}
      <div className="grid2">
        <Section title="summary.md">{d.summary ? <Markdown text={d.summary} /> : <div className="muted">none</div>}</Section>
        <Section title="rationale.md">{d.rationale ? <Markdown text={d.rationale} /> : <div className="muted">none</div>}</Section>
      </div>
      <Section title={`Metrics: ${id}, parent and root (${d.metrics.length})`}>
        <div className="grid2">
          <Table rows={d.metrics} filter={false} maxHeight={700} cols={[
            { key: "metric", width: "minmax(160px, 2fr)" }, { key: "node", label: id }, { key: "parent" }, { key: "root" },
            { key: "vs parent", value: (r: any) => r.node !== null && r.parent !== null && r.parent !== undefined ? r.node - r.parent : null,
              render: (r: any) => { const v = r.node !== null && r.parent !== null && r.parent !== undefined ? r.node - r.parent : null; return <span className={deltaClass(v)}>{fmtDelta(v)}</span>; } },
            { key: "vs root", value: (r: any) => r.node !== null && r.root !== null && r.root !== undefined ? r.node - r.root : null,
              render: (r: any) => { const v = r.node !== null && r.root !== null && r.root !== undefined ? r.node - r.root : null; return <span className={deltaClass(v)}>{fmtDelta(v)}</span>; } },
          ]} />
          {d.metrics.length > 0 && <Chart option={metricsChart} height={Math.max(260, d.metrics.length * 26 + 60)} />}
        </div>
      </Section>
      <Section title="edit.json">{d.edit ? <Json value={d.edit} open={3} /> : <div className="muted">none</div>}</Section>
      <Section title="Recipe vs parent" collapsed={!d.recipe}>
        {d.recipe ? <><Diff original={d.parent_recipe ?? ""} modified={d.recipe} path={`${id}/recipe.yaml`} height={420} />
          <Section title="unified diff" collapsed><Text text={d.recipe_diff} lines={200} /></Section></> : <div className="muted">no recipe.yaml</div>}
      </Section>
      <Section title={`Attempts (${d.attempts.length})`}>
        <Table rows={d.attempts} cols={[{ key: "phase", width: "130px" }, { key: "attempt", width: "70px" }, { key: "outcome", width: "110px", render: (r: any) => <Badge>{r.outcome}</Badge> },
          { key: "time", width: "150px" }, { key: "detail", width: "minmax(240px, 4fr)" }]} />
      </Section>
      <div className="grid2">
        <Section title="Phase timings">
          {timings.length > 0 && <Chart height={Math.max(160, timings.length * 24 + 40)} option={{
            tooltip: {}, grid: { left: 140, right: 30, top: 10, bottom: 20 },
            yAxis: { type: "category", data: timings.map(([k]) => k), inverse: true }, xAxis: { type: "value", name: "s" },
            series: [{ type: "bar", data: timings.map(([, v]) => v), itemStyle: { color: PALETTE[4] }, label: { show: true, position: "right", formatter: (p: any) => `${(p.value / 60).toFixed(1)} min` } }],
          }} />}
          <Json value={d.timings} open={1} />
        </Section>
        <Section title="Data commit">{d.data_commit ? <Json value={d.data_commit} open={2} /> : <div className="muted">none</div>}</Section>
      </div>
      <Section title="Archive row (nodes table)" collapsed><Json value={d.node} open={1} /></Section>
    </div>
  );
}

function Events({ id }: { id: string }) {
  const query = useApi(`/api/events?${q({ node: id, limit: 5000 })}`);
  const [seq, setSeq] = useState<number | null>(null);
  return <Q query={query}>{(d) => <>
    <EventDrawer seq={seq} onClose={() => setSeq(null)} />
    <div className="row gap"><span className="muted small">{d.total} events of {id}{d.total > 5000 ? " (first 5000 shown)" : ""}</span>
      <Link className="btn small" to={`/trace?${q({ node: id })}`}>open in Trace with filters</Link></div>
    <Table rows={d.rows} cols={eventCols} rowKey={(r: any) => String(r.seq)} maxHeight={700} onRowClick={(r: any) => setSeq(r.seq)} />
  </>}</Q>;
}

export default function NodePage() {
  const { id = "", tab = "summary" } = useParams();
  const query = useApi(`/api/node/${encodeURIComponent(id)}`);
  return (
    <div className="page">
      <Q query={query} what={`node ${id}`}>{(d) => {
        const n = d.node;
        const lineage: string[] = d.lineage.split(" → ");
        return (
          <>
            <div className="node-head">
              <h2>{id}</h2>
              <Badge>{n.status}</Badge>
              <span className="mono big">{n.score !== null ? n.score.toFixed(4) : "no score"}</span>
              <span className="muted">created {fmtTs(n.created_at)} · depth {n.depth}</span>
              <span className="lineage">lineage: {lineage.map((x, i) => <span key={x}>{i > 0 && " → "}{x === id ? <b>{x}</b> : <Link to={`/node/${x}/${tab}`}>{x}</Link>}</span>)}</span>
              {d.children.length > 0 && <span>children: {d.children.map((c: string) => <Link key={c} to={`/node/${c}/${tab}`} className="chip">{c}</Link>)}</span>}
            </div>
            <Tabs value={tab} items={TABS.map((t) => ({ key: t, label: t, to: `/node/${id}/${t}` }))} />
            <Suspense key={id} fallback={<div className="muted pad">Loading…</div>}>
              {tab === "summary" && <Summary d={d} id={id} />}
              {tab === "timeline" && <Section title={`Timeline of ${id}`} note="every recorded span of this node"><Timeline node={id} height={800} /></Section>}
              {tab === "agents" && <Agents id={id} attempts={d.agent_attempts} />}
              {tab === "code" && <CodeTab id={id} />}
              {tab === "data" && <DataTab id={id} />}
              {tab === "training" && <TrainingTab id={id} />}
              {tab === "eval" && <EvalTab id={id} />}
              {tab === "events" && <Events id={id} />}
            </Suspense>
          </>
        );
      }}</Q>
    </div>
  );
}
