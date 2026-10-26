import { Link } from "react-router-dom";
import { useApi } from "../api";
import { Chart, Empty, PALETTE, Q, Section, Table, useParam } from "../ui";

export default function Selection() {
  const query = useApi("/api/selection");
  const [pick, setPick] = useParam("pick");
  return (
    <div className="page">
      <Q query={query} what="selection">{(rows: any[]) => {
        if (!rows.length) return <Empty>No parent picks yet.</Empty>;
        const sel = rows[pick ? +pick : rows.length - 1];
        const keys = [...new Set(sel.candidates.flatMap((c: any) => Object.keys(c)))] as string[];
        return (
          <>
            <Section title={`Parent picks (${rows.length})`} note="click one for its candidates">
              <Table rows={rows.map((r, i) => ({ ...r, i, n: r.candidates.length }))} rowKey={(r: any) => String(r.i)} selected={String(rows.indexOf(sel))} onRowClick={(r: any) => setPick(String(r.i))} cols={[
                { key: "time", width: "160px" }, { key: "child", render: (r: any) => <Link to={`/node/${r.child}`}>{r.child}</Link> },
                { key: "chosen", render: (r: any) => <Link to={`/node/${r.chosen}`}>{r.chosen}</Link> }, { key: "seed" }, { key: "n", label: "candidates" }]} />
            </Section>
            <Section title={`Candidates when ${sel.child} was created (chosen: ${sel.chosen})`}>
              <Chart height={Math.max(180, sel.candidates.length * 30 + 60)} option={{
                tooltip: { trigger: "axis" }, legend: { top: 0 }, grid: { left: 100, right: 30, top: 30, bottom: 20 },
                yAxis: { type: "category", data: sel.candidates.map((c: any) => c.node_id), inverse: true }, xAxis: { type: "value" },
                series: ["P", "value", "score", "penalty"].filter((k) => keys.includes(k)).map((k, i) => ({ name: k, type: "bar", data: sel.candidates.map((c: any) => ({ value: c[k], itemStyle: c.node_id === sel.chosen && k === "P" ? { color: "#30a46c" } : undefined })), color: PALETTE[i] })),
              }} />
              <Table rows={sel.candidates} rowKey={(r: any) => r.node_id} selected={sel.chosen} filter={false} cols={keys.map((key) => ({ key }))} />
            </Section>
          </>
        );
      }}</Q>
    </div>
  );
}
