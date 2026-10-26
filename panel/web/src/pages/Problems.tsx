import { Link } from "react-router-dom";
import { q, useApi } from "../api";
import { EventDetail } from "../parts/Event";
import { Badge, Chips, Empty, Q, Section, Table, Text, useParam } from "../ui";

export default function Problems() {
  const query = useApi("/api/problems");
  const [kinds, setKinds] = useParam("kinds");
  const [source, setSource] = useParam("source", "any");
  const [id, setId] = useParam("id");
  const chosen = kinds ? kinds.split(",") : [];
  return (
    <div className="page">
      <Q query={query} what="problems">{(all: any[]) => {
        const counts: Record<string, number> = {};
        for (const p of all) counts[p.kind] = (counts[p.kind] ?? 0) + 1;
        const rows = all.filter((p) => (!chosen.length || chosen.includes(p.kind)) && (source === "any" || p.source === source));
        const sel = id ? all.find((p) => String(p.id) === id) : null;
        return (
          <>
            <div className="filters">
              <span className="muted">kinds:</span>
              {Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, n]) => (
                <button key={k} className={chosen.includes(k) ? "chip active" : "chip"} onClick={() => setKinds((chosen.includes(k) ? chosen.filter((x) => x !== k) : [...chosen, k]).join(","))}>{k} ({n})</button>
              ))}
              <span className="muted">source:</span>
              <Chips items={["any", "recorded", "inferred"]} value={source} onChange={setSource} />
            </div>
            <div className={sel ? "split" : ""}>
              <Section title={`${rows.length} problems`} note="newest first · inferred = read from a chat's tool output">
                <Table rows={rows} rowKey={(r: any) => String(r.id)} selected={id} onRowClick={(r: any) => setId(String(r.id))} maxHeight={window.innerHeight - 240} cols={[
                  { key: "time", width: "150px" }, { key: "source", width: "85px", render: (r: any) => <Badge>{r.source}</Badge> }, { key: "kind", width: "120px" },
                  { key: "node", width: "90px" }, { key: "phase", width: "110px" }, { key: "attempt", width: "60px" }, { key: "where", width: "minmax(100px, 1fr)" },
                  { key: "summary", width: "minmax(240px, 4fr)", mono: true }]} />
              </Section>
              {sel && (
                <Section title={`${sel.kind}: ${sel.node ?? ""} ${sel.phase ?? ""} ${sel.attempt ?? ""}`} actions={<button className="btn small" onClick={() => setId(null)}>close</button>}>
                  <div className="row gap">
                    {sel.node && sel.phase && <Link className="btn small" to={`/node/${sel.node}/agents?${q({ pa: `${sel.phase}-${sel.attempt}` })}`}>open the attempt's agents</Link>}
                    {sel.node && <Link className="btn small" to={`/node/${sel.node}`}>open {sel.node}</Link>}
                  </div>
                  <Text text={sel.summary} lines={20} />
                  {sel.detail && <Section title="tool output"><Text text={sel.detail} lines={200} /></Section>}
                  {sel.seq !== null && sel.seq !== undefined ? <EventDetail seq={sel.seq} /> : <div className="muted">no event (inferred from a chat)</div>}
                </Section>
              )}
            </div>
            {!all.length && <Empty>No problems recorded.</Empty>}
          </>
        );
      }}</Q>
    </div>
  );
}
