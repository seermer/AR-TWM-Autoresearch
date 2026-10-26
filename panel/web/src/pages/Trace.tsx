import { useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";
import { get, q, useApi } from "../api";
import { EventDetail } from "../parts/Event";
import { Empty, fmtInt, Q, Section, Table, useParam } from "../ui";
import { eventCols } from "./Overview";

const PAGE = 5000;

export default function Trace() {
  const choices = useApi("/api/event-choices");
  const [node, setNode] = useParam("node");
  const [phase, setPhase] = useParam("phase");
  const [attempt, setAttempt] = useParam("attempt");
  const [types, setTypes] = useParam("types");
  const [component, setComponent] = useParam("component");
  const [text, setText] = useParam("text");
  const [gpu, setGpu] = useParam("gpu");
  const [seq, setSeq] = useParam("seq");
  const [draft, setDraft] = useState(text);
  const filters = q({ node, phase, attempt, types, component, text, gpu: gpu ? "true" : "", limit: PAGE });
  const query = useInfiniteQuery({
    queryKey: ["/api/events", filters], initialPageParam: 0,
    queryFn: ({ pageParam }) => get(`/api/events?${filters}&offset=${pageParam}`),
    getNextPageParam: (last: any) => (last.offset + last.rows.length < last.total ? last.offset + last.rows.length : undefined),
  });
  const data = query.data && { total: query.data.pages[0].total, rows: query.data.pages.flatMap((p: any) => p.rows) };
  const typeList = types ? types.split(",") : [];
  const select = (label: string, value: string, set: (v: string | null) => void, options: string[]) => (
    <label className="field">{label}
      <select value={value} onChange={(e) => set(e.target.value)}><option value="">any</option>{options.map((o) => <option key={o} value={o}>{o}</option>)}</select>
    </label>
  );
  return (
    <div className="page">
      <Q query={choices} what="filters">{(c) => (
        <div className="filters">
          {select("source (node, run, gpu)", node, setNode, c.node)}
          {select("phase", phase, setPhase, c.phase)}
          {select("attempt (0 = a kernel step)", attempt, setAttempt, c.attempt)}
          {select("component", component, setComponent, c.component)}
          <label className="field">types
            <select value="" onChange={(e) => e.target.value && setTypes([...new Set([...typeList, e.target.value])].join(","))}>
              <option value="">add a type…</option>{c.type.map((t: string) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          {typeList.map((t) => <button key={t} className="chip active" onClick={() => setTypes(typeList.filter((x) => x !== t).join(","))}>{t} ✕</button>)}
          <label className="field">search the event records
            <input value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => e.key === "Enter" && setText(draft)} onBlur={() => setText(draft)} placeholder="Enter to apply" />
          </label>
          <label className="small"><input type="checkbox" checked={!!gpu} onChange={(e) => setGpu(e.target.checked ? "1" : null)} /> include GPU samples</label>
          <button className="btn small" onClick={() => { [setNode, setPhase, setAttempt, setTypes, setComponent, setText].forEach((f) => f(null)); setDraft(""); }}>clear</button>
        </div>
      )}</Q>
      <div className={seq ? "split" : ""}>
        <Q query={{ data, error: query.error, isLoading: query.isLoading }} what="events">{(d: any) => (
          <Section title={`${fmtInt(d.total)} events`} note={d.total > d.rows.length ? `first ${fmtInt(d.rows.length)} loaded, oldest first` : "oldest first"}
            actions={query.hasNextPage ? <button className="btn small" disabled={query.isFetchingNextPage} onClick={() => query.fetchNextPage()}>
              {query.isFetchingNextPage ? "loading…" : `load ${fmtInt(Math.min(PAGE, d.total - d.rows.length))} more`}</button> : null}>
            <Table rows={d.rows} cols={eventCols} rowKey={(r: any) => String(r.seq)} selected={seq} onRowClick={(r: any) => setSeq(String(r.seq))} maxHeight={window.innerHeight - 260} />
          </Section>
        )}</Q>
        {seq ? <Section title={`event ${seq}`} actions={<button className="btn small" onClick={() => setSeq(null)}>close</button>}><EventDetail seq={+seq} /></Section> : null}
      </div>
      {!seq && <Empty>Click an event for its full record, its payload and, for model calls, the request or reply as a chat.</Empty>}
    </div>
  );
}
