import { useVirtualizer } from "@tanstack/react-virtual";
import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Chart, fmtDur, fmtInt, fmtTs, fmtUsd, Markdown, Text } from "../ui";

export type Item = {
  kind: string; title: string | null; text: string; at?: number; ref?: string; prior?: boolean; tool?: string; links?: Record<string, string>;
  turn?: number; ts?: number; done_ts?: number; prompt_tokens?: number; completion_tokens?: number; cost_usd?: number; latency_s?: number; status?: number;
};

const LABEL: Record<string, string> = {
  system: "system prompt", tools: "tools offered", user: "user", assistant: "assistant", reasoning: "reasoning", tool_call: "tool call",
  tool_output: "tool output", tool_error: "tool error", usage: "usage", note: "note", context: "context after compaction",
  compaction_request: "compaction request", compaction_summary: "compaction summary",
};
export const KINDS = Object.keys(LABEL);
const PROSE = new Set(["user", "assistant", "system", "context", "compaction_request", "compaction_summary", "reasoning"]);
const FOLDED = new Set(["system", "tools", "context"]);

function Highlighted({ text, needle }: { text: string; needle: string }) {
  if (!needle) return <>{text}</>;
  const parts = text.split(new RegExp(`(${needle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi"));
  return <>{parts.map((p, i) => (i % 2 ? <mark key={i}>{p}</mark> : p))}</>;
}

export function ChatItem({ it, needle = "", onJump, index }: { it: Item; needle?: string; onJump?: (ref: string, kind: "call" | "output") => void; index?: number }) {
  const [open, setOpen] = useState(!FOLDED.has(it.kind) && !it.prior);
  const foldable = FOLDED.has(it.kind) || !!it.prior;
  if (it.kind === "usage") {
    return (
      <div className="chat-usage" id={`turn-${it.turn}`}>
        <b>{it.turn ? `turn ${it.turn}` : "compaction call"}</b> · {fmtInt(it.prompt_tokens)} in / {fmtInt(it.completion_tokens)} out
        {it.cost_usd !== undefined && it.cost_usd !== null && <> · {fmtUsd(it.cost_usd)}</>}
        {it.latency_s !== undefined && it.latency_s !== null && <> · {fmtDur(it.latency_s)}</>}
        {it.ts && <> · sent {fmtTs(it.ts)}</>}
        {it.status && it.status !== 200 && <b className="neg"> · status {it.status}</b>}
      </div>
    );
  }
  if (it.kind === "note") return <div className="chat-note">{it.text}</div>;
  const body = needle ? <pre className="wrap"><Highlighted text={it.text} needle={needle} /></pre>
    : PROSE.has(it.kind) ? <Markdown text={it.text} />
    : <Text text={it.text} lines={it.kind === "tool_error" ? 200 : 40} />;
  return (
    <div className={`chat-item k-${it.kind} ${it.prior ? "prior" : ""}`} data-index={index}>
      <div className="chat-head" onClick={() => foldable && setOpen(!open)}>
        {it.prior && <span className="chat-prior">earlier context</span>}
        <span className="chat-kind">{LABEL[it.kind] ?? it.kind}</span>
        {it.title && it.title !== LABEL[it.kind] && <span className="chat-title">{it.title}</span>}
        {it.ref && onJump && (it.kind === "tool_call"
          ? <button className="btn tiny" onClick={(e) => { e.stopPropagation(); onJump(it.ref!, "output"); }}>↓ its output</button>
          : <button className="btn tiny" onClick={(e) => { e.stopPropagation(); onJump(it.ref!, "call"); }}>↑ its call</button>)}
        <span className="grow" />
        {it.at !== undefined && <span className="muted small">turn {it.at}</span>}
        <span className="muted small">{fmtInt(it.text?.length)} chars</span>
        {foldable && <button className="btn tiny">{open ? "fold" : "show"}</button>}
      </div>
      {open ? body : <div className="muted small chat-preview">{it.text.slice(0, 240).replace(/\n/g, " ")}{it.text.length > 240 ? "…" : ""}</div>}
      {it.links && (
        <div className="chat-links">
          {Object.entries(it.links).map(([c, h]) => <Link key={c} to={`/files?${new URLSearchParams({ path: h })}`} target="_blank" className="chip" title={h}>📄 {c}</Link>)}
        </div>
      )}
    </div>
  );
}

/** A whole conversation: every item, virtualized; search, kind filters, call/output pairing, a turn strip. */
export function ChatView({ items, turn, onTurn, height = "calc(100vh - 260px)", header }: {
  items: Item[]; turn?: number | null; onTurn?: (t: number) => void; height?: string | number; header?: ReactNode;
}) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [errorsOnly, setErrorsOnly] = useState(false);
  const [hidePrior, setHidePrior] = useState(false);
  const [search, setSearch] = useState("");
  const [hit, setHit] = useState(0);
  const [jump, setJump] = useState("");
  const shown = useMemo(() => items.map((it, i) => [it, i] as [Item, number]).filter(([it]) =>
    !hidden.has(it.kind) && !(hidePrior && it.prior) && (!errorsOnly || it.kind === "tool_error" || (it.kind === "usage" && it.status && it.status !== 200) || it.kind === "note")), [items, hidden, errorsOnly, hidePrior]);
  const needle = search.trim().toLowerCase();
  const hits = useMemo(() => needle ? shown.map(([it], k) => [it, k] as [Item, number]).filter(([it]) => (it.text || "").toLowerCase().includes(needle) || (it.title || "").toLowerCase().includes(needle)).map(([, k]) => k) : [], [shown, needle]);
  const scroller = useRef<HTMLDivElement>(null);
  const virtual = useVirtualizer({ count: shown.length, getScrollElement: () => scroller.current, estimateSize: () => 120, overscan: 6 });
  const toTurn = (t: number) => {
    const k = shown.findIndex(([it]) => it.at === t);
    if (k >= 0) virtual.scrollToIndex(k, { align: "start" });
  };
  const shownTurn = useRef<number | null>(null);         // a live refresh must not pull the reader back
  useEffect(() => {
    if (!turn || turn === shownTurn.current || !items.some((it) => it.at === turn)) return;
    shownTurn.current = turn;
    setTimeout(() => toTurn(turn), 50);
  }, [turn, items]);
  useEffect(() => { if (hits.length) virtual.scrollToIndex(hits[Math.min(hit, hits.length - 1)], { align: "start" }); }, [hit, hits]);
  const onJump = (ref: string, want: "call" | "output") => {
    const k = shown.findIndex(([it]) => it.ref === ref && (want === "call" ? it.kind === "tool_call" : it.kind !== "tool_call"));
    if (k >= 0) virtual.scrollToIndex(k, { align: "start" });
  };
  const usage = items.filter((i) => i.kind === "usage");
  const priorCount = useMemo(() => items.filter((i) => i.prior).length, [items]);
  const counts = useMemo(() => Object.fromEntries(KINDS.map((k) => [k, items.filter((i) => i.kind === k).length])), [items]);
  const strip = useMemo(() => ({
    tooltip: { trigger: "axis" },
    legend: { top: 0, data: ["prompt tokens", "completion tokens", "latency s"] },
    grid: { left: 60, right: 50, top: 24, bottom: 20 },
    xAxis: { type: "category", data: usage.map((u) => u.turn), axisLabel: { fontSize: 9 } },
    yAxis: [{ type: "value", name: "tokens", nameTextStyle: { fontSize: 9 } }, { type: "value", name: "s", nameTextStyle: { fontSize: 9 } }],
    series: [
      { name: "prompt tokens", type: "bar", stack: "t", data: usage.map((u) => u.prompt_tokens ?? null), itemStyle: { color: "#5b8ff9" } },
      { name: "completion tokens", type: "bar", stack: "t", data: usage.map((u) => u.completion_tokens ?? null), itemStyle: { color: "#5ad8a6" } },
      { name: "latency s", type: "line", yAxisIndex: 1, symbol: "none", data: usage.map((u) => u.latency_s ?? null), lineStyle: { width: 1 }, itemStyle: { color: "#f6bd16" } },
    ],
  }), [items]);
  return (
    <div className="chatview">
      {header}
      {usage.length > 1 && <Chart option={strip} height={110} onClick={(p) => { const t = usage[p.dataIndex]?.turn; if (t) { toTurn(t); onTurn?.(t); } }} />}
      <div className="row gap wrap chat-tools">
        <input className="search" placeholder="search this conversation…" value={search} onChange={(e) => { setSearch(e.target.value); setHit(0); }} />
        {needle && <span className="small">{hits.length ? `${Math.min(hit, hits.length - 1) + 1} / ${hits.length}` : "no match"}</span>}
        {needle && <button className="btn small" onClick={() => setHit((hit - 1 + hits.length) % Math.max(hits.length, 1))}>prev</button>}
        {needle && <button className="btn small" onClick={() => setHit((hit + 1) % Math.max(hits.length, 1))}>next</button>}
        <input className="turn-input" placeholder="turn #" value={jump} onChange={(e) => setJump(e.target.value)} onKeyDown={(e) => e.key === "Enter" && jump && (toTurn(+jump), onTurn?.(+jump))} />
        <label className="small"><input type="checkbox" checked={errorsOnly} onChange={(e) => setErrorsOnly(e.target.checked)} /> errors only</label>
        {priorCount > 0 && <label className="small"><input type="checkbox" checked={hidePrior} onChange={(e) => setHidePrior(e.target.checked)} /> hide earlier context ({fmtInt(priorCount)})</label>}
        <span className="muted small">{fmtInt(shown.length)} of {fmtInt(items.length)} items · {usage.length} turns</span>
      </div>
      <div className="row gap wrap small kind-filters">
        {KINDS.filter((k) => counts[k]).map((k) => (
          <label key={k} className={`kind-toggle k-${k}`}><input type="checkbox" checked={!hidden.has(k)} onChange={() => {
            const h = new Set(hidden); h.has(k) ? h.delete(k) : h.add(k); setHidden(h);
          }} />{LABEL[k]} ({counts[k]})</label>
        ))}
      </div>
      <div className="chat-scroll" ref={scroller} style={{ height }}>
        <div style={{ height: virtual.getTotalSize(), position: "relative" }}>
          {virtual.getVirtualItems().map((v) => {
            const [it] = shown[v.index];
            const isHit = needle && hits[Math.min(hit, hits.length - 1)] === v.index;
            return (
              <div key={v.key} data-index={v.index} ref={virtual.measureElement} className={`chat-row ${isHit ? "hit" : ""}`} style={{ transform: `translateY(${v.start}px)` }}>
                <ChatItem it={it} needle={needle && (it.text || "").toLowerCase().includes(needle) ? needle : ""} onJump={onJump} />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
