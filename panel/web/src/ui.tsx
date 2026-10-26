import * as echarts from "echarts";
import { createContext, ReactNode, useContext, useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Link, useSearchParams } from "react-router-dom";

// ---- formatting ----

const pad = (n: number) => String(n).padStart(2, "0");
export function fmtTs(t?: number | null) {
  if (t === null || t === undefined) return "";
  const d = new Date(t * 1000);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
export function fmtDur(s?: number | null) {
  if (s === null || s === undefined || !isFinite(s)) return "";
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  const m = Math.floor(s / 60), h = Math.floor(m / 60), d = Math.floor(h / 24);
  if (d) return `${d} d ${h % 24} h`;
  if (h) return `${h} h ${m % 60} min`;
  return `${m} min ${Math.round(s % 60)} s`;
}
export const fmtInt = (n?: number | null) => (n === null || n === undefined ? "" : n.toLocaleString("en-US"));
export function fmtBytes(n?: number | null) {
  if (n === null || n === undefined) return "";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i ? 1 : 0)} ${u[i]}`;
}
export const fmtUsd = (n?: number | null) => (n === null || n === undefined ? "n/a" : `$${n.toFixed(4)}`);
export function fmtVal(v: any): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isInteger(v) ? fmtInt(v) : v.toFixed(4);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
export const deltaClass = (d?: number | null) => (d === null || d === undefined || d === 0 ? "" : d > 0 ? "pos" : "neg");
export const fmtDelta = (d?: number | null) => (d === null || d === undefined ? "" : `${d > 0 ? "+" : ""}${d.toFixed(4)}`);

// ---- theme ----

export const ThemeContext = createContext<{ dark: boolean; toggle: () => void }>({ dark: true, toggle: () => {} });
export const useTheme = () => useContext(ThemeContext);

// ---- small pieces ----

export function Err({ error }: { error: any }) {
  if (!error) return null;
  return <div className="err">Error: {String(error.message || error)}</div>;
}
export const Loading = ({ what }: { what?: string }) => <div className="muted pad">Loading{what ? ` ${what}` : ""}…</div>;
export const Empty = ({ children }: { children: ReactNode }) => <div className="muted pad empty">{children}</div>;

/** Shows the query's error, a loading line until its first data, then children(data). */
export function Q<T>({ query, what, children }: { query: { data?: T; error: any; isLoading: boolean }; what?: string; children: (d: T) => ReactNode }) {
  if (query.error) return <Err error={query.error} />;
  if (query.data === undefined) return <Loading what={what} />;
  return <>{children(query.data)}</>;
}

const STATUS: Record<string, string> = {
  ok: "good", scored: "good", passed: "good", done: "good", accepted: "good", success: "good", true: "good", recorded: "info",
  running: "info", started: "info", inferred: "warn", warning: "warn", cancelled: "warn", canceled: "warn", "no end recorded": "warn",
  error: "bad", failed: "bad", rejected: "bad", false: "bad", critical: "bad",
};
export function statusTone(s: any): string {
  const k = String(s ?? "").toLowerCase();
  if (STATUS[k]) return STATUS[k];
  if (/fail|error|reject|invalid/.test(k)) return "bad";
  if (/warn|cancel|stall|retry/.test(k)) return "warn";
  if (/ok|pass|done|scored|success|accept/.test(k)) return "good";
  return "neutral";
}
export const Badge = ({ children, tone }: { children: ReactNode; tone?: string }) => (
  <span className={`badge ${tone ?? statusTone(children)}`}>{children}</span>
);

export function Section({ title, actions, children, collapsed, id, note }: {
  title: ReactNode; actions?: ReactNode; children: ReactNode; collapsed?: boolean; id?: string; note?: ReactNode;
}) {
  const [open, setOpen] = useState(!collapsed);
  return (
    <section className="section" id={id}>
      <header>
        <button className="fold" onClick={() => setOpen(!open)}>{open ? "▾" : "▸"}</button>
        <h3 onClick={() => setOpen(!open)}>{title}</h3>
        {note && <span className="muted small">{note}</span>}
        <div className="grow" />
        {actions}
      </header>
      {open && <div className="section-body">{children}</div>}
    </section>
  );
}

export function KV({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <div className="kv">
      {items.filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v], i) => (
        <div key={i} className="kv-row"><span className="kv-k">{k}</span><span className="kv-v">{v}</span></div>
      ))}
    </div>
  );
}

export function Tabs({ items, value, onChange }: { items: { key: string; label: ReactNode; to?: string }[]; value: string; onChange?: (k: string) => void }) {
  return (
    <nav className="tabs">
      {items.map((t) => t.to
        ? <Link key={t.key} to={t.to} className={t.key === value ? "tab active" : "tab"}>{t.label}</Link>
        : <button key={t.key} className={t.key === value ? "tab active" : "tab"} onClick={() => onChange?.(t.key)}>{t.label}</button>)}
    </nav>
  );
}

export function Chips<T extends string | number>({ items, value, onChange, render }: {
  items: T[]; value: T | null | undefined; onChange: (v: T) => void; render?: (v: T) => ReactNode;
}) {
  return (
    <div className="chips">
      {items.map((v) => (
        <button key={String(v)} className={v === value ? "chip active" : "chip"} onClick={() => onChange(v)}>{render ? render(v) : String(v)}</button>
      ))}
    </div>
  );
}

export function Copy({ text, label = "copy" }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button className="btn small" onClick={() => { navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1000); }}>
      {done ? "copied" : label}
    </button>
  );
}

/** URL search param as state, so every selection can be linked and survives a reload. */
let pending: URLSearchParams | null = null;     // several params set in one handler add up instead of the last winning

export function useParam(name: string, fallback = ""): [string, (v: string | null) => void] {
  const [params, setParams] = useSearchParams();
  const value = params.get(name) ?? fallback;
  const set = (v: string | null) => {
    if (!pending) { pending = new URLSearchParams(window.location.search); queueMicrotask(() => { pending = null; }); }
    if (v === null || v === "") pending.delete(name); else pending.set(name, v);
    setParams(new URLSearchParams(pending), { replace: true });
  };
  return [value, set];
}

// ---- text ----

/** Preformatted text, folded after `lines` lines; wrap toggle and copy. */
export function Text({ text, lines = 30, className, lang }: { text: string | null | undefined; lines?: number; className?: string; lang?: string }) {
  const [open, setOpen] = useState(false);
  const [wrap, setWrap] = useState(true);
  const t = text ?? "";
  const all = useMemo(() => t.split("\n"), [t]);
  const long = all.length > lines;
  const shown = long && !open ? all.slice(0, lines).join("\n") : t;
  return (
    <div className={`text ${className ?? ""}`}>
      <div className="text-tools">
        <span className="muted small">{fmtInt(all.length)} lines · {fmtInt(t.length)} chars{lang ? ` · ${lang}` : ""}</span>
        <button className="btn small" onClick={() => setWrap(!wrap)}>{wrap ? "no wrap" : "wrap"}</button>
        <Copy text={t} />
      </div>
      <pre className={wrap ? "wrap" : ""}>{shown}</pre>
      {long && <button className="btn small more" onClick={() => setOpen(!open)}>{open ? "fold" : `show all ${fmtInt(all.length)} lines`}</button>}
    </div>
  );
}

export function Markdown({ text }: { text: string | null | undefined }) {
  const [rawMode, setRaw] = useState(false);
  if (!text) return null;
  return (
    <div className="md-wrap">
      <div className="text-tools"><button className="btn small" onClick={() => setRaw(!rawMode)}>{rawMode ? "rendered" : "raw"}</button><Copy text={text} /></div>
      {rawMode ? <pre className="wrap">{text}</pre> : <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown></div>}
    </div>
  );
}

// ---- JSON ----

function JsonNode({ k, v, depth, open }: { k?: string; v: any; depth: number; open: number }) {
  const [isOpen, setOpen] = useState(depth < open);
  const [full, setFull] = useState(false);
  const key = k !== undefined ? <span className="j-key">{k}: </span> : null;
  if (v === null || typeof v !== "object") {
    let body: ReactNode;
    if (typeof v === "string") {
      const long = v.length > 300 || v.includes("\n");
      body = long
        ? <span className="j-str">
            {full ? <pre className="wrap j-pre">{v}</pre> : <span>"{v.slice(0, 300).replace(/\n/g, "⏎")}{v.length > 300 ? "…" : ""}"</span>}
            <button className="btn tiny" onClick={() => setFull(!full)}>{full ? "fold" : `${fmtInt(v.length)} chars`}</button>
          </span>
        : <span className="j-str">"{v}"</span>;
    } else body = <span className={`j-${v === null ? "null" : typeof v}`}>{String(v)}</span>;
    return <div className="j-row">{key}{body}</div>;
  }
  const entries = Array.isArray(v) ? v.map((x, i) => [String(i), x] as [string, any]) : Object.entries(v);
  const brace = Array.isArray(v) ? ["[", "]"] : ["{", "}"];
  return (
    <div className="j-row">
      <span className="j-toggle" onClick={() => setOpen(!isOpen)}>{isOpen ? "▾" : "▸"} </span>{key}
      <span className="muted">{brace[0]}{!isOpen && <span onClick={() => setOpen(true)} className="j-count"> {entries.length} {Array.isArray(v) ? "items" : "keys"} </span>}{!isOpen && brace[1]}</span>
      {isOpen && <div className="j-children">{entries.map(([ek, ev]) => <JsonNode key={ek} k={ek} v={ev} depth={depth + 1} open={open} />)}</div>}
      {isOpen && <span className="muted">{brace[1]}</span>}
    </div>
  );
}

export function Json({ value, open = 2, title }: { value: any; open?: number; title?: ReactNode }) {
  const [rawMode, setRaw] = useState(false);
  const [key, setKey] = useState(0);
  const [depth, setDepth] = useState(open);
  if (value === undefined) return null;
  const text = JSON.stringify(value, null, 2);
  return (
    <div className="json">
      <div className="text-tools">
        {title && <span className="json-title">{title}</span>}
        <span className="muted small">{fmtInt(text.length)} chars</span>
        <button className="btn small" onClick={() => { setDepth(99); setKey(key + 1); }}>expand all</button>
        <button className="btn small" onClick={() => { setDepth(1); setKey(key + 1); }}>collapse</button>
        <button className="btn small" onClick={() => setRaw(!rawMode)}>{rawMode ? "tree" : "raw"}</button>
        <Copy text={text} />
      </div>
      {rawMode ? <pre className="wrap">{text}</pre> : <div className="j-tree" key={key}><JsonNode v={value} depth={0} open={depth} /></div>}
    </div>
  );
}

// ---- table ----

export type Col<T = any> = {
  key: string; label?: ReactNode; width?: string; render?: (row: T) => ReactNode; value?: (row: T) => any;
  align?: "right" | "left"; mono?: boolean; nowrap?: boolean; title?: string;
};

function cellValue(col: Col, row: any) {
  return col.value ? col.value(row) : row[col.key];
}

/** Sortable, filterable, virtualized table. Every row stays reachable: nothing is paged away. */
export function Table<T = any>({ rows, cols, rowKey, onRowClick, selected, maxHeight = 520, filter = true, sort: initialSort, empty = "No rows.", title, actions, rowClass }: {
  rows: T[]; cols: Col<T>[]; rowKey?: (r: T, i: number) => string; onRowClick?: (r: T) => void; selected?: string | null;
  maxHeight?: number; filter?: boolean; sort?: [string, "asc" | "desc"]; empty?: ReactNode; title?: ReactNode; actions?: ReactNode;
  rowClass?: (r: T) => string;
}) {
  const [sort, setSort] = useState<[string, "asc" | "desc"] | null>(initialSort ?? null);
  const [search, setSearch] = useState("");
  const [colFilters, setColFilters] = useState<Record<string, string>>({});
  const [showFilters, setShowFilters] = useState(false);
  const shown = useMemo(() => {
    let out = rows.map((r, i) => [r, i] as [T, number]);
    const s = search.trim().toLowerCase();
    if (s) out = out.filter(([r]) => cols.some((c) => fmtVal(cellValue(c, r)).toLowerCase().includes(s)));
    for (const [k, f] of Object.entries(colFilters)) {
      const col = cols.find((c) => c.key === k);
      const needle = f.trim().toLowerCase();
      if (!col || !needle) continue;
      const m = needle.match(/^(>=|<=|>|<|=)\s*(-?[\d.e+-]+)$/);
      out = out.filter(([r]) => {
        const v = cellValue(col, r);
        if (m && typeof v === "number") {
          const x = parseFloat(m[2]);
          return m[1] === ">" ? v > x : m[1] === "<" ? v < x : m[1] === ">=" ? v >= x : m[1] === "<=" ? v <= x : v === x;
        }
        return fmtVal(v).toLowerCase().includes(needle);
      });
    }
    if (sort) {
      const col = cols.find((c) => c.key === sort[0]);
      if (col) {
        const dir = sort[1] === "asc" ? 1 : -1;
        out = [...out].sort(([a], [b]) => {
          const x = cellValue(col, a), y = cellValue(col, b);
          if (x === y) return 0;
          if (x === null || x === undefined || x === "") return 1;
          if (y === null || y === undefined || y === "") return -1;
          return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true })) * dir;
        });
      }
    }
    return out;
  }, [rows, cols, search, colFilters, sort]);
  const scroller = useRef<HTMLDivElement>(null);
  const virtual = useVirtualizer({ count: shown.length, getScrollElement: () => scroller.current, estimateSize: () => 27, overscan: 12 });
  const template = cols.map((c) => c.width ?? "minmax(90px, 1fr)").join(" ");
  const tsv = () => [cols.map((c) => c.key).join("\t"), ...shown.map(([r]) => cols.map((c) => fmtVal(cellValue(c, r)).replace(/[\t\n]/g, " ")).join("\t"))].join("\n");
  const clickSort = (k: string) => setSort(!sort || sort[0] !== k ? [k, "asc"] : sort[1] === "asc" ? [k, "desc"] : null);
  return (
    <div className="table">
      {(filter || title || actions) && (
        <div className="table-tools">
          {title && <span className="table-title">{title}</span>}
          {filter && <input className="search" placeholder="filter rows…" value={search} onChange={(e) => setSearch(e.target.value)} />}
          {filter && <button className={showFilters ? "btn small active" : "btn small"} onClick={() => setShowFilters(!showFilters)} title="per-column filters; numbers accept >, <, >=, <=, =">column filters</button>}
          <span className="muted small">{shown.length === rows.length ? `${fmtInt(rows.length)} rows` : `${fmtInt(shown.length)} of ${fmtInt(rows.length)} rows`}</span>
          <div className="grow" />
          {actions}
          <Copy text={rows.length ? tsv() : ""} label="copy TSV" />
        </div>
      )}
      <div className="table-scroll" ref={scroller} style={{ maxHeight }}>
        <div className="t-head" style={{ gridTemplateColumns: template }}>
          {cols.map((c) => (
            <div key={c.key} className={`t-th ${c.align === "right" ? "right" : ""}`} onClick={() => clickSort(c.key)} title={c.title ?? (typeof c.label === "string" ? c.label : c.key)}>
              {c.label ?? c.key}{sort?.[0] === c.key ? (sort[1] === "asc" ? " ▲" : " ▼") : ""}
            </div>
          ))}
        </div>
        {showFilters && (
          <div className="t-head t-filters" style={{ gridTemplateColumns: template }}>
            {cols.map((c) => <div key={c.key} className="t-th"><input value={colFilters[c.key] ?? ""} onChange={(e) => setColFilters({ ...colFilters, [c.key]: e.target.value })} /></div>)}
          </div>
        )}
        {shown.length === 0 ? <div className="muted pad">{empty}</div> : (
          <div style={{ height: virtual.getTotalSize(), position: "relative" }}>
            {virtual.getVirtualItems().map((v) => {
              const [r, i] = shown[v.index];
              const k = rowKey ? rowKey(r, i) : String(i);
              return (
                <div key={k} data-index={v.index} ref={virtual.measureElement}
                  className={`t-row ${onRowClick ? "clickable" : ""} ${selected === k ? "selected" : ""} ${rowClass ? rowClass(r) : ""}`}
                  style={{ gridTemplateColumns: template, transform: `translateY(${v.start}px)` }}
                  onClick={() => onRowClick?.(r)}>
                  {cols.map((c) => {
                    const v = cellValue(c, r);
                    return (
                      <div key={c.key} className={`t-td ${c.align === "right" || typeof v === "number" ? "right" : ""} ${c.mono ? "mono" : ""} ${c.nowrap ? "nowrap" : ""}`}
                        title={typeof v === "number" ? String(v) : undefined}>
                        {c.render ? c.render(r) : fmtVal(v)}
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ---- charts ----

/** An ECharts chart; new options merge into the old ones, so zoom and legend picks survive a live refresh. */
export function Chart({ option, height = 300, onClick, onReady, className }: {
  option: any; height?: number | string; onClick?: (p: any) => void; onReady?: (c: echarts.ECharts) => void; className?: string;
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const click = useRef(onClick);
  click.current = onClick;
  const { dark } = useTheme();
  useEffect(() => {
    const c = echarts.init(el.current!, dark ? "dark" : undefined, { renderer: "canvas" });
    chart.current = c;
    c.on("click", (p) => click.current?.(p));
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current!);
    onReady?.(c);
    return () => { ro.disconnect(); c.dispose(); chart.current = null; };
  }, [dark]);
  useEffect(() => {
    chart.current?.setOption({ backgroundColor: "transparent", animation: false, ...option }, { replaceMerge: ["series", "xAxis", "yAxis", "grid", "visualMap"] });
  }, [option, dark]);
  return <div ref={el} className={className} style={{ height, width: "100%" }} />;
}

export const PALETTE = ["#5b8ff9", "#5ad8a6", "#f6bd16", "#e8684a", "#6dc8ec", "#9270ca", "#ff9d4d", "#269a99", "#ff99c3", "#5d7092", "#b8e986", "#c2b0e2"];
export const colorOf = (name: string, names: string[]) => PALETTE[Math.max(0, names.indexOf(name)) % PALETTE.length];

// ---- media ----

/** A video that only loads once scrolled into view. */
export function LazyVideo({ src, className, controls = true, muted = true, onRef }: { src: string; className?: string; controls?: boolean; muted?: boolean; onRef?: (v: HTMLVideoElement | null) => void }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const io = new IntersectionObserver(([e]) => e.isIntersecting && setSeen(true), { rootMargin: "200px" });
    if (ref.current) io.observe(ref.current);
    return () => io.disconnect();
  }, []);
  useEffect(() => { onRef?.(ref.current); }, [ref.current]);
  return <video ref={ref} className={className} src={seen ? `${src}#t=0.1` : undefined} preload="metadata" controls={controls} muted={muted} playsInline />;
}

/** Plays, pauses, seeks and changes speed of several videos together. */
export function SyncedVideos({ items }: { items: { label: ReactNode; src: string | null }[] }) {
  const refs = useRef<(HTMLVideoElement | null)[]>([]);
  const [rate, setRate] = useState(1);
  const [linked, setLinked] = useState(true);
  const leader = useRef(-1);
  const all = () => refs.current.filter(Boolean) as HTMLVideoElement[];
  const follow = (i: number, what: "play" | "pause" | "seek") => {
    if (!linked || (leader.current !== -1 && leader.current !== i)) return;
    leader.current = i;
    const src = refs.current[i]!;
    for (const v of all()) {
      if (v === src) continue;
      if (what === "seek" || Math.abs(v.currentTime - src.currentTime) > 0.05) v.currentTime = src.currentTime;
      if (what === "play") v.play().catch(() => {}); else if (what === "pause") v.pause();
    }
    setTimeout(() => (leader.current = -1), 50);
  };
  useEffect(() => { all().forEach((v) => (v.playbackRate = rate)); }, [rate, items]);
  const step = (d: number) => all().forEach((v) => { v.pause(); v.currentTime = Math.max(0, v.currentTime + d); });
  return (
    <div>
      <div className="row gap">
        <label className="small"><input type="checkbox" checked={linked} onChange={(e) => setLinked(e.target.checked)} /> play and seek together</label>
        <span className="small muted">speed</span>
        <Chips items={[0.25, 0.5, 1, 2]} value={rate} onChange={setRate} render={(r) => `${r}×`} />
        <button className="btn small" onClick={() => step(-1 / 24)}>◀ frame</button>
        <button className="btn small" onClick={() => step(1 / 24)}>frame ▶</button>
        <button className="btn small" onClick={() => { all().forEach((v) => { v.currentTime = 0; v.play().catch(() => {}); }); }}>restart both</button>
      </div>
      <div className="video-grid" style={{ gridTemplateColumns: `repeat(${items.length}, 1fr)` }}>
        {items.map((it, i) => (
          <div key={i} className="video-cell">
            <div className="video-label">{it.label}</div>
            {it.src ? <video ref={(v) => { refs.current[i] = v; }} src={it.src} controls muted playsInline preload="metadata"
              onPlay={() => follow(i, "play")} onPause={() => follow(i, "pause")} onSeeked={() => follow(i, "seek")} />
              : <div className="muted pad">no video</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

export function Drawer({ open, onClose, title, children, wide }: { open: boolean; onClose: () => void; title: ReactNode; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);
  if (!open) return null;
  return (
    <div className="drawer-back" onClick={onClose}>
      <aside className={`drawer ${wide ? "wide" : ""}`} onClick={(e) => e.stopPropagation()}>
        <header><h3>{title}</h3><div className="grow" /><button className="btn" onClick={onClose}>close (Esc)</button></header>
        <div className="drawer-body">{children}</div>
      </aside>
    </div>
  );
}
