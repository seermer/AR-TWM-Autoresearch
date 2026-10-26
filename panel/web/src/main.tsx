import { lazy, StrictMode, Suspense, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Link, NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { LiveContext, queryClient, useApi, useLive, useLiveStatus } from "./api";
import { fmtDur, fmtInt, fmtTs, fmtUsd, ThemeContext } from "./ui";
import "./styles.css";

const Overview = lazy(() => import("./pages/Overview"));
const NodePage = lazy(() => import("./pages/Node"));
const Trace = lazy(() => import("./pages/Trace"));
const Problems = lazy(() => import("./pages/Problems"));
const Compare = lazy(() => import("./pages/Compare"));
const Cost = lazy(() => import("./pages/Cost"));
const Selection = lazy(() => import("./pages/Selection"));
const Files = lazy(() => import("./pages/Files"));
const TimelinePage = lazy(() => import("./pages/TimelinePage"));

function StatusBar() {
  const { status: s, connected, paused, setPaused, updated } = useLiveStatus();
  const [, tick] = useState(0);
  useEffect(() => { const t = setInterval(() => tick((x) => x + 1), 1000); return () => clearInterval(t); }, []);
  if (!s) return <div className="statusbar muted">{connected ? "loading run…" : "connecting…"}</div>;
  const st = s.state || {};
  const since = st.since ? Date.now() / 1000 - st.since : null;
  return (
    <div className="statusbar">
      <span className={`dot ${s.alive ? "on" : "off"}`} />
      <b>{s.alive ? `running (pid ${s.pid})` : "stopped"}</b>
      <span className="muted">{s.alive ? "now:" : "last recorded:"}</span>
      {st.node ? <Link to={`/node/${st.node}`}>{st.node}</Link> : <span>—</span>}
      <span>· {st.phase ?? "—"} · attempt {st.attempt ?? "—"}</span>
      {since !== null && <span className="muted">{s.alive ? `for ${fmtDur(since)}` : `since ${fmtTs(st.since)}`}</span>}
      <span className="sep" />
      <span>nodes <b>{Math.max(s.nodes - 1, 0)}</b> of {s.max_nodes ?? "?"}</span>
      <span>spend <b>{fmtUsd(s.usd)}</b> · {fmtInt(s.tokens)} tokens · {fmtInt(s.calls)} calls</span>
      {s.judge && <span>judge {s.judge}</span>}
      <span>disk free {s.disk_free_gb} GB</span>
      <span className="sep" />
      <span className="muted" title="time of the run's newest event">last event {fmtTs(s.last_event)}</span>
      <span className={connected ? "live on" : "live off"} title="the server pushes every change of the run">{connected ? "● live" : "○ disconnected"}</span>
      <button className="btn small" onClick={() => setPaused(!paused)} title="pause refreshing open views">{paused ? "resume updates" : "pause updates"}</button>
      {updated > 0 && <span className="muted small">updated {new Date(updated).toLocaleTimeString()}</span>}
    </div>
  );
}

function NodeJump() {
  const nodes = useApi<any>("/api/overview");
  const navigate = useNavigate();
  const ids: string[] = (nodes.data?.nodes ?? []).map((n: any) => n.node);
  return (
    <select className="node-jump" value="" onChange={(e) => e.target.value && navigate(`/node/${e.target.value}`)}>
      <option value="">go to node…</option>
      {ids.map((n) => <option key={n} value={n}>{n}</option>)}
    </select>
  );
}

function Shell() {
  const live = useLive();
  const [dark, setDark] = useState(() => localStorage.getItem("theme") !== "light");
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);
  useEffect(() => { if (live.status) document.title = `${live.status.alive ? "● " : ""}${live.status.name} · AR panel`; }, [live.status?.name, live.status?.alive]);
  const nav = [["/", "Overview"], ["/timeline", "Timeline"], ["/compare", "Compare nodes"], ["/trace", "Trace"], ["/problems", "Problems"],
    ["/cost", "Cost"], ["/selection", "Selection"], ["/files", "Files"]];
  return (
    <ThemeContext.Provider value={{ dark, toggle: () => setDark(!dark) }}>
      <LiveContext.Provider value={live}>
        <div className="app">
          <header className="topbar">
            <Link to="/" className="brand">AR panel · <b>{live.status?.name ?? "…"}</b></Link>
            <nav className="mainnav">
              {nav.map(([to, label]) => <NavLink key={to} to={to} end={to === "/"}>{label}</NavLink>)}
            </nav>
            <div className="grow" />
            <NodeJump />
            <button className="btn small" onClick={() => setDark(!dark)}>{dark ? "light" : "dark"} theme</button>
          </header>
          <StatusBar />
          <main>
            <Suspense fallback={<div className="muted pad">Loading view…</div>}>
              <Routes>
                <Route path="/" element={<Overview />} />
                <Route path="/timeline" element={<TimelinePage />} />
                <Route path="/node/:id" element={<NodePage />} />
                <Route path="/node/:id/:tab" element={<NodePage />} />
                <Route path="/compare" element={<Compare />} />
                <Route path="/trace" element={<Trace />} />
                <Route path="/problems" element={<Problems />} />
                <Route path="/cost" element={<Cost />} />
                <Route path="/selection" element={<Selection />} />
                <Route path="/files" element={<Files />} />
                <Route path="*" element={<div className="pad">No such page. <Link to="/">Overview</Link></div>} />
              </Routes>
            </Suspense>
          </main>
        </div>
      </LiveContext.Provider>
    </ThemeContext.Provider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter><Shell /></BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
