import { QueryClient, useQuery } from "@tanstack/react-query";
import { createContext, useContext, useEffect, useRef, useState } from "react";

export const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: Infinity, refetchOnWindowFocus: false, retry: 1 } },
});

export async function get<T = any>(path: string): Promise<T> {
  const r = await fetch(path);
  const body = await r.json().catch(() => ({ error: `${r.status} ${r.statusText}` }));
  const detail = typeof body?.detail === "string" ? body.detail : body?.detail && JSON.stringify(body.detail);
  if (!r.ok) throw new Error(body?.error || detail || `${r.status} ${r.statusText}`);
  return body as T;
}

/** A query on the panel API, refetched in place whenever the run changes (see useLive). */
export function useApi<T = any>(path: string | null) {
  return useQuery<T>({ queryKey: [path], queryFn: () => get<T>(path!), enabled: !!path });
}

export const q = (params: Record<string, any>) =>
  new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => [k, String(v)])).toString();

export const raw = (rel: string) => "/raw/" + rel.split("/").map(encodeURIComponent).join("/");

export type Status = {
  name: string; alive: boolean; pid: number | null; state: any; max_nodes: number | null; nodes: number;
  judge: string | null; calls: number; tokens: number; usd: number | null; disk_free_gb: number;
  events: number; last_event: number | null; last_seq: number | null; now: number; config: any;
};

export type Live = { status: Status | null; connected: boolean; paused: boolean; setPaused: (p: boolean) => void; updated: number };
export const LiveContext = createContext<Live>({ status: null, connected: false, paused: false, setPaused: () => {}, updated: 0 });
export const useLiveStatus = () => useContext(LiveContext);

/** Follows /api/stream; on every change of the run, refetches the open views (at most every 5 s) unless paused. */
export function useLive(): Live {
  const [status, setStatus] = useState<Status | null>(null);
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const [updated, setUpdated] = useState(0);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;
  const last = useRef<string | null>(null);
  const pending = useRef<number | null>(null);
  const lastRefresh = useRef(0);
  useEffect(() => {
    const source = new EventSource("/api/stream");
    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    source.onmessage = (m) => {
      const s: Status = JSON.parse(m.data);
      setStatus(s);
      const key = JSON.stringify([s.last_seq, s.alive, s.state, s.nodes]);
      if (last.current === null) { last.current = key; return; }
      if (key === last.current) return;
      last.current = key;
      const refresh = () => {
        pending.current = null;
        if (pausedRef.current) return;
        lastRefresh.current = Date.now();
        queryClient.invalidateQueries();
        setUpdated(Date.now());
      };
      const wait = Math.max(0, 5000 - (Date.now() - lastRefresh.current));
      if (pending.current === null) pending.current = window.setTimeout(refresh, wait);
    };
    return () => source.close();
  }, []);
  useEffect(() => { if (!paused) queryClient.invalidateQueries(); }, [paused]);
  return { status, connected, paused, setPaused, updated };
}
