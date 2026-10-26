import { useMemo, useState } from "react";
import { q, useApi } from "../../api";
import { ChatView } from "../../parts/Chat";
import { Dag } from "../../parts/Dag";
import { Chips, colorOf, Drawer, Empty, fmtDur, fmtInt, fmtUsd, Q, Section, Table, Text, useParam } from "../../ui";

function TaskGraph({ graph, selected, onTask }: { graph: any; selected: number | null; onTask: (t: number) => void }) {
  const roles = useMemo(() => [...new Set<string>(graph.tasks.map((t: any) => t.role))], [graph]);
  const dag = useMemo(() => ({
    nodes: graph.tasks.map((t: any) => ({
      id: String(t.index), width: 230, height: 70, style: { borderLeft: `4px solid ${colorOf(t.role, roles)}` },
      content: (
        <div>
          <div className="row gap"><b>{t.role}</b><span className="muted small">{t.of_role}/{t.role_tasks}</span><span className="grow" /><span className="muted small">task {t.index}</span></div>
          <div className="small">{t.started.slice(11)} → {t.ended.slice(11)} · {fmtDur(t.end - t.start)}</div>
          <div className="small muted">{fmtInt(t.calls)} calls · conv #{t.conversation} turns {t.turns[0] ?? "?"}–{t.turns[1] ?? "compaction"} · {fmtInt(t.tokens)} tok{t.usd !== null ? ` · ${fmtUsd(t.usd)}` : ""}</div>
          {t.alongside.length > 0 && <div className="small warn">began alongside task {t.alongside.join(", ")}</div>}
        </div>
      ),
    })),
    edges: graph.edges.map((e: any, i: number) => ({
      id: `e${i}`, source: String(e.source), target: String(e.target), dashed: graph.tasks[e.target].alongside.length > 0,
      label: e.tools.length ? `after ${e.tools.join(", ")}` : undefined,
    })),
  }), [graph, roles]);
  return (
    <div>
      <Dag nodes={dag.nodes} edges={dag.edges} natural ranksep={44} vertical
        selected={selected !== null ? String(selected) : null} onNode={(id) => onTask(+id)} />
      <div className="muted small">Inferred from timing alone, whatever the orchestration. One node per task: a conversation's calls stay one task until it resumes after another conversation made a whole call while it was idle. Each task follows the task whose call ended last before it (the label: the tools that call asked for); dashed: it began while another task still had a call in flight. Click a task for the context its model really had.</div>
    </div>
  );
}

function ToolLog({ path, onClose }: { path: string | null; onClose: () => void }) {
  const query = useApi(path ? `/api/preview?${q({ path })}` : null);
  return (
    <Drawer open={!!path} onClose={onClose} title={path ?? ""} wide>
      <Q query={query}>{(p) => <Text text={p.text ?? `(${p.kind}, ${p.bytes} bytes)`} lines={400} />}</Q>
    </Drawer>
  );
}

export default function Agents({ id, attempts }: { id: string; attempts: any[] }) {
  const keys = attempts.map((a) => `${a.phase}-${a.attempt}`);
  const [pa, setPa] = useParam("pa", keys[keys.length - 1] ?? "");
  const [convParam, setConv] = useParam("conv", "");
  const [turnParam, setTurn] = useParam("turn", "");
  const [taskParam, setTask] = useParam("task", "");
  const [log, setLog] = useState<string | null>(null);
  const cut = pa.lastIndexOf("-");
  const phase = pa.slice(0, cut), attempt = pa.slice(cut + 1);
  const query = useApi(pa ? `/api/agents/${encodeURIComponent(id)}/${encodeURIComponent(phase)}/${attempt}` : null);
  if (!keys.length) return <Empty>No agent conversations recorded for {id}.</Empty>;
  return (
    <div>
      <div className="row gap wrap">
        <span className="muted">phase-attempt:</span>
        <Chips items={keys} value={pa} onChange={(k) => { setPa(k); setConv(null); setTurn(null); setTask(null); }}
          render={(k) => { const a = attempts[keys.indexOf(k)]; return `${k} (${a.conversations.length})`; }} />
      </div>
      <Q query={query} what="agents">{(d) => {
        const convs = d.conversations;
        const task = taskParam !== "" ? +taskParam : convParam === "" && d.graph.tasks.length ? 0 : null;
        const conv = task === null && convParam !== "" ? +convParam : null;
        const open = (i: number, turn?: number | null) => { setConv(String(i)); setTurn(turn ? String(turn) : null); setTask(null); };
        const openTask = (t: number) => { const first = d.graph.tasks[t]?.turns[0]; setTask(String(t)); setConv(null); setTurn(first ? String(first) : null); };
        return (
          <div className="agents-layout">
            <div className="agents-left">
              <Section title={`Delegation tasks (${d.graph.tasks.length}, inferred from timing)`}><TaskGraph graph={d.graph} selected={task} onTask={openTask} /></Section>
              <Section title={`Conversations (${convs.length})`} note="role = the agent's own label · the whole stitched history · compactions and tool errors inferred">
                <Table rows={convs} rowKey={(r: any) => String(r.index)} selected={conv !== null ? String(conv) : null} onRowClick={(r: any) => open(r.index)} filter={false} cols={[
                  { key: "index", label: "#", width: "40px" }, { key: "role", width: "minmax(110px, 1fr)" }, { key: "started", width: "140px" },
                  { key: "calls", width: "60px" }, { key: "compactions", label: "compact.", width: "70px" }, { key: "tool_errors", label: "tool err.", width: "70px" },
                  { key: "tokens", width: "100px" }, { key: "usd", width: "70px" }]} />
              </Section>
              <Section title={`Full tool outputs (${d.tool_logs.length})`} note="tool_output/ of this attempt's workspace" collapsed={!d.tool_logs.length}>
                <Table rows={d.tool_logs.map((p: string) => ({ path: p, name: p.split("/").pop() }))} onRowClick={(r: any) => setLog(r.path)} cols={[{ key: "name", mono: true }]} maxHeight={300} />
              </Section>
            </div>
            <div className="agents-right">
              {task !== null && d.graph.tasks[task] ? <TaskPanel key={`${pa}-t${task}`} id={id} phase={phase} attempt={attempt} task={d.graph.tasks[task]} turn={turnParam ? +turnParam : null} onTurn={(t) => setTurn(String(t))} />
                : conv === null ? <Empty>No task or conversation selected.</Empty> : <ConvPanel key={`${pa}-${conv}`} id={id} phase={phase} attempt={attempt} conv={convs[conv]} turn={turnParam ? +turnParam : null} onTurn={(t) => setTurn(String(t))} />}
            </div>
            <ToolLog path={log} onClose={() => setLog(null)} />
          </div>
        );
      }}</Q>
    </div>
  );
}

function ConvPanel({ id, phase, attempt, conv, turn, onTurn }: { id: string; phase: string; attempt: string; conv: any; turn: number | null; onTurn: (t: number) => void }) {
  const query = useApi(conv ? `/api/chat/${encodeURIComponent(id)}/${encodeURIComponent(phase)}/${attempt}/${conv.index}` : null);
  if (!conv) return <Empty>No such conversation.</Empty>;
  return (
    <Q query={query} what="conversation">{(items) => (
      <ChatView items={items} turn={turn} onTurn={onTurn} header={
        <div className="conv-head">
          <b>#{conv.index} {conv.role}</b> · started {conv.started} · {conv.calls} calls · {fmtInt(conv.tokens)} tokens{conv.usd !== null ? ` · ${fmtUsd(conv.usd)}` : ""}
          {" · "}{conv.compactions} compactions (inferred) · {conv.tool_errors} tool errors (inferred)
          <span className="muted small"> · conversation ids {conv.conversation_ids.join(" → ")}</span>
        </div>
      } />
    )}</Q>
  );
}

function TaskPanel({ id, phase, attempt, task, turn, onTurn }: { id: string; phase: string; attempt: string; task: any; turn: number | null; onTurn: (t: number) => void }) {
  const query = useApi(`/api/task/${encodeURIComponent(id)}/${encodeURIComponent(phase)}/${attempt}/${task.index}`);
  return (
    <Q query={query} what="task context">{(items) => (
      <ChatView items={items} turn={turn} onTurn={onTurn} header={
        <div className="conv-head">
          <b>task {task.index}: {task.role}</b> ({task.of_role} of {task.role_tasks}) · {task.started} → {task.ended} ({fmtDur(task.end - task.start)}) · {task.calls} calls · {fmtInt(task.tokens)} tokens{task.usd !== null ? ` · ${fmtUsd(task.usd)}` : ""}
          <div className="muted small">The context its model really had: the task's last request exactly as sent, then the reply. "earlier context" = messages already there when the task began (folded); conversation #{task.conversation} has the whole stitched history.</div>
        </div>
      } />
    )}</Q>
  );
}
