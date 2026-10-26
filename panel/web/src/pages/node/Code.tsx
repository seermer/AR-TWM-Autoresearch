import { useState } from "react";
import { q, useApi } from "../../api";
import { Code, Diff } from "../../parts/lazy";
import { Badge, Chips, Empty, Json, KV, Q, Section, Text, useParam } from "../../ui";

const STATUS: Record<string, string> = { A: "added", M: "modified", D: "deleted", R: "renamed", C: "copied", T: "type changed" };

function FileDiff({ base, head, file }: { base: string; head: string; file: any }) {
  const a = useApi(file.status[0] === "A" ? null : `/api/agent/file?${q({ commit: base, path: file.old_path })}`);
  const b = useApi(file.status[0] === "D" ? null : `/api/agent/file?${q({ commit: head, path: file.path })}`);
  if (a.isLoading || b.isLoading) return <div className="muted pad">Loading file…</div>;
  return <Diff original={a.data?.text ?? ""} modified={b.data?.text ?? ""} path={file.path} height={600} />;
}

/** A commit range: the changed files on the left, the selected file side by side on the right. */
function Changes({ base, head, files, unified }: { base: string; head: string; files: any[]; unified: string | null }) {
  const [sel, setSel] = useState(0);
  if (!files.length) return <div className="muted">no changes</div>;
  const f = files[Math.min(sel, files.length - 1)];
  return (
    <div>
      <div className="code-layout">
        <div className="file-list">
          {files.map((x, i) => (
            <button key={x.path} className={i === sel ? "file active" : "file"} onClick={() => setSel(i)} title={x.path}>
              <span className={`fstat s-${x.status[0]}`}>{x.status[0]}</span> {x.path}{x.status[0] === "R" ? ` ← ${x.old_path}` : ""}
            </button>
          ))}
          <div className="muted small pad">{files.length} files · {base.slice(0, 8)} → {head.slice(0, 8)}</div>
        </div>
        <div className="grow-min"><div className="muted small">{STATUS[f.status[0]] ?? f.status}: {f.path}</div><FileDiff key={f.path + base + head} base={base} head={head} file={f} /></div>
      </div>
      <Section title="unified diff (git diff)" collapsed><Text text={unified} lines={400} /></Section>
    </div>
  );
}

function Browse({ commits }: { commits: any[] }) {
  const [commit, setCommit] = useState(commits[0]?.commit ?? "");
  const [path, setPath] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const tree = useApi<string[]>(commit ? `/api/agent/tree?${q({ commit })}` : null);
  const file = useApi(commit && path ? `/api/agent/file?${q({ commit, path })}` : null);
  return (
    <div>
      <div className="row gap">
        <select value={commit} onChange={(e) => { setCommit(e.target.value); }}>
          {commits.map((c) => <option key={c.commit} value={c.commit}>{c.label}</option>)}
        </select>
        <span className="mono small">{commit}</span>
      </div>
      <div className="code-layout">
        <div className="file-list">
          <input className="search" placeholder="filter files…" value={filter} onChange={(e) => setFilter(e.target.value)} />
          <Q query={tree}>{(paths) => <>{paths.filter((p) => p.toLowerCase().includes(filter.toLowerCase())).map((p) => (
            <button key={p} className={p === path ? "file active" : "file"} onClick={() => setPath(p)} title={p}>{p}</button>
          ))}<div className="muted small pad">{paths.length} files</div></>}</Q>
        </div>
        <div className="grow-min">{path ? <Q query={file}>{(f) => f.text === null ? <div className="muted">not in this commit</div> : <Code text={f.text} path={path} height={640} />}</Q> : <div className="muted pad">Pick a file.</div>}</div>
      </div>
    </div>
  );
}

export default function CodeTab({ id }: { id: string }) {
  const query = useApi(`/api/code/${encodeURIComponent(id)}`);
  const [att, setAtt] = useParam("edit", "");
  return (
    <Q query={query} what="code edits">{(d) => {
      const attempt = d.attempts.find((a: any) => String(a.attempt) === att) ?? d.attempts[d.attempts.length - 1];
      return (
        <div>
          {d.baseline ? <div className="note">Baseline: the root runs the seed agent unedited.</div> : (
            <Section title="Node diff: parent's agent code → this node's" note={d.parent_commit && d.agent_commit ? `${d.parent_commit.slice(0, 12)} → ${d.agent_commit.slice(0, 12)}` : ""}>
              {d.parent_commit && d.agent_commit ? <Changes base={d.parent_commit} head={d.agent_commit} files={d.node_files} unified={d.node_diff} /> : <div className="muted">no agent commit yet</div>}
            </Section>
          )}
          {d.attempts.length > 0 && (
            <Section title={`edit_self attempts (${d.attempts.length})`}>
              <Chips items={d.attempts.map((a: any) => String(a.attempt))} value={String(attempt.attempt)} onChange={setAtt}
                render={(k) => { const a = d.attempts.find((x: any) => String(x.attempt) === k); return <>attempt {k} · <Badge>{a.outcome}</Badge></>; }} />
              <KV items={[["outcome", <Badge>{attempt.outcome}</Badge>], ["commit", attempt.commit ? <span className="mono">{attempt.commit}</span> : "no commit (checkout or commit failed)"],
                ["base", attempt.base && <span className="mono">{attempt.base}</span>], ["detail", attempt.detail && <Text text={attempt.detail} lines={20} />]]} />
              {attempt.commit && attempt.base && <Changes key={attempt.commit} base={attempt.base} head={attempt.commit} files={attempt.files} unified={attempt.diff} />}
              <Section title="Contract report">{attempt.contract ? <Json value={attempt.contract} open={3} /> : <div className="muted">no contract check: the attempt failed before it</div>}</Section>
            </Section>
          )}
          <Section title="Browse agent code">{d.commits.length ? <Browse commits={d.commits} /> : <Empty>No agent commit.</Empty>}</Section>
        </div>
      );
    }}</Q>
  );
}
