import { Link } from "react-router-dom";
import { useApi } from "../api";
import { Drawer, fmtTs, Json, KV, Q, Section, Text } from "../ui";
import { ChatItem } from "./Chat";

const TEXT_KEYS = ["stdout", "stderr", "error", "traceback", "detail", "output", "result_text"];

/** One event in full: its fields, its payload as a chat (model calls), its long texts, and both records raw. */
export function EventDetail({ seq }: { seq: number }) {
  const query = useApi(`/api/event/${seq}`);
  return (
    <Q query={query} what="event">{(d) => {
      const e = d.event, p = d.payload;
      const llm = e.type?.startsWith("llm.") && e.node && e.phase;
      const texts = p && typeof p === "object" ? TEXT_KEYS.filter((k) => typeof p[k] === "string" && p[k]) : [];
      return (
        <div>
          <KV items={[["type", <b>{e.type}</b>], ["time", fmtTs(e.ts_wall)], ["node", e.node && e.node !== "run" ? <Link to={`/node/${e.node}`}>{e.node}</Link> : e.node],
            ["phase", e.phase], ["attempt", e.attempt], ["component", e.component], ["seq", e._seq ?? seq], ["span", e.span_id], ["parent span", e.parent_span_id],
            ["payload digest", e.payload]]} />
          <div className="row gap">
            {llm && <Link className="btn small" to={`/node/${e.node}/agents?${new URLSearchParams({ pa: `${e.phase}-${e.attempt}` })}`}>open this attempt's agents</Link>}
            {e.node && <Link className="btn small" to={`/trace?${new URLSearchParams({ node: e.node, phase: e.phase ?? "", attempt: String(e.attempt ?? "") })}`}>events of this phase-attempt</Link>}
          </div>
          {p && Array.isArray(p.args) && <Section title="command"><Text text={p.args.join(" ")} lines={5} /></Section>}
          {texts.map((k) => <Section key={k} title={k}><Text text={p[k]} lines={60} /></Section>)}
          {d.chat?.length > 0 && (
            <Section title={`as a chat (${d.chat.length} items)`}>
              <div className="chat-static">{d.chat.map((it: any, i: number) => <ChatItem key={i} it={it} />)}</div>
            </Section>
          )}
          <Section title="event record"><Json value={e} open={1} /></Section>
          <Section title="payload">{p === null || p === undefined ? <div className="muted">no payload</div> : <Json value={p} open={2} />}</Section>
        </div>
      );
    }}</Q>
  );
}

export function EventDrawer({ seq, onClose }: { seq: number | null; onClose: () => void }) {
  return (
    <Drawer open={seq !== null} onClose={onClose} title={`event ${seq}`} wide>
      {seq !== null && <EventDetail seq={seq} />}
    </Drawer>
  );
}
