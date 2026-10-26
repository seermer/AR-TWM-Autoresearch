import { Timeline } from "../parts/Timeline";
import { Section } from "../ui";

export default function TimelinePage() {
  return (
    <div className="page">
      <Section title="Run timeline" note="every recorded start/end pair and mark of the run · wheel to zoom · click a bar for its event">
        <Timeline height={2000} />
      </Section>
    </div>
  );
}
