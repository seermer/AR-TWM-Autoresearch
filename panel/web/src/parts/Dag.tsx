import dagre from "@dagrejs/dagre";
import { Background, Controls, Edge, Handle, MarkerType, Node, Position, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { ReactNode, useMemo } from "react";

export type DagNode = { id: string; content: ReactNode; width?: number; height?: number; className?: string; style?: any };
export type DagEdge = { id: string; source: string; target: string; label?: ReactNode; dashed?: boolean; color?: string; width?: number; data?: any };

function Box({ data }: { data: any }) {
  return (
    <div className={`dag-node ${data.className ?? ""}`} style={{ width: data.width, minHeight: data.height, ...data.style }}>
      <Handle type="target" position={data.vertical ? Position.Top : Position.Left} />
      {data.content}
      <Handle type="source" position={data.vertical ? Position.Bottom : Position.Right} />
    </div>
  );
}
const nodeTypes = { box: Box };

/** A directed graph laid out by dagre; clicks on nodes and edges go to the callers. `natural`: shown at zoom 1 at its
 * own height, and the wheel scrolls the page. */
export function Dag({ nodes, edges, height = 420, vertical = true, natural = false, ranksep = 90, onNode, onEdge, selected }: {
  nodes: DagNode[]; edges: DagEdge[]; height?: number; vertical?: boolean; natural?: boolean; ranksep?: number; selected?: string | null;
  onNode?: (id: string) => void; onEdge?: (e: DagEdge) => void;
}) {
  const laid = useMemo(() => {
    const g = new dagre.graphlib.Graph({ multigraph: true });
    g.setGraph({ rankdir: vertical ? "TB" : "LR", nodesep: 60, ranksep, edgesep: 30 });
    g.setDefaultEdgeLabel(() => ({}));
    for (const n of nodes) g.setNode(n.id, { width: n.width ?? 170, height: n.height ?? 70 });
    for (const e of edges) g.setEdge(e.source, e.target, { width: 120, height: 30 }, e.id);
    dagre.layout(g);
    const rfNodes: Node[] = nodes.map((n) => {
      const p = g.node(n.id);
      return {
        id: n.id, type: "box", position: { x: p.x - (n.width ?? 170) / 2, y: p.y - (n.height ?? 70) / 2 },
        data: { content: n.content, width: n.width ?? 170, height: n.height ?? 70, vertical,
                className: `${n.className ?? ""} ${selected === n.id ? "selected" : ""}`, style: n.style },
      };
    });
    const rfEdges: Edge[] = edges.map((e) => ({
      id: e.id, source: e.source, target: e.target, label: e.label as any, data: e.data,
      style: { stroke: e.color ?? "var(--edge)", strokeWidth: e.width ?? 1.5, strokeDasharray: e.dashed ? "6 4" : undefined },
      markerEnd: { type: MarkerType.ArrowClosed, color: e.color ?? "var(--edge)" },
      labelBgStyle: { fill: "var(--panel)" }, labelStyle: { fill: "var(--fg)", fontSize: 11 },
    }));
    return { rfNodes, rfEdges, height: (g.graph().height ?? 0) + 40 };
  }, [nodes, edges, vertical, selected, ranksep]);
  return (
    <div className="dag" style={{ height: natural ? laid.height : height }}>
      <ReactFlow nodes={laid.rfNodes} edges={laid.rfEdges} nodeTypes={nodeTypes} fitView minZoom={0.1} maxZoom={2}
        fitViewOptions={natural ? { minZoom: 1, maxZoom: 1 } : undefined} zoomOnScroll={!natural} preventScrolling={!natural}
        nodesDraggable={false} nodesConnectable={false} proOptions={{ hideAttribution: true }}
        onNodeClick={(_, n) => onNode?.(n.id)} onEdgeClick={(_, e) => onEdge?.(edges.find((x) => x.id === e.id)!)}>
        <Background gap={20} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
