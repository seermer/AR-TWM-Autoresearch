import { ComponentProps, lazy, Suspense } from "react";
import type { CodeView as CV, DiffView as DV } from "./Monaco";

const LCode = lazy(() => import("./Monaco").then((m) => ({ default: m.CodeView })));
const LDiff = lazy(() => import("./Monaco").then((m) => ({ default: m.DiffView })));
const wait = <div className="muted pad">Loading editor…</div>;

export const Code = (p: ComponentProps<typeof CV>) => <Suspense fallback={wait}><LCode {...p} /></Suspense>;
export const Diff = (p: ComponentProps<typeof DV>) => <Suspense fallback={wait}><LDiff {...p} /></Suspense>;
