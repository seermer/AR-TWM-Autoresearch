import Editor, { DiffEditor, loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor";
import EditorWorker from "monaco-editor/editor/editor.worker?worker";
import CssWorker from "monaco-editor/language/css/css.worker?worker";
import HtmlWorker from "monaco-editor/language/html/html.worker?worker";
import JsonWorker from "monaco-editor/language/json/json.worker?worker";
import TsWorker from "monaco-editor/language/typescript/ts.worker?worker";
import { useEffect, useRef, useState } from "react";
import { useTheme } from "../ui";

const WORKERS: Record<string, new () => Worker> = { json: JsonWorker, css: CssWorker, scss: CssWorker, less: CssWorker, html: HtmlWorker,
  handlebars: HtmlWorker, razor: HtmlWorker, typescript: TsWorker, javascript: TsWorker };
(self as any).MonacoEnvironment = { getWorker: (_: string, label: string) => new (WORKERS[label] ?? EditorWorker)() };
loader.config({ monaco });

const LANG: Record<string, string> = {
  py: "python", js: "javascript", ts: "typescript", tsx: "typescript", json: "json", jsonl: "json", yaml: "yaml", yml: "yaml",
  md: "markdown", sh: "shell", toml: "ini", cfg: "ini", ini: "ini", html: "html", css: "css", txt: "plaintext", log: "plaintext", diff: "diff",
};
export const langOf = (path: string) => LANG[path.split(".").pop()?.toLowerCase() ?? ""] ?? "plaintext";

const OPTIONS = { readOnly: true, minimap: { enabled: true }, scrollBeyondLastLine: false, fontSize: 12, wordWrap: "on" as const, renderWhitespace: "boundary" as const };

/** Read-only code with highlighting, search (Ctrl+F) and folding. */
export function CodeView({ text, path, height = 520 }: { text: string; path: string; height?: number | string }) {
  const { dark } = useTheme();
  return <div className="monaco"><Editor height={height} value={text} language={langOf(path)} theme={dark ? "vs-dark" : "vs"} options={OPTIONS} /></div>;
}

/** Side-by-side (or inline) diff of two texts. */
export function DiffView({ original, modified, path, height = 520 }: { original: string; modified: string; path: string; height?: number | string }) {
  const { dark } = useTheme();
  const [inline, setInline] = useState(false);
  const editor = useRef<monaco.editor.IStandaloneDiffEditor | null>(null);
  useEffect(() => () => {                     // detach, then dispose: the other order makes Monaco throw
    const models = editor.current?.getModel();
    editor.current?.setModel(null);
    models?.original.dispose();
    models?.modified.dispose();
  }, []);
  return (
    <div className="monaco">
      <div className="text-tools"><span className="muted small">{path}</span><button className="btn small" onClick={() => setInline(!inline)}>{inline ? "side by side" : "inline"}</button></div>
      <DiffEditor height={height} original={original} modified={modified} language={langOf(path)} theme={dark ? "vs-dark" : "vs"}
        options={{ ...OPTIONS, renderSideBySide: !inline, originalEditable: false, ignoreTrimWhitespace: false }}
        keepCurrentOriginalModel keepCurrentModifiedModel onMount={(e) => { editor.current = e; }} />
    </div>
  );
}

export default CodeView;
