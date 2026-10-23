// Stitching panel (bottom left of the viewer): import a stitching JSON, stitch the layout on screen (the server runs the
// stitcher; nothing moves), the seam list with ratio / ease / verdict (click: highlight and zoom), and the exports:
// the stitching JSON, stitching.result.json, and the zip with every stitch image.
import { Vector3 } from "three";
import { importStitching, runStitching, stitchingResultText, stitchingZip } from "../api";
import type { StitchAnswer, StitchLayout } from "../api";
import { seamColour, type StitchView } from "../stitchView";
import { h } from "./dom";

export interface StitchPanelContext {
  /** The layout on screen, or null without a build. */
  layout(): StitchLayout | null;
  view: StitchView;
  focus(centre: Vector3, radius: number): void;
  /** Called when the stitching file or the stitch result changes (the simulation panel follows it). */
  onChange?(): void;
}

/** What the panel offers to the simulation: the request body of the layout with its stitching, once stitched. */
export interface StitchPanelHandle {
  stitched(): (StitchLayout & { stitching: unknown }) | null;
}

function save(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = h("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

const message = (e: unknown): string => {
  const err = (e as { error?: { message?: string } }).error;
  return err?.message ?? (e instanceof Error ? e.message : String(e));
};

export function mountStitchPanel(host: HTMLElement, ctx: StitchPanelContext): StitchPanelHandle {
  const panel = h("div", { class: "stitch-panel", "data-testid": "stitch-panel" });
  host.append(panel);
  let doc: Record<string, unknown> | null = null;
  let answer: StitchAnswer | null = null;
  let selected: string | null = null;
  let busy = "";
  let error = "";

  const fileIn = h("input", { type: "file", accept: ".json,application/json", hidden: true, "data-testid": "import-stitching-input", "aria-label": "Stitching file" });
  fileIn.addEventListener("change", () => {
    const f = fileIn.files?.[0];
    fileIn.value = "";
    if (f) void f.text().then(load);
  });

  const body = (): (StitchLayout & { stitching: unknown }) | null => {
    const l = ctx.layout();
    return l && doc ? { ...l, stitching: doc } : null;
  };
  const task = async (text: string, fn: () => Promise<void>): Promise<void> => {
    busy = text;
    error = "";
    render();
    try {
      await fn();
    } catch (e) {
      error = message(e);
    }
    busy = "";
    render();
    ctx.onChange?.();
  };
  const load = (text: string): Promise<void> =>
    task("reading the stitching file...", async () => {
      doc = (await importStitching(text)).stitching;
      answer = null;
      selected = null;
      ctx.view.setSeams([]);
    });
  const run = (): Promise<void> =>
    task("stitching...", async () => {
      const b = body();
      if (!b) throw new Error("Build a job and place its pieces first.");
      answer = await runStitching(b);
      ctx.view.setSeams(answer.seams, selected);
    });
  const pick = (id: string): void => {
    selected = selected === id ? null : id;
    const s = answer?.seams.find((x) => x.id === id);
    ctx.view.setSeams(answer?.seams ?? [], selected);
    if (s && selected) {
      const pts = [...s.a, ...s.b].map((p) => new Vector3(p[0], p[1], p[2]));
      const c = pts.reduce((acc, p) => acc.add(p), new Vector3()).multiplyScalar(1 / pts.length);
      ctx.focus(c, Math.max(0.12, ...pts.map((p) => p.distanceTo(c))));
    }
    render();
  };

  const render = (): void => {
    const btn = (text: string, id: string, fn: () => void, enabled: boolean, title = ""): HTMLButtonElement =>
      h("button", { class: "btn small", "data-testid": id, disabled: !enabled || busy !== "", title, onclick: fn }, text);
    const garment = String(doc?.garment ?? "garment");
    const sm = answer?.summary;
    const head = h(
      "div",
      { class: "assembly-head" },
      h("strong", {}, doc ? `Stitching: ${garment} (${(doc.seams as unknown[]).length} seams)` : "Stitching"),
      h("span", { class: "spacer" }),
      btn("Import stitching", "import-stitching", () => fileIn.click(), true, "Load a <garment>.stitching.json"),
      btn("Stitch", "run-stitching", () => void run(), doc !== null, "Run the stitcher on the layout on screen (nothing moves)"),
      btn("Export JSON", "export-stitching", () => save(new Blob([JSON.stringify(doc, null, 2) + "\n"], { type: "application/json" }), `${garment}.stitching.json`), doc !== null),
      btn("Export result", "export-stitching-result", () => {
        const b = body();
        if (b) void task("writing stitching.result.json...", async () => save(new Blob([await stitchingResultText(b)], { type: "application/json" }), "stitching.result.json"));
      }, answer !== null),
      btn("Export images", "export-stitching-images", () => {
        const b = body();
        if (b) void task("rendering the stitch images on the server...", async () => save(await stitchingZip(b), `${garment}_stitching.zip`));
      }, answer !== null, "Every view, the close-up of every seam, the coverage map and the result (zip)"),
      fileIn,
    );
    const status = busy || error || (sm
      ? `${answer?.pass ? "PASS" : "FAIL"}: ${sm.seams} seams, ${sm.crossings} crossings, ${sm.unstitched_edges} unstitched edges, ${sm.errors} errors`
      : doc ? "not stitched yet" : "no stitching file");
    const rows = (answer?.seams ?? []).map((s, i) =>
      h(
        "li",
        { class: `seam-row ${s.status} ${selected === s.id ? "selected" : ""}`, "data-testid": `seam-${s.id}`, title: [s.label, ...s.problems].join("\n"), onclick: () => pick(s.id) },
        h("span", { class: "swatch", style: `background:${seamColour(i)}` }),
        `${s.id}  ${s.kind}/${s.ease}${s.ease === "gather" ? ` ${s.declared_ratio}` : ""}  ratio ${s.ratio.toFixed(3)}  ${s.status === "fail" ? "FAIL" : "ok"}`,
      ),
    );
    panel.className = `stitch-panel ${answer ? (answer.pass ? "ok" : "bad") : ""}`;
    panel.replaceChildren(
      head,
      h("div", { class: error ? "warn" : "meta", "data-testid": "stitch-status" }, status),
      ...(answer?.failures.length ? [h("details", {}, h("summary", {}, `Failures (${answer.failures.length})`), h("ul", { class: "problems" }, ...answer.failures.map((f) => h("li", {}, f))))] : []),
      ...(rows.length ? [h("ul", { class: "seam-list", "data-testid": "seam-list" }, ...rows)] : []),
    );
  };
  render();
  return { stitched: () => (answer ? body() : null) };
}
