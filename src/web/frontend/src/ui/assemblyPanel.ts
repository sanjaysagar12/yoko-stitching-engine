// Panel over the viewer after an import: "Assembly: <garment> (job, N pieces, M derived) - valid / N problems", the notes
// and expectations of the file (read-only), import warnings, and the buttons Re-assemble from file and Revert.
import type { Store } from "../state";
import type { AssemblyExpect } from "../types";
import type { Actions } from "./actions";
import { h } from "./dom";

/** One line per thing the file declared it expects (read-only). */
export function expectLines(e: AssemblyExpect): string[] {
  const out: string[] = [];
  if (e.required_pieces?.length) out.push(`required pieces (${e.required_pieces.length}): ${e.required_pieces.join(", ")}`);
  for (const r of e.registration ?? []) {
    const where = r.piece_point ? ` at ${r.piece_point}` : r.edge ? ` (${r.edge} edge)` : "";
    const lift = r.offset_m !== undefined ? `, ${(r.offset_m * 100).toFixed(1)} cm above` : "";
    out.push(`${r.piece} registered on ${r.level}${where}${lift}`);
  }
  for (const r of e.rings ?? []) out.push(`ring ${r.name}: ${r.pieces.join(", ")}`);
  if (e.sections?.length) out.push(`sections at ${e.sections.join(", ")}`);
  if (e.far_ok?.length) out.push(`allowed to stand off: ${e.far_ok.join(", ")}`);
  const grain = Object.entries(e.grain ?? {});
  if (grain.length) out.push(`grain: ${grain.map(([k, v]) => `${k} ${v}`).join(", ")}`);
  return out;
}

export function mountAssemblyPanel(host: HTMLElement, store: Store, actions: Actions): void {
  const panel = h("div", { class: "assembly-banner", "data-testid": "assembly-banner", hidden: true });
  host.append(panel);

  const render = (): void => {
    const s = store.get();
    const a = s.assembly;
    if (!a || a.dismissed) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    const p = a.payload;
    const v = s.lastAlign?.validity;
    const verdict = s.busy ? "assembling..." : !v ? "not assembled" : v.valid ? "valid" : `${v.violations.length} problem${v.violations.length === 1 ? "" : "s"}`;
    panel.className = `assembly-banner ${v && !s.busy ? (v.valid ? "ok" : "bad") : ""}`;
    const lines = expectLines(p.expect);
    panel.replaceChildren(
      h(
        "div",
        { class: "assembly-head" },
        h("strong", { "data-testid": "assembly-summary" }, `Assembly: ${p.garment} (${p.job}, ${p.placements.length} pieces, ${p.derived_pieces.length} derived) - ${verdict}`),
        h("span", { class: "spacer" }),
        h("button", { class: "btn small", "data-testid": "assembly-reassemble", disabled: s.busy !== null, title: "Reset to the imported file and assemble again", onclick: () => actions.reassemble() }, "Re-assemble from file"),
        h("button", { class: "btn small", "data-testid": "assembly-revert", disabled: s.busy !== null || !a.before?.job, title: "Back to the layout before the import", onclick: () => actions.revertImport() }, "Revert"),
        h("button", { class: "btn small", "data-testid": "assembly-dismiss", "aria-label": "Dismiss", onclick: () => actions.dismissAssembly() }, "x"),
      ),
      ...(p.warnings.length ? [h("ul", { class: "problems", "data-testid": "assembly-warnings" }, ...p.warnings.map((w) => h("li", {}, w)))] : []),
      ...(p.notes ? [h("details", {}, h("summary", {}, "Notes"), h("p", { class: "meta", "data-testid": "assembly-notes" }, p.notes))] : []),
      ...(lines.length ? [h("details", {}, h("summary", {}, `Expect (${lines.length})`), h("ul", { class: "problems", "data-testid": "assembly-expect" }, ...lines.map((l) => h("li", {}, l))))] : []),
    );
  };
  store.subscribe((_, changed) => {
    if (["assembly", "lastAlign", "busy"].some((k) => changed.includes(k as never))) render();
  });
  render();
}
