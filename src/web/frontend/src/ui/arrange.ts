// Left panel: Auto-arrange. Options, the Arrange and Reset buttons, a progress bar, the result tables (true ease per ring,
// violations, notices) and the picked edge or point. Everything shown is a number from /api/arrange.
import { formatCm, formatMm, nearestPiece, splitWarning } from "../lib";
import type { Store } from "../state";
import type { Actions } from "./actions";
import { h } from "./dom";

interface Slider {
  key: "clearance_body" | "gap_piece" | "gap_layer" | "max_standoff";
  label: string;
  min: number; // cm
  max: number; // cm
  step: number; // cm
  testid: string;
}

const SLIDERS: Slider[] = [
  { key: "clearance_body", label: "Clearance to the body (cm)", min: 0, max: 8, step: 0.5, testid: "arr-clearance" },
  { key: "gap_piece", label: "Gap between pieces (cm)", min: 0, max: 10, step: 0.5, testid: "arr-gap-piece" },
  { key: "gap_layer", label: "Gap between layers (cm)", min: 0, max: 5, step: 0.5, testid: "arr-gap-layer" },
  { key: "max_standoff", label: "Largest standoff (cm)", min: 3, max: 40, step: 1, testid: "arr-max-standoff" },
];

export function mountArrange(host: HTMLElement, store: Store, actions: Actions): void {
  const box = h("div", { class: "arrange", "data-testid": "arrange" });
  host.append(h("h2", {}, "Auto-arrange"), box);

  // controls are built once (so that a slider keeps focus while it is dragged); the results are redrawn
  const controls = h("div", { class: "arrange-controls" });
  const readouts = new Map<string, HTMLElement>();
  for (const sl of SLIDERS) {
    const out = h("span", { class: "readout", "data-testid": `${sl.testid}-value` }, "");
    readouts.set(sl.key, out);
    const input = h("input", {
      type: "range",
      min: sl.min,
      max: sl.max,
      step: sl.step,
      value: store.get().arrangeOptions[sl.key] * 100,
      "data-testid": sl.testid,
    });
    input.addEventListener("input", () => {
      out.textContent = Number(input.value).toFixed(1);
    });
    input.addEventListener("change", () => actions.setArrangeOptions({ [sl.key]: Number(input.value) / 100 }));
    controls.append(h("label", { class: "arr-row" }, h("span", {}, sl.label), input, out));
  }
  const mesh = h("input", { type: "checkbox", "data-testid": "arr-mesh-lines" });
  mesh.addEventListener("change", () => actions.setMeshLines(mesh.checked));
  controls.append(h("label", { class: "arr-row" }, h("span", {}, "Show mesh lines"), mesh));
  const run = h("button", { class: "btn primary", "data-testid": "arrange-run", onclick: () => actions.arrange() }, "Auto-arrange");
  const reset = h("button", { class: "btn", "data-testid": "arrange-reset", onclick: () => actions.resetArrange() }, "Place again from anchors");
  const bar = h("div", { class: "progress arr-progress", "data-testid": "arrange-progress" }, h("div", { class: "progress-fill" }));
  const status = h("div", { class: "meta", "data-testid": "arrange-status" }, "");
  controls.append(h("div", { class: "btn-row" }, run, reset), bar, status);
  const honesty = h(
    "p",
    { class: "hint", "data-testid": "arrange-honesty" },
    "Every placement is checked: pieces keep their exact size (rolled, never stretched), stay clear of the body and of each other. Auto-arrange spreads all pieces again. The standoff is display space, never ease; ease is the garment loop minus the skin loop.",
  );
  const results = h("div", { class: "arrange-results", "data-testid": "arrange-results" });
  const hover = h("div", { class: "meta", "data-testid": "hover-text" });
  box.append(controls, honesty, results, hover);

  const renderControls = (): void => {
    const s = store.get();
    for (const sl of SLIDERS) {
      const out = readouts.get(sl.key);
      if (out) out.textContent = (s.arrangeOptions[sl.key] * 100).toFixed(1);
    }
    mesh.checked = s.showMeshLines;
    const placed = s.placements.size;
    run.disabled = s.busy !== null || placed === 0;
    run.title = placed === 0 ? "Place pieces on arrangement points first" : "";
    reset.disabled = placed === 0;
    bar.classList.toggle("active", s.busy?.text.startsWith("arranging") ?? false);
    (bar.firstElementChild as HTMLElement).style.width = s.busy?.text.startsWith("arranging") ? "100%" : "0";
    status.textContent = s.busy?.text.startsWith("arranging") ? s.busy.text : "";
  };

  const renderResults = (): void => {
    const s = store.get();
    const a = s.arrangement;
    if (!a) {
      results.replaceChildren(h("p", { class: "hint" }, "Place pieces, then press Auto-arrange."));
      return;
    }
    const placed = [...s.placed.values()].filter((p) => p.arrange);
    const minAvatar = Math.min(...placed.map((p) => p.arrange?.min_distance_to_avatar_m ?? Infinity));
    const minPiece = Math.min(...placed.flatMap((p) => Object.values(p.arrange?.min_distance_to_pieces_m ?? {})));
    const summary = h(
      "div",
      { class: "meta", "data-testid": "arrange-summary" },
      `closest to the body ${formatMm(minAvatar)} mm, closest pieces ${formatMm(minPiece)} mm, solved in ${a.timing_s.toFixed(1)} s`,
    );
    const rings = h(
      "table",
      { class: "mtable small", "data-testid": "arrange-rings" },
      h("thead", {}, h("tr", {}, ...["ring", "body", "garment", "true ease", "standoff", "row"].map((t) => h("th", {}, t)))),
      h(
        "tbody",
        {},
        ...a.rings.map((r) =>
          h(
            "tr",
            { "data-testid": `ring-${r.ring}`, class: r.exceeded ? "bad" : "", title: r.pieces.join(", ") },
            h("td", {}, r.ring),
            h("td", {}, formatCm(r.body_loop_m * 100, 1)),
            h("td", {}, formatCm(r.garment_loop_m * 100, 1)),
            h("td", { "data-testid": `ease-${r.ring}` }, formatCm(r.true_ease_m * 100, 1)),
            h("td", { "data-testid": `standoff-${r.ring}` }, formatCm(r.standoff_used_m * 100, 1) + (r.exceeded ? ` / ${formatCm(r.standoff_required_m * 100, 1)} needed` : "")),
            h("td", {}, r.binding_row_m === null ? "-" : formatCm(r.binding_row_m * 100, 0)),
          ),
        ),
      ),
    );
    const ringNote = h("div", { class: "meta" }, "cm. body = skin loop, garment = sum of piece widths at the binding row, true ease = garment - body. Standoff is display space.");
    const vlist = s.lastAlign?.validity?.violations ?? a.violations;
    const viol = h(
      "div",
      { class: "violations", "data-testid": "arrange-violations" },
      h("h3", {}, `Violations (${vlist.length})`),
      ...(vlist.length === 0
        ? [h("div", { class: "meta" }, "none: every piece keeps its size, the clearance and the gaps")]
        : vlist.map((v, i) =>
            h(
              "button",
              { class: "btn violation", "data-testid": `violation-${i}`, onclick: () => actions.focusViolation(i) },
              h("strong", {}, v.kind),
              ` ${v.pieces.join(" / ")}: ${formatMm(v.value_m)} mm (needs ${formatMm(v.required_m)} mm)`,
            ),
          )),
    );
    const notices = [...a.notes, ...(s.lastAlign?.warnings ?? [])].map((m) => {
      const { code, text } = splitWarning(m);
      return h("div", { class: `notice ${code}`, "data-testid": `notice-${code || "info"}` }, h("strong", {}, code || "note"), ` ${text}`);
    });
    const sel = s.selection;
    const picked = h(
      "div",
      { class: "selection", "data-testid": "selection" },
      sel === null
        ? h("span", { class: "meta" }, "Click an outline edge or a named point of an arranged piece.")
        : sel.kind === "edge"
          ? h("span", {}, h("strong", {}, `${sel.piece}: edge ${sel.id}`), ` from ${sel.from} to ${sel.to}, length ${formatCm(sel.length_m * 100, 2)} cm`)
          : h("span", {}, h("strong", {}, `${sel.piece}: point ${sel.name}`), ` at ${sel.world_m.map((x) => (x * 100).toFixed(1)).join(", ")} cm`),
    );
    results.replaceChildren(
      summary,
      rings,
      ringNote,
      viol,
      notices.length > 0 ? h("div", { class: "notices", "data-testid": "arrange-notices" }, ...notices) : h("span"),
      h("h3", {}, "Picked"),
      picked,
    );
  };

  const renderHover = (): void => {
    const s = store.get();
    hover.textContent = s.arranged ? (s.hoverText ?? nearestText(store)) : "";
  };

  store.subscribe((_, changed) => {
    if (["arrangeOptions", "showMeshLines", "placements", "busy", "arranged"].some((k) => changed.includes(k as never))) renderControls();
    if (["arranged", "arrangement", "placed", "selection", "lastAlign"].some((k) => changed.includes(k as never))) renderResults();
    if (["hoverText", "selectedPiece", "placed", "arranged"].some((k) => changed.includes(k as never))) renderHover();
  });
  renderControls();
  renderResults();
  renderHover();
}

function nearestText(store: Store): string {
  const s = store.get();
  const p = s.selectedPiece ? s.placed.get(s.selectedPiece)?.arrange : undefined;
  const n = p ? nearestPiece(p.min_distance_to_pieces_m) : null;
  return n ? `${s.selectedPiece}: nearest piece ${n.name} ${formatMm(n.distance_m)} mm` : "";
}
