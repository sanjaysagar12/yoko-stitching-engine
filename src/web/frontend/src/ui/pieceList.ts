// Left panel: pieces (name, size in cm). Derived pieces (mirrored clones, copies) are nested under their base piece with
// a link label.
import { baseOf, formatCm } from "../lib";
import type { Store } from "../state";
import type { PieceInfo } from "../types";
import type { Actions } from "./actions";
import { h } from "./dom";

export function mountPieceList(host: HTMLElement, store: Store, actions: Actions): void {
  const list = h("div", { class: "piece-list", "data-testid": "piece-list" });
  host.append(h("h2", {}, "Pieces"), list);

  const row = (p: PieceInfo, nested: boolean): HTMLElement => {
    const s = store.get();
    if (p.status !== "ok") {
      return h(
        "div",
        { class: "piece failed", title: p.reason ?? "", "data-testid": `piece-${p.name}` },
        h("span", { class: "pname" }, p.name),
        h("span", { class: "psize" }, p.tool ? `unsupported: ${p.tool}` : (p.code ?? "failed")),
        h("span", { class: "preason" }, p.reason ?? ""),
      );
    }
    const placed = s.placed.get(p.name);
    const side = placed ? ` (${placed.extends})` : "";
    const btn = h(
      "button",
      {
        class: `piece${nested ? " nested" : ""}${s.selectedPiece === p.name ? " selected" : ""}${placed ? " placed" : ""}`,
        "data-testid": `piece-${p.name}`,
        onclick: () => actions.selectPiece(s.selectedPiece === p.name ? null : p.name),
      },
      h("span", { class: "pname" }, nested ? `↳ ${p.name}` : p.name),
      h("span", { class: "psize" }, `${formatCm(p.width_cm)} x ${formatCm(p.height_cm)} cm`),
      p.mirror_of ? h("span", { class: "plink", title: `linked mirrored copy of ${p.mirror_of}` }, `⇆ mirror of ${p.mirror_of}`) : null,
      p.copy_of ? h("span", { class: "plink", "data-testid": `copy-badge-${p.name}`, title: `exact copy of ${p.copy_of}` }, `⧉ copy of ${p.copy_of}`) : null,
      placed ? h("span", { class: "pplaced" }, `on ${placed.point}${side}`) : null,
    );
    const hiddenNow = s.hiddenPieces.has(p.name);
    const eye = placed
      ? h("button", { class: `btn small eye${hiddenNow ? " off" : ""}`, title: hiddenNow ? `Show ${p.name}` : `Hide ${p.name}`, "data-testid": `eye-${p.name}`, onclick: () => actions.toggleHidden(p.name) }, hiddenNow ? "show" : "hide")
      : null;
    if (!nested) return eye ? h("div", { class: "piece-row" }, btn, eye) : btn;
    return h(
      "div",
      { class: "nested-row" },
      btn,
      eye,
      h("button", { class: "btn small del", title: `Delete ${p.name}`, "data-testid": `derived-del-${p.name}`, onclick: () => actions.deleteDerived(p.name) }, "✕"),
    );
  };

  const render = (): void => {
    const s = store.get();
    const children: HTMLElement[] = [];
    if (s.pendingPoint) children.push(h("div", { class: "hint" }, `Point ${s.pendingPoint} chosen: now click a piece`));
    if (s.pendingPair) children.push(h("div", { class: "hint", "data-testid": "pending-pair" }, `Click an arrangement point to place ${s.pendingPair.source} and ${s.pendingPair.derivedId} as a pair`));
    for (const p of s.build?.pieces ?? []) {
      children.push(row(p, false));
      for (const d of s.derived.filter((x) => baseOf(s.derived, x.id) === p.name)) {
        const info = s.derivedInfos.get(d.id);
        if (info) children.push(row(info, true));
      }
    }
    list.replaceChildren(...children);
  };
  store.subscribe((_, changed) => {
    if (["build", "placed", "selectedPiece", "pendingPoint", "pendingPair", "derived", "derivedInfos", "hiddenPieces"].some((k) => changed.includes(k as never))) render();
  });
  render();
}
