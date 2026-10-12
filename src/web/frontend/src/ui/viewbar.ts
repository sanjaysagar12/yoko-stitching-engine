// Floating bar over the 3D view: camera sides, X-ray for hidden points, and a filter for point names.
import type { Store } from "../state";
import type { Actions, ViewSide } from "./actions";
import { h } from "./dom";

export function mountViewbar(host: HTMLElement, store: Store, actions: Actions): void {
  const sides: Array<[ViewSide, string]> = [
    ["front", "Front"],
    ["back", "Back"],
    ["left", "Left"],
    ["right", "Right"],
    ["top", "Top"],
  ];
  const xray = h("input", { type: "checkbox", "data-testid": "toggle-xray" });
  xray.addEventListener("change", () => store.set({ xray: xray.checked }));
  const filter = h("input", { type: "search", placeholder: "Filter points...", "data-testid": "point-filter", "aria-label": "Filter points" });
  filter.addEventListener("input", () => store.set({ pointFilter: filter.value }));
  host.append(
    ...sides.map(([s, label]) => h("button", { class: "btn small", "data-testid": `view-${s}`, title: s === "left" ? "the wearer's left (+x)" : "", onclick: () => actions.viewSide(s) }, label)),
    h("button", { class: "btn small", "data-testid": "view-pieces", onclick: actions.viewPieces }, "Pieces"),
    h("label", { class: "toggle", title: "make points behind the avatar pickable" }, xray, "X-ray"),
    filter,
  );
}
