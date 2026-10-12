// Clone & mirror dialog: choose the mirror axis, the name of the copy, and whether to place the pair right away.
// Clone dialog: exact copies (same orientation, no axis), a name, how many, and whether to place one at the counterpart.
import { cloneName, copyNames } from "../lib";
import type { Store } from "../state";
import type { Axis, PairMode } from "../types";
import type { Actions } from "./actions";
import { h } from "./dom";
import type { Inspector } from "./inspector";

type AxisKind = "bbox_center" | "edge" | "x";

export function mountCloneDialog(store: Store, actions: Actions, inspector: Inspector): (piece: string) => void {
  const dialog = h("dialog", { class: "clone-dialog", "data-testid": "clone-dialog" });
  document.body.append(dialog);

  const open = (piece: string): void => {
    const s = store.get();
    const info = s.build?.pieces.find((p) => p.name === piece) ?? s.derivedInfos.get(piece);
    if (!info || info.status !== "ok") return;
    const points = [...(info.named_points ?? [])].sort((a, b) => Number(b.on_outline) - Number(a.on_outline) || a.name.localeCompare(b.name));
    const taken = new Set([...(s.build?.pieces ?? []).map((p) => p.name), ...s.derived.map((d) => d.id)]);
    let kind: AxisKind = "edge";
    let from = points.find((p) => p.on_outline)?.name ?? "";
    let to = points.filter((p) => p.on_outline)[1]?.name ?? "";
    let xpt = from;
    let nextPick: "from" | "to" = "from";
    const side = s.placed.get(piece)?.extends;
    let name = cloneName(piece, side);
    for (let i = 2; taken.has(name); i++) name = `${cloneName(piece, side)}${i}`;
    let pairNow = true;
    let mode: PairMode = "same_anchor";
    let modeTouched = false;

    const opts = (sel: string): HTMLOptionElement[] =>
      points.map((p) => h("option", { value: p.name, selected: p.name === sel }, `${p.name}${p.on_outline ? "" : " (inside)"}`));
    const selFrom = h("select", { "data-testid": "clone-from" }, ...opts(from));
    const selTo = h("select", { "data-testid": "clone-to" }, ...opts(to));
    const selX = h("select", { "data-testid": "clone-xpoint" }, ...opts(xpt));
    const nameIn = h("input", { type: "text", value: name, "data-testid": "clone-name" });
    const modeSel = h(
      "select",
      { "data-testid": "clone-pair-mode" },
      h("option", { value: "same_anchor" }, "centre-line pair: same anchor"),
      h("option", { value: "mirror_anchor" }, "side pair: mirrored anchor"),
    );
    const pairIn = h("input", { type: "checkbox", "data-testid": "clone-pair-now" });
    pairIn.checked = true;
    const msg = h("div", { class: "warn", "data-testid": "clone-msg" });
    const rows = {
      edge: h("div", { class: "row2" }, h("label", {}, "From ", selFrom), h("label", {}, "To ", selTo)),
      x: h("div", { class: "row2" }, h("label", {}, "Through ", selX)),
    };
    const sync = (): void => {
      rows.edge.hidden = kind !== "edge";
      rows.x.hidden = kind !== "x";
      selFrom.value = from;
      selTo.value = to;
      selX.value = xpt;
      if (!modeTouched) {
        mode = kind === "edge" ? "same_anchor" : "mirror_anchor";
        modeSel.value = mode;
      }
      inspector.setPickHandler(
        kind === "bbox_center"
          ? null
          : (n) => {
              if (kind === "x") xpt = n;
              else if (nextPick === "from") [from, nextPick] = [n, "to"];
              else [to, nextPick] = [n, "from"];
              sync();
            },
      );
    };
    const radio = (k: AxisKind, label: string, tid: string): HTMLElement => {
      const r = h("input", { type: "radio", name: "axis", value: k, "data-testid": tid });
      r.checked = kind === k;
      r.addEventListener("change", () => {
        kind = k;
        sync();
      });
      return h("label", { class: "radio" }, r, label);
    };
    selFrom.addEventListener("change", () => (from = selFrom.value));
    selTo.addEventListener("change", () => (to = selTo.value));
    selX.addEventListener("change", () => (xpt = selX.value));
    nameIn.addEventListener("input", () => (name = nameIn.value.trim()));
    modeSel.addEventListener("change", () => {
      mode = modeSel.value as PairMode;
      modeTouched = true;
    });
    pairIn.addEventListener("change", () => (pairNow = pairIn.checked));

    const close = (): void => {
      inspector.setPickHandler(null);
      dialog.close();
    };
    const create = (): void => {
      name = nameIn.value.trim();
      let axis: Axis;
      if (kind === "edge") {
        if (!from || !to || from === to) return void (msg.textContent = "Choose two different named points for the axis.");
        axis = { type: "edge", from_point: from, to_point: to };
      } else if (kind === "x") {
        const p = points.find((q) => q.name === xpt);
        if (!p) return void (msg.textContent = "Choose a point for the vertical line.");
        axis = { type: "x", value_m: p.xy_m[0] };
      } else {
        axis = { type: "bbox_center" };
      }
      if (!/^[A-Za-z0-9_.-]+$/.test(name)) return void (msg.textContent = "The name may use letters, digits, _ . - only.");
      if (taken.has(name)) return void (msg.textContent = `${name} is already used.`);
      close();
      void actions.cloneAndPair({ source: piece, axis, id: name, pairNow, mode });
    };

    dialog.replaceChildren(
      h("h3", {}, `Clone & mirror ${piece}`),
      h("p", { class: "meta" }, "The copy is a separate, linked piece (as in CLO3D); both are placed independently or as a pair."),
      h("fieldset", {}, h("legend", {}, "Mirror axis"), radio("bbox_center", "Flip in place (bbox centre)", "axis-bbox"), radio("edge", "Across an edge: two named points (click dots in the outline, or choose)", "axis-edge"), rows.edge, radio("x", "Across a vertical line through a point", "axis-x"), rows.x),
      h("label", { class: "row2" }, "Name ", nameIn),
      h("label", { class: "radio" }, pairIn, "Place pair now"),
      h("label", { class: "row2" }, "Pair mode ", modeSel),
      msg,
      h("div", { class: "dialog-buttons" }, h("button", { class: "btn", "data-testid": "clone-cancel", onclick: close }, "Cancel"), h("button", { class: "btn primary", "data-testid": "clone-create", onclick: create }, "Create")),
    );
    dialog.addEventListener("close", () => inspector.setPickHandler(null), { once: true });
    sync();
    dialog.show(); // non-modal, so the outline in the side panel stays clickable
  };
  return open;
}

export function mountCopyDialog(store: Store, actions: Actions): (piece: string) => void {
  const dialog = h("dialog", { class: "clone-dialog", "data-testid": "copy-dialog" });
  document.body.append(dialog);

  return (piece: string): void => {
    const s = store.get();
    const taken = new Set([...(s.build?.pieces ?? []).map((p) => p.name), ...s.derived.map((d) => d.id)]);
    const nameIn = h("input", { type: "text", value: copyNames(`${piece}_copy`, 1, taken)[0] ?? "", "data-testid": "copy-name" });
    const countIn = h("input", { type: "number", min: 1, max: 10, step: 1, value: 1, "data-testid": "copy-count" });
    const placeIn = h("input", { type: "checkbox", "data-testid": "copy-place-counterpart" });
    placeIn.checked = true;
    const msg = h("div", { class: "warn", "data-testid": "copy-msg" });
    const close = (): void => dialog.close();
    const create = (): void => {
      const name = nameIn.value.trim();
      const count = Number(countIn.value);
      if (!/^[A-Za-z0-9_.-]+$/.test(name)) return void (msg.textContent = "The name may use letters, digits, _ . - only.");
      if (taken.has(name)) return void (msg.textContent = `${name} is already used.`);
      if (!Number.isInteger(count) || count < 1 || count > 10) return void (msg.textContent = "Make 1 to 10 copies.");
      close();
      void actions.copyPieces({ source: piece, ids: copyNames(name, count, taken), placeAtCounterpart: placeIn.checked });
    };
    dialog.replaceChildren(
      h("h3", {}, `Clone ${piece}`),
      h("p", { class: "meta" }, "Exact copies, same orientation: pieces cut twice or identical pieces (pockets, belt loops). For left/right halves use Clone & mirror."),
      h("label", { class: "row2" }, "Name ", nameIn),
      h("label", { class: "row2" }, "Copies ", countIn),
      h("label", { class: "radio" }, placeIn, "Place at the counterpart point now (same rotation)"),
      msg,
      h("div", { class: "dialog-buttons" }, h("button", { class: "btn", "data-testid": "copy-cancel", onclick: close }, "Cancel"), h("button", { class: "btn primary", "data-testid": "copy-create", onclick: create }, "Create")),
    );
    dialog.show();
  };
}
