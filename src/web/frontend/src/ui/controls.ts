// Right panel: the selected piece (clone, pair, placement controls, side, notices, scale check).
import { formatCm, splitWarning } from "../lib";
import type { Store } from "../state";
import type { PairMode, PieceInfo, PlacedResult, WrapMode } from "../types";
import { RING_NAMES } from "../types";
import type { Actions } from "./actions";
import { h } from "./dom";

const LEVELS = ["Chest", "Waist", "Hip"] as const;

/** Body loop of the level named in a placement's fit (set by body_slice) in cm, or null. */
function loopCm(res: PlacedResult | undefined, level: string): number | null {
  const levels = res?.fit["levels"] as Record<string, { body_loop_cm?: number }> | undefined;
  return levels?.[level]?.body_loop_cm ?? null;
}

export function mountControls(host: HTMLElement, store: Store, actions: Actions): void {
  const body = h("div", { class: "controls", "data-testid": "controls" });
  host.append(h("h2", {}, "Selected piece"), body);

  const put = (...nodes: Array<Node | null>): void => body.replaceChildren(...nodes.filter((n): n is Node => n !== null));

  const render = (): void => {
    const s = store.get();
    const name = s.selectedPiece;
    const info: PieceInfo | undefined = s.build?.pieces.find((p) => p.name === name) ?? (name ? s.derivedInfos.get(name) : undefined);
    if (!name || !info) {
      body.replaceChildren(h("p", { class: "hint" }, "Select a piece in the list, then click an arrangement point (or the point first, then the piece)."));
      return;
    }
    const pl = s.placements.get(name);
    const res = s.placed.get(name);
    const draftRef = s.referenceDraft.get(name) ?? "bbox_centre";
    const off = !pl;
    const num = (id: string, value: number | null, step: number, onChange: (v: number | null) => void, ph = ""): HTMLInputElement => {
      const el = h("input", { type: "number", step, value: value ?? "", placeholder: ph, "data-testid": id, disabled: off });
      el.addEventListener("change", () => onChange(el.value === "" ? null : Number(el.value)));
      return el;
    };
    const row = (label: string, ...c: Array<Node | string>): HTMLElement => h("label", { class: "row" }, h("span", {}, label), ...c);
    const check = (id: string, v: boolean, on: (b: boolean) => void): HTMLInputElement => {
      const el = h("input", { type: "checkbox", "data-testid": id, disabled: off });
      el.checked = v;
      el.addEventListener("change", () => on(el.checked));
      return el;
    };
    const wrap = h(
      "select",
      { "data-testid": "ctl-wrap", disabled: off },
      h("option", { value: "flat" }, "flat"),
      h("option", { value: "around" }, "around the body part"),
      h("option", { value: "body_slice" }, "body slice (torso)"),
    );
    wrap.value = pl?.wrap ?? "flat";
    wrap.addEventListener("change", () => actions.updatePlacement(name, { wrap: wrap.value as WrapMode }));
    const ref = h(
      "select",
      { "data-testid": "ctl-reference" },
      h("option", { value: "bbox_centre" }, "bbox centre"),
      ...(info.named_points ?? []).map((p) => h("option", { value: p.name }, p.name + (p.on_outline ? "" : " (inside)"))),
    );
    ref.value = pl?.reference ?? draftRef;
    ref.addEventListener("change", () => actions.setReference(name, ref.value));
    const rot = h("input", { type: "range", min: -180, max: 180, step: 1, value: pl?.rotation_deg ?? 0, disabled: off, "data-testid": "ctl-rotation-slider" });
    const rotNum = num("ctl-rotation", pl?.rotation_deg ?? 0, 1, (v) => actions.updatePlacement(name, { rotation_deg: v ?? 0 }));
    rot.addEventListener("input", () => {
      rotNum.value = rot.value;
    });
    rot.addEventListener("change", () => actions.updatePlacement(name, { rotation_deg: Number(rot.value) }));

    // -- auto-arrange: which body region the piece wraps, its layer and whether it stays where it is ------------------------
    const ring = h("select", { "data-testid": "ctl-ring", disabled: off, title: "Body region of the shell (automatic: the region of the point)" }, h("option", { value: "" }, "automatic"), ...RING_NAMES.map((n) => h("option", { value: n }, n)));
    ring.value = pl?.ring ?? "";
    ring.addEventListener("change", () => actions.updatePlacement(name, { ring: ring.value === "" ? undefined : ring.value }));
    const layer = h("select", { "data-testid": "ctl-layer", disabled: off, title: "0 sits on the shell, 1 and 2 float above it (pockets, flaps)" }, h("option", { value: "" }, "automatic"), ...[0, 1, 2].map((n) => h("option", { value: n }, String(n))));
    layer.value = pl?.layer === undefined ? "" : String(pl.layer);
    layer.addEventListener("change", () => actions.updatePlacement(name, { layer: layer.value === "" ? undefined : Number(layer.value) }));
    const arr = res?.arrange;

    // -- pairs: a base piece with clones, or a clone with its source -------------------------------------------------
    const sourceName = info.mirror_of ?? name;
    const clones = s.derived.filter((d) => d.source === sourceName);
    const pairBox = h("div", { class: "pair-box", "data-testid": "pair-box" });
    if (clones.length > 0) {
      pairBox.append(h("h3", {}, "Pairs"));
      const sourcePlaced = s.placements.has(sourceName);
      for (const d of clones) {
        const mk = (mode: PairMode, label: string, tid: string): HTMLButtonElement => {
          const b = h("button", { class: "btn", "data-testid": `${tid}-${d.id}`, disabled: !sourcePlaced, title: sourcePlaced ? "" : `place ${sourceName} first`, onclick: () => actions.placePair(sourceName, d.id, mode) }, label);
          if (mode === "mirror_anchor") {
            b.addEventListener("mouseenter", () => actions.hoverPair(sourceName, mode));
            b.addEventListener("mouseleave", () => actions.hoverPair(sourceName, null));
          }
          return b;
        };
        pairBox.append(
          h("div", { class: "pair-row" }, h("span", { class: "pname" }, `${sourceName} + ${d.id}`), mk("same_anchor", "Place pair (same anchor)", "pair-same"), mk("mirror_anchor", "Place pair (mirrored anchor)", "pair-mirror")),
        );
      }
    }

    // -- scale check -------------------------------------------------------------------------------------------------
    const w = info.width_cm ?? 0;
    const hh = info.height_cm ?? 0;
    const same = res ? Math.abs(res.bbox_cm.width - w) < 0.01 && Math.abs(res.bbox_cm.height - hh) < 0.01 : null;
    const ext = res?.fit["placed_extent_cm"];
    const mate = clones.map((d) => d.id).concat(info.mirror_of ? [info.mirror_of] : []).filter((n) => n !== name && s.placed.has(n));
    const pairLines: HTMLElement[] = [];
    if (res && mate.length > 0) {
      const other = s.placed.get(mate[0] as string);
      const ow = s.derivedInfos.get(mate[0] as string)?.width_cm ?? s.build?.pieces.find((p) => p.name === mate[0])?.width_cm ?? 0;
      const total = w + ow;
      const label = `${name} + ${mate[0]}: ${formatCm(total, 1)} cm`;
      const loop = LEVELS.map((l) => [l, loopCm(res, l) ?? loopCm(other, l)] as const).find(([, v]) => v !== null);
      pairLines.push(h("div", { "data-testid": "scale-check-pair" }, `Pair total width ${label}${loop?.[1] != null && loop[1] !== undefined ? ` vs avatar ${loop[0].toLowerCase()} half-loop ${formatCm(loop[1] / 2, 1)} cm` : ""}`));
    }

    const notices = (res?.warnings ?? []).map((m) => {
      const { code, text } = splitWarning(m);
      return h("div", { class: `notice ${code}`, "data-testid": `notice-${code || "info"}` }, h("strong", {}, code || "note"), ` ${text}`);
    });
    const sideText = res ? { left: "→ wearer's left", right: "→ wearer's right", centre: "centred on its reference" }[res.extends] : "";

    put(
      h("div", { class: "pname-big" }, name),
      info.mirror_of ? h("div", { class: "meta", "data-testid": "mirror-of" }, `⇆ linked mirrored copy of ${info.mirror_of}`) : null,
      h(
        "div",
        { class: "btn-row" },
        h("button", { class: "btn", "data-testid": "clone-open", onclick: () => actions.openClone(name) }, "Clone & mirror"),
        info.mirror_of ? h("button", { class: "btn danger", "data-testid": "derived-delete", onclick: () => actions.deleteDerived(name) }, "Delete clone") : null,
      ),
      h("div", { class: "placed-on", "data-testid": "placed-on" }, pl ? `placed on ${pl.point}` : "not placed - click an arrangement point"),
      res ? h("div", { class: `side side-${res.extends}`, "data-testid": "extends" }, `extends ${sideText}`) : null,
      row("Offset (cm)", num("ctl-offset", pl?.offset_m == null ? null : pl.offset_m * 100, 0.5, (v) => actions.updatePlacement(name, { offset_m: v === null ? null : v / 100 }), "point default")),
      row("Rotation (deg)", rot, rotNum),
      row("Mirror", check("ctl-mirror", pl?.mirror ?? false, (b) => actions.updatePlacement(name, { mirror: b }))),
      row("Flip facing", check("ctl-flip", pl?.flip_facing ?? false, (b) => actions.updatePlacement(name, { flip_facing: b }))),
      row("Wrap", wrap),
      row("Reference", ref),
      row("Arrange: ring", ring),
      row("Arrange: layer", layer),
      row("Arrange: shift along axis (cm)", num("ctl-shift", pl?.shift_m ? pl.shift_m[1] * 100 : 0, 1, (v) => actions.updatePlacement(name, { shift_m: v ? [pl?.shift_m?.[0] ?? 0, v / 100] : undefined }))),
      row("Arrange: locked", check("ctl-locked", pl?.locked ?? false, (b) => actions.updatePlacement(name, { locked: b || undefined }))),
      arr ? h("div", { class: "meta", "data-testid": "arrange-piece" }, `arranged on ${arr.ring}, layer ${arr.layer}${arr.locked ? " (locked)" : ""}, standoff ${arr.standoff_m === null ? "-" : (arr.standoff_m * 100).toFixed(1) + " cm"}, shift ${arr.shift_m.map((x) => (x * 100).toFixed(1)).join(" / ")} cm, edges change ${arr.edge_change_p50_pct ?? "-"} % (median)`) : null,
      h("button", { class: "btn danger", "data-testid": "ctl-remove", disabled: off, onclick: () => actions.removePlacement(name) }, "Remove placement"),
      pairBox,
      notices.length > 0 ? h("div", { class: "notices", "data-testid": "notices" }, ...notices) : null,
      h(
        "div",
        { class: "scale-check", "data-testid": "scale-check" },
        h("h3", {}, "Scale check"),
        h("div", {}, `Flat pattern: ${formatCm(w)} x ${formatCm(hh)} cm`),
        res
          ? h("div", { "data-testid": "scale-check-placed" }, `Placed (bbox reported): ${formatCm(res.bbox_cm.width)} x ${formatCm(res.bbox_cm.height)} cm ${same ? "- unchanged" : "- DIFFERENT"}`)
          : h("div", { class: "meta" }, "Placed: -"),
        ...pairLines,
        Array.isArray(ext) ? h("div", { class: "meta" }, `3D extent after placement: ${ext.map((x) => Number(x).toFixed(1)).join(" x ")} cm`) : null,
        res?.fit["wrap_angle_deg"] !== undefined ? h("div", { class: "meta" }, `Wrapped over ${Number(res.fit["wrap_angle_deg"]).toFixed(0)} deg (${(Number(res.fit["width_over_circumference"]) * 100).toFixed(0)} % of the circumference)`) : null,
        res ? levelTable(res) : null,
      ),
    );
  };
  store.subscribe((_, changed) => {
    if (["selectedPiece", "placements", "placed", "build", "derived", "derivedInfos", "referenceDraft"].some((k) => changed.includes(k as never))) render();
  });
  render();
}

/** Garment width against the body loop at the chest, waist and hip (body_slice only). */
function levelTable(res: PlacedResult): HTMLElement | null {
  const levels = res.fit["levels"] as Record<string, { piece_width_cm: number; body_loop_cm: number; ease_quarter_cm: number; ease_half_cm: number }> | undefined;
  if (!levels || Object.keys(levels).length === 0) return null;
  return h(
    "table",
    { class: "mtable small", "data-testid": "levels" },
    h("thead", {}, h("tr", {}, ...["", "piece", "body loop", "vs 1/4", "vs 1/2"].map((t) => h("th", {}, t)))),
    h("tbody", {}, ...LEVELS.filter((l) => levels[l]).map((l) => {
      const e = levels[l] as { piece_width_cm: number; body_loop_cm: number; ease_quarter_cm: number; ease_half_cm: number };
      return h("tr", {}, h("td", {}, l), h("td", {}, formatCm(e.piece_width_cm, 1)), h("td", {}, formatCm(e.body_loop_cm, 1)), h("td", {}, formatCm(e.ease_quarter_cm, 1)), h("td", {}, formatCm(e.ease_half_cm, 1)));
    })),
  );
}
