// Orientation tools of the selected piece: rotate, align to grain, flip, upside down, tilt (all isometries: the engine
// never changes a size), hang from a body level, undo / redo of these edits. Pure DOM; edits go through actions.
import type { LevelAnchor, Placement, PlacedResult } from "../types";
import { BODY_LEVELS } from "../types";
import type { Actions } from "./actions";
import { h } from "./dom";

type OrientEdit = Pick<Placement, "rotation_deg" | "orient" | "tilt" | "flip_facing" | "anchor">;

const STEPS = [-90, -15, -5, -1, 1, 5, 15, 90];
const undo = new Map<string, OrientEdit[]>();
const redo = new Map<string, OrientEdit[]>();

function snapshot(pl: Placement): OrientEdit {
  return { rotation_deg: pl.rotation_deg, orient: pl.orient, tilt: pl.tilt ? { ...pl.tilt } : undefined, flip_facing: pl.flip_facing, anchor: pl.anchor };
}

/** Normalise an angle to (-180, 180]. */
export function normAngle(a: number): number {
  const r = ((((a + 180) % 360) + 360) % 360) - 180;
  return r === -180 ? 180 : r;
}

/** Apply an orientation edit and remember the previous state for undo. */
export function editOrientation(actions: Actions, piece: string, pl: Placement, patch: Partial<OrientEdit>): void {
  const u = undo.get(piece) ?? [];
  u.push(snapshot(pl));
  undo.set(piece, u.slice(-50));
  redo.set(piece, []);
  actions.updatePlacement(piece, patch);
}

function step(actions: Actions, piece: string, pl: Placement, from: Map<string, OrientEdit[]>, to: Map<string, OrientEdit[]>): void {
  const stack = from.get(piece) ?? [];
  const prev = stack.pop();
  if (!prev) return;
  to.set(piece, [...(to.get(piece) ?? []), snapshot(pl)]);
  actions.updatePlacement(piece, prev);
}

export function orientationSection(piece: string, pl: Placement | undefined, res: PlacedResult | undefined, actions: Actions, pointNames: string[]): HTMLElement {
  const off = !pl;
  const box = h("div", { class: "orient-box", "data-testid": "orient-box" }, h("h3", {}, "Rotate"));
  if (!pl) {
    box.append(h("p", { class: "hint" }, "Place the piece to rotate it."));
    return box;
  }
  const rot = pl.rotation_deg ?? 0;
  const num = h("input", { type: "number", step: 1, value: rot, "data-testid": "rot-value", disabled: off });
  num.addEventListener("change", () => editOrientation(actions, piece, pl, { rotation_deg: normAngle(Number(num.value)) }));
  const btns = STEPS.map((d) =>
    h("button", { class: "btn small", "data-testid": `rot-${d > 0 ? "p" : "m"}${Math.abs(d)}`, onclick: () => editOrientation(actions, piece, pl, { rotation_deg: normAngle(rot + d) }) }, `${d > 0 ? "+" : ""}${d}`),
  );
  const grain = h("button", { class: "btn", "data-testid": "rot-grain", title: "Grainline (else the long axis of a tall piece) along the body / limb axis, top up", onclick: () => editOrientation(actions, piece, pl, { orient: "grain", rotation_deg: 0 }) }, "Align to grain");
  const flip = h("button", { class: "btn", "data-testid": "rot-flip", title: "Turn the pattern face to the other side", onclick: () => editOrientation(actions, piece, pl, { flip_facing: !pl.flip_facing }) }, "Flip face");
  const upside = h("button", { class: "btn", "data-testid": "rot-upside", onclick: () => editOrientation(actions, piece, pl, { rotation_deg: normAngle(rot + 180) }) }, "Upside down");
  const tilt = (key: "pitch_deg" | "roll_deg", label: string): HTMLElement => {
    const v = pl.tilt?.[key] ?? 0;
    const out = h("span", { class: "meta", "data-testid": `tilt-${key}-value` }, `${v.toFixed(0)}°`);
    const sl = h("input", { type: "range", min: -45, max: 45, step: 1, value: v, "data-testid": `tilt-${key}`, disabled: off });
    sl.addEventListener("input", () => (out.textContent = `${sl.value}°`));
    sl.addEventListener("change", () => editOrientation(actions, piece, pl, { tilt: { ...(pl.tilt ?? {}), [key]: Number(sl.value) } }));
    return h("label", { class: "row" }, h("span", {}, label), sl, out);
  };
  const u = h("button", { class: "btn", "data-testid": "rot-undo", disabled: !(undo.get(piece)?.length), onclick: () => step(actions, piece, pl, undo, redo) }, "Undo");
  const r = h("button", { class: "btn", "data-testid": "rot-redo", disabled: !(redo.get(piece)?.length), onclick: () => step(actions, piece, pl, redo, undo) }, "Redo");
  const arr = res?.arrange;
  const applied = arr
    ? `applied: ${(arr.applied_rotation_deg ?? rot).toFixed(1)}° in plane${arr.orient ? ` (orient: ${arr.orient})` : ""}${arr.tilt_deg ? `, tilt pitch ${arr.tilt_deg.pitch_deg.toFixed(0)}° roll ${arr.tilt_deg.roll_deg.toFixed(0)}°` : ""}`
    : "";
  box.append(
    h("label", { class: "row" }, h("span", {}, "Angle (deg)"), num),
    h("div", { class: "btn-row" }, ...btns),
    h("div", { class: "btn-row" }, grain, flip, upside),
    tilt("pitch_deg", "Tilt: top out (+)"),
    tilt("roll_deg", "Tilt: top to +x (+)"),
    h("div", { class: "btn-row" }, u, r),
    ...(applied ? [h("div", { class: "meta", "data-testid": "rot-applied" }, applied)] : []),
    hangSection(piece, pl, res, actions, pointNames),
  );
  return box;
}

/** Hang the piece from a body level: a named piece point (or its top / bottom edge) at the level's height. */
function hangSection(piece: string, pl: Placement, res: PlacedResult | undefined, actions: Actions, pointNames: string[]): HTMLElement {
  const cur: LevelAnchor | null = typeof pl.anchor === "object" && pl.anchor !== null ? pl.anchor : null;
  const level = h("select", { "data-testid": "hang-level" }, h("option", { value: "" }, "no (point / limb anchor)"), ...BODY_LEVELS.map((l) => h("option", { value: l }, l)));
  level.value = cur?.level ?? "";
  const what = h(
    "select",
    { "data-testid": "hang-what" },
    h("option", { value: "edge:top" }, "top edge"),
    h("option", { value: "edge:bottom" }, "bottom edge"),
    ...pointNames.map((n) => h("option", { value: `pt:${n}` }, n)),
  );
  what.value = cur?.piece_point ? `pt:${cur.piece_point}` : `edge:${cur?.edge ?? "top"}`;
  const offset = h("input", { type: "number", step: 0.5, value: cur?.offset_m ? cur.offset_m * 100 : 0, "data-testid": "hang-offset" });
  const apply = (): void => {
    if (level.value === "") return editOrientation(actions, piece, pl, { anchor: undefined });
    const [kind, val] = what.value.split(":") as [string, string];
    const a: LevelAnchor = { level: level.value as LevelAnchor["level"] };
    if (kind === "pt") a.piece_point = val;
    else a.edge = val as "top" | "bottom";
    if (Number(offset.value)) a.offset_m = Number(offset.value) / 100;
    editOrientation(actions, piece, pl, { anchor: a });
  };
  for (const el of [level, what, offset]) el.addEventListener("change", apply);
  const reg = res?.arrange?.registration;
  return h(
    "div",
    { class: "hang-box", "data-testid": "hang-box" },
    h("h3", {}, "Hang from level"),
    h("label", { class: "row" }, h("span", {}, "Level"), level),
    h("label", { class: "row" }, h("span", {}, "Piece point"), what),
    h("label", { class: "row" }, h("span", {}, "Above the level (cm)"), offset),
    reg
      ? h("div", { class: "meta", "data-testid": "hang-result" }, `${reg.registered} at the ${reg.level} level (${reg.level_y_m.toFixed(3)} m), ${(reg.offset_m * 100).toFixed(1)} cm off`)
      : null,
  );
}
