// Strip under the 3D view: the engine's verdict on the last placement (sizes, body clearance, gaps between pieces).
// Every number is the engine's own measurement from the answer's `validity` block; nothing is recomputed here.
import { formatCm } from "../lib";
import type { Store } from "../state";
import type { Validity, Violation } from "../types";
import type { Actions } from "./actions";
import { h } from "./dom";

const KIND: Record<Violation["kind"], string> = {
  CLEARANCE_BODY: "too close to the body",
  GAP_PIECES: "gap too small",
  STANDOFF_EXCEEDED: "standoff limit",
  SIZE_CHANGED: "size changed",
  PAIR_ASYMMETRY: "mirror pair not symmetric",
  PAIR_STACKED: "mirror pair stacked, not side by side",
};

const cm = (m: number | null): string => (m === null ? "-" : `${formatCm(m * 100, 1)} cm`);

/** The one-line summary of a valid layout (also used by tests through the DOM). */
export function summary(v: Validity): string {
  const size = v.size_ok ? `Size unchanged (max ${v.max_size_change_pct.toFixed(2)} %)` : "Size CHANGED";
  const gap = v.min_distance_between_pieces_m === null ? "no pieces near each other" : `Smallest gap between pieces ${cm(v.min_distance_between_pieces_m)}`;
  return `${size} · Body clearance ${cm(v.min_distance_to_avatar_m)} · ${gap}`;
}

export function mountStatus(host: HTMLElement, store: Store, actions: Actions): void {
  const strip = h("div", { class: "status-strip", "data-testid": "status-strip", role: "status", "aria-live": "polite" });
  strip.hidden = true;
  host.append(strip);

  const render = (): void => {
    const s = store.get();
    const v = s.lastAlign?.validity;
    if (!v || s.placements.size === 0) {
      strip.hidden = true;
      return;
    }
    strip.hidden = false;
    strip.className = `status-strip ${v.valid ? "ok" : "bad"}`;
    if (v.valid) {
      strip.replaceChildren(h("span", { "data-testid": "status-summary" }, summary(v)));
      return;
    }
    const items = v.violations.map((x, i) =>
      h(
        "button",
        { class: "status-violation", "data-testid": `status-violation-${i}`, title: "Show where", onclick: () => actions.focusViolation(i) },
        `${x.pieces.join(" / ")}: ${KIND[x.kind]}` + (x.kind === "SIZE_CHANGED" ? "" : ` ${cm(x.value_m)} (needs ${cm(x.required_m)})`) + (x.limit && x.limit !== "none" ? ` [${x.limit}]` : ""),
      ),
    );
    strip.replaceChildren(
      h("strong", { "data-testid": "status-summary" }, `Not valid: ${v.violations.length} problem${v.violations.length === 1 ? "" : "s"}. `),
      h("span", {}, summary(v)),
      h("div", { class: "status-list" }, ...items),
    );
  };
  store.subscribe((_, changed) => {
    if (changed.includes("lastAlign") || changed.includes("placements")) render();
  });
  render();
}
