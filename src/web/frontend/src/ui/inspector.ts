// Piece inspector: the outline of the selected piece with its named points as labelled, clickable dots.
// Clicking a dot sets the piece reference (or, while the clone dialog is open, picks an axis point). Display only.
import type { Store } from "../state";
import type { PieceBundle, PieceInfo } from "../types";
import type { Actions } from "./actions";

const W = 276;
const H = 230;
const M = 14;
const NS = "http://www.w3.org/2000/svg";

export interface Inspector {
  /** While set, dot clicks go to this handler instead of setting the reference (used by the clone dialog). */
  setPickHandler(fn: ((pointName: string) => void) | null): void;
}

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

export function mountInspector(
  host: HTMLElement,
  store: Store,
  actions: Actions,
  getBundle: (info: PieceInfo) => Promise<PieceBundle>,
): Inspector {
  const title = document.createElement("div");
  title.className = "meta";
  const box = document.createElement("div");
  box.className = "inspector";
  box.dataset["testid"] = "inspector";
  host.append(title, box);
  let pick: ((n: string) => void) | null = null;
  let token = 0;

  const render = async (): Promise<void> => {
    const s = store.get();
    const info = s.selectedPiece ? (s.build?.pieces.find((p) => p.name === s.selectedPiece) ?? s.derivedInfos.get(s.selectedPiece)) : undefined;
    const my = ++token;
    if (!info || info.status !== "ok" || !info.bundle) {
      box.replaceChildren();
      title.textContent = "Select a piece to see its outline and named points.";
      return;
    }
    const b = await getBundle(info);
    if (my !== token) return;
    const [w, h] = b.bbox_m.max;
    const k = Math.min((W - 2 * M) / Math.max(w, 1e-9), (H - 2 * M) / Math.max(h, 1e-9));
    const ox = (W - w * k) / 2;
    const px = (x: number): number => ox + x * k;
    const py = (y: number): number => H - M - y * k;
    const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": `Outline of ${info.name}` });
    root.append(svg("path", { d: `M ${b.outline_m.map(([x, y]) => `${px(x)} ${py(y)}`).join(" L ")} Z`, class: "outline" }));

    // mirror axes: of the clones made from this piece, or the axis this clone was made with
    const axes: Array<{ x1: number; y1: number; x2: number; y2: number }> = [];
    const named = new Map(b.named_points.map((q) => [q.name, q.xy_m]));
    const addAxis = (a: PieceInfo["axis"]): void => {
      if (!a) return;
      if (a.type === "edge") {
        const p = named.get(a.from_point);
        const q = named.get(a.to_point);
        if (p && q) {
          const d = [q[0] - p[0], q[1] - p[1]];
          const n = Math.hypot(d[0] ?? 0, d[1] ?? 0) || 1;
          const L = (Math.hypot(w, h) * 1.1) / n; // extend the edge across the whole piece
          axes.push({ x1: px(p[0] - (d[0] ?? 0) * L), y1: py(p[1] - (d[1] ?? 0) * L), x2: px(p[0] + (d[0] ?? 0) * L), y2: py(p[1] + (d[1] ?? 0) * L) });
        }
      } else {
        const x = a.type === "x" ? a.value_m : w / 2;
        axes.push({ x1: px(x), y1: py(-0.02 * h), x2: px(x), y2: py(h * 1.02) });
      }
    };
    if (info.mirror_of) addAxis(info.axis);
    else for (const d of s.derived.filter((d) => d.source === info.name)) addAxis(d.axis);
    for (const a of axes) root.append(svg("line", { ...a, class: "axis" }));

    const placement = s.placements.get(info.name);
    const ref = placement?.reference ?? s.referenceDraft.get(info.name) ?? "bbox_centre";
    if (ref === "bbox_centre") {
      const [cx, cy] = b.default_reference.xy_m;
      root.append(svg("path", { d: `M ${px(cx) - 6} ${py(cy)} H ${px(cx) + 6} M ${px(cx)} ${py(cy) - 6} V ${py(cy) + 6}`, class: "refcross" }));
    }
    for (const q of b.named_points) {
      const g = svg("g", { class: "pt" });
      const isRef = q.name === ref;
      const c = svg("circle", {
        cx: px(q.xy_m[0]),
        cy: py(q.xy_m[1]),
        r: isRef ? 6 : 4,
        class: `dot${q.on_outline ? " on" : ""}${isRef ? " ref" : ""}`,
        "data-testid": `insp-pt-${q.name}`,
        tabindex: 0,
      });
      const t = svg("text", { x: px(q.xy_m[0]) + 7, y: py(q.xy_m[1]) - 5, class: isRef ? "lbl show" : "lbl" });
      t.textContent = q.name;
      const tip = svg("title");
      tip.textContent = `${q.name}${q.on_outline ? " (on the outline)" : " (inside)"}`;
      c.append(tip);
      const click = (): void => (pick ? pick(q.name) : actions.setReference(info.name, q.name));
      c.addEventListener("click", click);
      c.addEventListener("keydown", (ev) => {
        if ((ev as KeyboardEvent).key === "Enter") click();
      });
      g.append(c, t);
      root.append(g);
    }
    box.replaceChildren(root);
    title.textContent = `${info.name}: click a dot to set the reference${info.mirror_of ? ` (mirror of ${info.mirror_of})` : ""}. Dashed line = mirror axis.`;
  };

  store.subscribe((_, changed) => {
    if (["selectedPiece", "placements", "referenceDraft", "derived", "build"].some((k) => changed.includes(k as never))) void render();
  });
  void render();
  return {
    setPickHandler(fn) {
      pick = fn;
      box.classList.toggle("picking", fn !== null);
    },
  };
}
