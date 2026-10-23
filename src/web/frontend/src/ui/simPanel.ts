// Simulation panel (bottom right of the viewer): once the layout is stitched, Simulate sews and drapes it with the
// cloth solver on the server (fabric, mesh quality, frames). While it runs the newest frame is shown live; when it is
// done: the verdict (seams closed, stretch against the pattern, nothing inside the body), the animation (play,
// frame slider), Draped / Placed, piece colours, and the downloads (OBJ, zip with result and images).
import type { Color } from "three";
import {
  cancelSimulation,
  simulationDownloadUrl,
  simulationEmbedding,
  simulationFrames,
  simulationInfo,
  simulationLatest,
  simulationStatus,
  startSimulation,
} from "../api";
import type { SimulationInfo, SimulationOptions, SimulationStatus, StitchLayout } from "../api";
import { decodeF32, type SimView } from "../simView";
import { h } from "./dom";

export interface SimPanelContext {
  /** The stitched layout (request body with its stitching), or null when it is not stitched. */
  stitched(): (StitchLayout & { stitching: unknown }) | null;
  view: SimView;
  /** Faces of a placed piece's mesh (the draped mesh uses them, vertex for vertex). */
  faces(piece: string): Uint32Array | undefined;
  colour(piece: string): Color | null;
  /** Draped garment on (placed pieces and sewing lines hidden) or off. */
  showDraped(on: boolean): void;
}

export interface SimPanelHandle {
  /** The stitching changed: the panel re-reads whether it can simulate. */
  refresh(): void;
  /** The layout changed: show the placed pieces again (the draped garment belongs to the earlier layout). */
  hide(): void;
}

const message = (e: unknown): string => {
  const err = (e as { error?: { message?: string } }).error;
  return err?.message ?? (e instanceof Error ? e.message : String(e));
};
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

export function mountSimPanel(host: HTMLElement, ctx: SimPanelContext): SimPanelHandle {
  const panel = h("div", { class: "sim-panel", "data-testid": "sim-panel" });
  host.append(panel);
  let info: SimulationInfo | null = null;
  let opts: SimulationOptions = { fabric: "cotton_shirting", quality: "standard", frames: 240 };
  let job: SimulationStatus | null = null;
  let error = "";
  let draped = false;
  let pieceColours = false;
  let frameText = "";
  let token = 0;

  simulationInfo()
    .then((i) => {
      info = i;
      opts = { fabric: i.defaults.fabric, quality: i.defaults.quality, frames: i.defaults.frames };
      render();
    })
    .catch((e: unknown) => {
      error = message(e);
      render();
    });

  const setDraped = (on: boolean): void => {
    draped = on && ctx.view.loaded;
    ctx.view.group.visible = draped;
    ctx.showDraped(draped);
    render();
  };

  const loadEmbedding = async (id: string): Promise<void> => {
    const emb = await simulationEmbedding(id);
    ctx.view.setEmbedding(emb, ctx.faces, ctx.colour);
    ctx.view.setPieceColours(pieceColours);
    setDraped(true);
  };

  /** Follow a job until it ends: live frames while it runs, then every kept frame. */
  const follow = async (id: string, my: number): Promise<void> => {
    let embedded = false;
    let shown = -1;
    for (;;) {
      if (my !== token) return;
      job = await simulationStatus(id);
      if (!embedded && job.has_scene) {
        await loadEmbedding(id);
        embedded = true;
      }
      if (embedded && job.status === "running" && job.frame > shown) {
        const l = await simulationLatest(id);
        if (l.x && my === token) {
          ctx.view.show(decodeF32(l.x));
          shown = l.frame;
        }
      }
      render();
      if (job.status !== "queued" && job.status !== "running") break;
      await sleep(1500);
    }
    if (job.status === "done" && my === token) {
      const f = await simulationFrames(id);
      ctx.view.setFrames(decodeF32(f.x), f.frames);
    }
    render();
  };

  const simulate = async (): Promise<void> => {
    const body = ctx.stitched();
    if (!body) return;
    const my = ++token;
    error = "";
    ctx.view.clear();
    setDraped(false);
    try {
      job = await startSimulation({ ...body, options: opts });
      render();
      await follow(job.id, my);
    } catch (e) {
      if (my === token) error = message(e);
      render();
    }
  };

  const cancel = async (): Promise<void> => {
    if (!job) return;
    try {
      job = await cancelSimulation(job.id);
    } catch (e) {
      error = message(e);
    }
    render();
  };

  ctx.view.onFrame = (i, frame) => {
    frameText = `frame ${frame}`;
    const slider = panel.querySelector<HTMLInputElement>("[data-testid=sim-frame]");
    if (slider) slider.value = String(i);
    const label = panel.querySelector<HTMLElement>("[data-testid=sim-frame-label]");
    if (label) label.textContent = frameText;
    const play = panel.querySelector<HTMLButtonElement>("[data-testid=sim-play]");
    if (play) play.textContent = ctx.view.isPlaying ? "Pause" : "Play";
  };

  const select = (id: string, value: string, choices: { name: string; label: string }[], on: (v: string) => void, title: string): HTMLSelectElement => {
    const s = h("select", { "data-testid": id, title }, ...choices.map((c) => h("option", { value: c.name, selected: c.name === value }, c.label)));
    s.addEventListener("change", () => on(s.value));
    return s;
  };

  const render = (): void => {
    const running = job?.status === "queued" || job?.status === "running";
    const canRun = !!info?.solver.available && ctx.stitched() !== null && !running;
    const btn = (text: string, id: string, fn: () => void, enabled: boolean, title = ""): HTMLButtonElement =>
      h("button", { class: "btn small", "data-testid": id, disabled: !enabled, title, onclick: fn }, text);
    const head = h(
      "div",
      { class: "assembly-head" },
      h("strong", {}, "Simulation"),
      h("span", { class: "spacer" }),
      btn("Simulate", "run-simulation", () => void simulate(), canRun,
        ctx.stitched() ? "Sew and drape the stitched layout with the cloth solver (GPU, takes minutes)" : "Stitch the layout first"),
      ...(running ? [btn("Cancel", "cancel-simulation", () => void cancel(), true)] : []),
    );
    const controls = h(
      "div",
      { class: "sim-row" },
      select("sim-fabric", opts.fabric, info?.fabrics.map((f) => ({ name: f.name, label: f.label })) ?? [], (v) => (opts = { ...opts, fabric: v }), "Fabric (measured weight, bending and stretch)"),
      select("sim-quality", opts.quality, info?.qualities.map((q) => ({ name: q.name, label: `${q.name} (${q.mesh_mm} mm)` })) ?? [], (v) => (opts = { ...opts, quality: v }), "Simulation mesh size"),
      h("label", { title: "Frames at 60 fps" }, "frames ",
        (() => {
          const i = h("input", { type: "number", min: 10, max: 1200, step: 10, value: opts.frames, "data-testid": "sim-frames" });
          i.addEventListener("change", () => (opts = { ...opts, frames: Math.max(10, Math.min(1200, Number(i.value) || 240)) }));
          return i;
        })()),
    );
    const lines: (HTMLElement | string)[] = [];
    if (info && !info.solver.available) lines.push(h("div", { class: "warn" }, `Cloth solver not found: ${info.solver.message}`));
    if (error) lines.push(h("div", { class: "warn", "data-testid": "sim-error" }, error));
    if (job) {
      const s = job.summary;
      let text = `${job.status}: ${job.stage}`;
      if (running && job.frame >= 0) text += ` - frame ${job.frame}/${job.frames}${job.gap_max_m !== null ? `, largest seam gap ${(job.gap_max_m * 100).toFixed(1)} cm` : ""}`;
      if (job.status === "failed") text += ` - ${job.error?.message ?? ""}`;
      if (s) {
        text = `${s.pass ? "PASS" : "FAIL"}: ${s.frames} frames${s.wall_s ? ` in ${(s.wall_s / 60).toFixed(1)} min` : ""}${job.cached ? " (cached)" : ""} - seams closed to ${s.seam_gap_max_cm.toFixed(2)} cm, stretch max ${s.stretch_max_pct ?? "?"} %, ${s.inside_body ?? "?"} vertices inside the body`;
      }
      lines.push(h("div", { class: s ? (s.pass ? "meta ok" : "warn") : "meta", "data-testid": "sim-status" }, text));
      if (running) lines.push(h("progress", { class: "arr-progress", max: 1, value: job.progress }));
      if (s?.fails.length) lines.push(h("ul", { class: "problems" }, ...s.fails.map((f) => h("li", {}, f))));
    }
    const view: HTMLElement[] = [];
    if (ctx.view.loaded) {
      view.push(
        h("div", { class: "sim-row" },
          btn(draped ? "Show placed" : "Show draped", "sim-toggle", () => setDraped(!draped), true),
          h("label", { class: "toggle" },
            (() => {
              const c = h("input", { type: "checkbox", checked: pieceColours, "data-testid": "sim-colours" });
              c.addEventListener("change", () => {
                pieceColours = c.checked;
                ctx.view.setPieceColours(pieceColours);
              });
              return c;
            })(), "piece colours"),
          ...(job?.status === "done"
            ? [
                h("a", { class: "btn small", href: simulationDownloadUrl(job.id, "obj"), download: "", "data-testid": "sim-obj" }, "OBJ"),
                h("a", { class: "btn small", href: simulationDownloadUrl(job.id, "zip"), download: "", "data-testid": "sim-zip", title: "Result, draped OBJ and images" }, "Zip"),
              ]
            : []),
        ),
      );
      if (ctx.view.frameCount > 1) {
        const slider = h("input", { type: "range", min: 0, max: ctx.view.frameCount - 1, step: 1, value: ctx.view.frameCount - 1, "data-testid": "sim-frame" });
        slider.addEventListener("input", () => {
          ctx.view.play(false);
          ctx.view.showIndex(Number(slider.value));
        });
        view.push(
          h("div", { class: "sim-row" },
            btn(ctx.view.isPlaying ? "Pause" : "Play", "sim-play", () => {
              ctx.view.play(!ctx.view.isPlaying);
              render();
            }, true),
            slider,
            h("span", { class: "meta", "data-testid": "sim-frame-label" }, frameText),
          ),
        );
      }
    }
    panel.replaceChildren(head, controls, ...lines, ...view);
  };
  render();
  return { refresh: render, hide: () => draped && setDraped(false) };
}
