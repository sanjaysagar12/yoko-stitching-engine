// Application wiring: jobs -> build -> avatar, points, pieces -> select piece and point -> /api/align -> animate.
// Half pieces: clone & mirror makes a linked derived piece; pairs are placed through /api/pair, never computed here.
// Copies (clone, no mirror) are ordinary pieces: placed through /api/align like any other, never as a pair.
import "./style.css";
import { Box3, Vector2, Vector3 } from "three";
import {
  align,
  arrange as arrangeApi,
  ApiFailure,
  buildStatus,
  clonePiece,
  copyPiece,
  exportUrl,
  fetchBundle,
  fileUrl,
  listJobs,
  loadPlacements,
  pairPlacements,
  savePlacements,
  startBuild,
} from "./api";
import { AvatarView } from "./avatarView";
import { formatMm, mirrorName, nearestPiece, VIOLATION_FRAME_M, withDependents } from "./lib";
import { Occlusion } from "./occlusion";
import { PiecesView } from "./piecesView";
import { PointsView } from "./pointsView";
import { SceneKit } from "./scene";
import { StitchView } from "./stitchView";
import { defaultPlacement, Store } from "./state";
import type { AlignResult, ApiError, ArrangeOptions, BuildResult, PairMode, PieceBundle, PieceInfo, Placement, Stitch } from "./types";
import type { Actions, CloneRequest, CopyRequest } from "./ui/actions";
import { mountArrange } from "./ui/arrange";
import { mountCloneDialog, mountCopyDialog } from "./ui/cloneDialog";
import { mountControls } from "./ui/controls";
import { mountErrors } from "./ui/errors";
import { mountInspector } from "./ui/inspector";
import { mountMeasurements } from "./ui/measurements";
import { mountPieceList } from "./ui/pieceList";
import { mountStatus } from "./ui/status";
import { mountTopbar } from "./ui/topbar";
import { mountViewbar } from "./ui/viewbar";

declare global {
  interface Window {
    /** Read-only helpers for the end-to-end test. */
    __fit?: {
      pointClient(name: string): { x: number; y: number } | null;
      pointInfo(name: string): { x: number; y: number; hidden: boolean; pickable: boolean; filtered: boolean } | null;
      ready(): boolean;
      edgeClient(piece: string, edgeId: string): { x: number; y: number } | null;
      arrangedEdges(): Array<{ piece: string; ids: string[] }>;
      selection(): unknown;
      arrangeState(): { arranged: boolean; rings: number; violations: number; hidden: string[]; meshLines: boolean };
      status(): { visible: boolean; text: string; valid: boolean | null; className: string };
      state(): {
        selectedPiece: string | null;
        pendingPoint: string | null;
        pendingPair: string | null;
        placements: Placement[];
        derived: string[];
        xray: boolean;
      };
    };
  }
}

const must = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const store = new Store();
const kit = new SceneKit(must("canvas-host"));
const avatar = new AvatarView(kit.scene);
const points = new PointsView(kit.scene, kit, must("overlay"));
const pieces = new PiecesView(kit.scene);
const occlusion = new Occlusion(kit);
const stitchView = new StitchView(kit.scene);
kit.onFrame((dt) => pieces.update(dt));

let buildToken = 0;
let alignSeq = 0;
let alignTimer: number | undefined;
let occTimer: number | undefined;
const bundles = new Map<string, Promise<PieceBundle>>();

const fail = (e: unknown): void => {
  const err: ApiError = e instanceof ApiFailure ? e.error : { code: "client_error", message: e instanceof Error ? e.message : String(e) };
  store.set({ error: err, busy: null, pendingPair: null });
};

const getBundle = (info: PieceInfo): Promise<PieceBundle> => {
  const id = store.get().buildId ?? "";
  const key = `${id}/${info.bundle ?? info.name}`;
  let p = bundles.get(key);
  if (!p) {
    p = fetchBundle(id, info.bundle as string);
    bundles.set(key, p);
  }
  return p;
};

const bvOf = (point: string): string | undefined => store.get().build?.avatar.points.find((p) => p.name === point)?.bv;

// -- occlusion of arrangement points (display only) ---------------------------------------------------------------------
function refreshOcclusion(): void {
  window.clearTimeout(occTimer);
  occTimer = window.setTimeout(() => store.set({ occluded: occlusion.hidden(points.positions()) }), 80);
}
kit.onCameraChange(refreshOcclusion);

const actions: Actions = {
  selectJob(name) {
    void buildJob(name);
  },
  selectPiece(name) {
    store.set({ selectedPiece: name });
    const s = store.get();
    if (name && s.pendingPoint) {
      store.set({ pendingPoint: null });
      place(name, s.pendingPoint);
    }
  },
  choosePoint(point) {
    const s = store.get();
    if (s.pendingPair) {
      const { source, derivedId, mode } = s.pendingPair;
      store.set({ pendingPair: null });
      void placePairAt(source, derivedId, mode, point);
    } else if (s.selectedPiece) place(s.selectedPiece, point);
    else store.set({ pendingPoint: point });
  },
  updatePlacement(piece, patch) {
    const cur = store.get().placements.get(piece);
    if (!cur) return;
    const next: Placement = { ...cur, ...patch };
    // a changed piece is solved again against the others (which stay where they are); locking keeps its place
    if (Object.keys(patch).some((k) => k !== "locked")) delete next.solved;
    store.get().placements.set(piece, next);
    store.touch("placements");
    scheduleAlign(120);
  },
  removePlacement(piece) {
    const s = store.get();
    s.placements.delete(piece);
    s.placed.delete(piece);
    pieces.clear(piece);
    store.touch("placements", "placed");
    scheduleAlign(0);
  },
  setReference(piece, name) {
    const s = store.get();
    if (s.placements.has(piece)) return actions.updatePlacement(piece, { reference: name });
    s.referenceDraft.set(piece, name);
    store.touch("referenceDraft");
  },
  openClone(piece) {
    openCloneDialog(piece);
  },
  cloneAndPair(req) {
    return cloneAndPair(req);
  },
  openCopy(piece) {
    openCopyDialog(piece);
  },
  copyPieces(req) {
    return copyPieces(req);
  },
  deleteDerived(id) {
    const s = store.get();
    const gone = withDependents(s.derived, id); // copies made from this clone go with it
    store.set({ derived: s.derived.filter((d) => !gone.has(d.id)), selectedPiece: s.selectedPiece && gone.has(s.selectedPiece) ? null : s.selectedPiece });
    for (const g of gone) {
      s.placements.delete(g);
      s.placed.delete(g);
      s.derivedInfos.delete(g);
      s.referenceDraft.delete(g);
      pieces.remove(g);
    }
    store.touch("placements", "placed", "derivedInfos", "derived");
    scheduleAlign(0);
  },
  placePair(source, derivedId, mode) {
    void placePair(source, derivedId, mode);
  },
  hoverPair(source, mode) {
    const point = store.get().placements.get(source)?.point;
    const known = new Set(store.get().build?.avatar.points.map((p) => p.name) ?? []);
    const twin = point ? mirrorName(point, known) : null;
    points.setHighlights(mode && point ? [point, ...(twin ? [twin] : [])] : []);
  },
  save() {
    void saveAll();
  },
  load() {
    void loadSaved();
  },
  exportScene(format) {
    void saveAll().then(() => {
      const id = store.get().buildId;
      if (id) window.location.assign(exportUrl(id, format));
    });
  },
  viewAvatar() {
    kit.frame(avatar.bounds());
  },
  viewPieces() {
    const b = pieces.layoutBounds();
    kit.frame(b.isEmpty() ? new Box3().setFromObject(pieces.root) : b);
  },
  viewSide(side) {
    kit.view(side, avatar.bounds());
  },
  arrange() {
    void runArrange();
  },
  resetArrange() {
    resetArrange();
  },
  setArrangeOptions(patch: Partial<ArrangeOptions>) {
    store.set({ arrangeOptions: { ...store.get().arrangeOptions, ...patch } });
    forgetSolved(); // other clearances or gaps: every piece is solved again, in the order it was placed
    scheduleAlign(500);
  },
  setMeshLines(on) {
    store.set({ showMeshLines: on });
  },
  toggleHidden(piece) {
    const next = new Set(store.get().hiddenPieces);
    if (!next.delete(piece)) next.add(piece);
    store.set({ hiddenPieces: next });
  },
  focusViolation(index) {
    const v = store.get().lastAlign?.validity?.violations[index];
    if (!v) return;
    store.set({ highlight: v.pieces });
    if (v.at_m) kit.focus(new Vector3(...v.at_m), VIOLATION_FRAME_M);
    else {
      const b = pieces.boundsOf(v.pieces);
      if (!b.isEmpty()) kit.focus(b.getCenter(new Vector3()), Math.max(b.getSize(new Vector3()).length() / 2, 0.2));
    }
  },
};

// -- placing -------------------------------------------------------------------------------------------------------------
function newPlacement(piece: string, point: string): Placement {
  const ref = store.get().referenceDraft.get(piece) ?? "bbox_centre";
  return defaultPlacement(piece, point, bvOf(point), ref);
}

function place(piece: string, point: string): void {
  const s = store.get();
  const cur = s.placements.get(piece);
  const next = cur ? { ...cur, point } : newPlacement(piece, point);
  delete next.solved;
  s.placements.set(piece, next);
  store.set({ selectedPiece: piece, pendingPoint: null });
  store.touch("placements");
  scheduleAlign(0);
}

/** Place a piece and its clone as a pair; the backend (alignment engine) decides where both go. */
async function placePair(source: string, derivedId: string, mode: PairMode, point?: string): Promise<void> {
  const s = store.get();
  if (!s.buildId) return;
  const cur = s.placements.get(source);
  const base = cur ? (point ? { ...cur, point } : cur) : point ? newPlacement(source, point) : null;
  if (!base) return;
  try {
    const res = await pairPlacements(s.buildId, base, derivedId, mode, s.derived);
    for (const p of res.placements) s.placements.set(p.piece, p);
    store.set({ error: null, pendingPair: null });
    if (res.warnings?.length) {
      // e.g. REFERENCE_ADJUSTED: the pair was anchored on a point of the mirror axis
      store.set({ busy: { text: res.warnings.join(" "), progress: 1 } });
      window.setTimeout(() => store.set({ busy: null }), 4000);
    }
    store.touch("placements");
    await runAlign();
  } catch (e) {
    fail(e);
  }
}

const placePairAt = (source: string, derivedId: string, mode: PairMode, point: string): Promise<void> =>
  placePair(source, derivedId, mode, point);

function scheduleAlign(delayMs: number): void {
  window.clearTimeout(alignTimer);
  alignTimer = window.setTimeout(() => void runAlign(), delayMs);
}

/** Load the meshes of derived pieces the backend reports (after loading saved placements, for instance). */
async function ensureDerived(res: AlignResult): Promise<void> {
  const s = store.get();
  if (!s.buildId) return;
  for (const d of res.derived_pieces) {
    if (s.derivedInfos.has(d.id)) continue;
    const b = await fetchBundle(s.buildId, `derived/${d.bundle}`);
    const info: PieceInfo = {
      name: d.id,
      status: "ok",
      source: d.source,
      ...(d.op === "copy" ? { copy_of: d.source } : { mirror_of: d.source, axis: d.axis }),
      width_cm: d.bbox_cm.width,
      height_cm: d.bbox_cm.height,
      obj: `derived/${d.obj}`,
      bundle: `derived/${d.bundle}`,
      named_points: b.named_points.map((q) => ({ name: q.name, on_outline: q.on_outline, xy_m: q.xy_m })),
      default_reference: b.default_reference.xy_m,
      suggested_stitch: d.suggested_stitch,
    };
    s.derivedInfos.set(d.id, info);
    await pieces.addPiece(s.buildId, info);
  }
  store.touch("derivedInfos");
}

/** The seam between a placed piece and its placed clone along the mirror edge (equal lengths by construction). */
function suggestedStitches(): Stitch[] {
  const s = store.get();
  const out: Stitch[] = [];
  for (const d of s.derived) {
    const st = s.derivedInfos.get(d.id)?.suggested_stitch;
    if (st && s.placements.has(d.id) && s.placements.has(d.source)) out.push(st);
  }
  return out;
}

/** Faces of the rolled meshes by mesh id: they never change for a piece, so the backend sends them once. */
const meshFaces = new Map<string, Uint32Array>();

function decodeBase64(text: string): ArrayBuffer {
  const bin = atob(text);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}

/** Typed arrays of the rolled meshes of an answer (compact transport, or plain arrays). */
function decodeMeshes(res: AlignResult): void {
  for (const r of res.placements) {
    const m = r.wrapped_mesh;
    if (m) {
      if (m.faces) meshFaces.set(m.mesh_id, new Uint32Array(decodeBase64(m.faces)));
      r.vertexArray = new Float32Array(decodeBase64(m.vertices));
      r.faceArray = meshFaces.get(m.mesh_id);
    } else if (r.wrapped_vertices && r.wrapped_faces) {
      r.vertexArray = new Float32Array(r.wrapped_vertices.flat());
      r.faceArray = new Uint32Array(r.wrapped_faces.flat());
    }
  }
}

/** Show an answer: every piece where the engine put it, and remember that place (`solved`) for the next request. */
async function showResult(res: AlignResult, arranged: boolean): Promise<void> {
  const s = store.get();
  await ensureDerived(res);
  decodeMeshes(res);
  for (const r of res.placements) {
    s.placed.set(r.piece, r);
    pieces.apply(r, !arranged, arranged);
    const pl = s.placements.get(r.piece);
    if (pl && r.solved) s.placements.set(r.piece, { ...pl, solved: r.solved });
  }
  stitchView.setSegments(res.stitch_reports.flatMap((r) => (r.segments ? [r.segments] : [])));
}

/** Every piece is solved again from its anchor (in placement order) on the next request. */
function forgetSolved(): void {
  const s = store.get();
  for (const [k, pl] of s.placements) {
    if (pl.solved) {
      const next = { ...pl };
      delete next.solved;
      s.placements.set(k, next);
    }
  }
}

async function runAlign(): Promise<void> {
  const s = store.get();
  if (!s.buildId) return;
  const list = [...s.placements.values()];
  if (list.length === 0 && s.derived.length === 0) return;
  const seq = ++alignSeq;
  try {
    const res = await align(s.buildId, list, s.derived, suggestedStitches(), s.arrangeOptions, [...meshFaces.keys()]);
    if (seq !== alignSeq) return; // a newer request is on its way
    await showResult(res, s.arranged);
    store.set({ lastAlign: res, arrangement: res.arrangement ?? null, error: null });
    store.touch("placed", "placements");
  } catch (e) {
    if (seq === alignSeq) fail(e);
  }
}

// -- auto-arrange --------------------------------------------------------------------------------------------------------
async function runArrange(): Promise<void> {
  const s = store.get();
  if (!s.buildId) return;
  forgetSolved(); // a fresh arrangement of everything that is not locked
  const list = [...s.placements.values()];
  if (list.length === 0) return;
  const seq = ++alignSeq;
  store.set({ busy: { text: "arranging around the body...", progress: 0.5 }, error: null });
  try {
    const res = await arrangeApi(s.buildId, list, s.derived, s.arrangeOptions, suggestedStitches(), [...meshFaces.keys()]);
    if (seq !== alignSeq) return;
    await showResult(res, true);
    store.set({ lastAlign: res, arrangement: res.arrangement ?? null, arranged: true, busy: null, error: null, selection: null });
    store.touch("placed", "placements");
  } catch (e) {
    if (seq === alignSeq) fail(e);
  }
}

/** Leave the arrangement view and place every piece again from its anchor, in the order the pieces were placed. */
function resetArrange(): void {
  pieces.endArrangement();
  forgetSolved();
  store.set({ arranged: false, selection: null, highlight: [], hoverText: null });
  void runAlign();
}

// -- clones --------------------------------------------------------------------------------------------------------------
async function cloneAndPair(req: CloneRequest): Promise<void> {
  const s = store.get();
  if (!s.buildId) return;
  try {
    const info = await clonePiece(s.buildId, req.source, req.axis, req.id);
    s.derived.push({ id: req.id, source: req.source, op: "mirror", axis: req.axis });
    s.derivedInfos.set(req.id, info);
    await pieces.addPiece(s.buildId, info);
    store.set({ error: null });
    store.touch("derived", "derivedInfos");
  } catch (e) {
    return fail(e);
  }
  if (!req.pairNow) return;
  if (s.placements.has(req.source)) await placePair(req.source, req.id, req.mode);
  else if (s.pendingPoint) {
    const point = s.pendingPoint;
    store.set({ pendingPoint: null });
    await placePair(req.source, req.id, req.mode, point);
  } else store.set({ pendingPair: { source: req.source, derivedId: req.id, mode: req.mode } });
}

/** Exact copies (ordinary pieces). Optionally the first goes on the counterpart of the source's side point with the
 * same placement and rotation (a copy, not a mirror); otherwise it is selected and waits for a point click. */
async function copyPieces(req: CopyRequest): Promise<void> {
  const s = store.get();
  if (!s.buildId || req.ids.length === 0) return;
  try {
    for (const id of req.ids) {
      const info = await copyPiece(s.buildId, req.source, id, s.derived);
      s.derived.push({ id, source: req.source, op: "copy" });
      s.derivedInfos.set(id, info);
      await pieces.addPiece(s.buildId, info);
    }
    store.set({ error: null });
    store.touch("derived", "derivedInfos");
  } catch (e) {
    return fail(e);
  }
  const first = req.ids[0] as string;
  const src = s.placements.get(req.source);
  const known = new Set(s.build?.avatar.points.map((p) => p.name) ?? []);
  const twin = src ? mirrorName(src.point, known) : null;
  if (req.placeAtCounterpart && src && twin) {
    const next: Placement = { ...src, piece: first, point: twin };
    delete next.solved;
    delete next.locked;
    s.placements.set(first, next);
    store.set({ selectedPiece: first });
    store.touch("placements");
    scheduleAlign(0);
  } else actions.selectPiece(first); // placed on the next point click (or on the point chosen before)
}

// -- save and load ----------------------------------------------------------------------------------------------------------
async function saveAll(): Promise<void> {
  const s = store.get();
  if (!s.job) return;
  try {
    await savePlacements(s.job, [...s.placements.values()], s.derived);
    store.set({ busy: { text: "placements saved", progress: 1 } });
    window.setTimeout(() => store.set({ busy: null }), 1500);
  } catch (e) {
    fail(e);
  }
}

async function loadSaved(): Promise<void> {
  const s = store.get();
  if (!s.job || !s.build) return;
  try {
    const saved = await loadPlacements(s.job);
    const savedDerived = saved.derived_pieces ?? [];
    for (const name of [...s.placements.keys()]) {
      pieces.clear(name, false);
      s.placed.delete(name);
    }
    s.placements.clear();
    for (const d of s.derived) {
      if (!savedDerived.some((x) => x.id === d.id)) {
        pieces.remove(d.id);
        s.derivedInfos.delete(d.id);
      }
    }
    store.set({ derived: savedDerived });
    const known = new Set([...s.build.pieces.filter((p) => p.status === "ok").map((p) => p.name), ...savedDerived.map((d) => d.id)]);
    for (const p of saved.placements) if (known.has(p.piece)) s.placements.set(p.piece, p);
    store.touch("placements", "placed", "derived");
    await runAlign();
  } catch (e) {
    fail(e);
  }
}

// -- builds ---------------------------------------------------------------------------------------------------------------
async function buildJob(name: string): Promise<void> {
  const token = ++buildToken;
  bundles.clear();
  store.set({
    job: name, build: null, buildId: null, error: null, selectedPiece: null, pendingPoint: null, pendingPair: null,
    placements: new Map(), placed: new Map(), derived: [], derivedInfos: new Map(), referenceDraft: new Map(),
    busy: { text: "starting build...", progress: 0.02 },
  });
  try {
    const started = await startBuild(name);
    for (;;) {
      const st = await buildStatus(started.build_id);
      if (token !== buildToken) return;
      if (st.status === "failed") {
        store.set({ error: st.error ?? { code: "build_failed", message: "build failed" }, busy: null });
        return;
      }
      if (st.status === "done" && st.result) {
        await show(st.result, token);
        return;
      }
      store.set({ busy: { text: `building: ${st.stage}`, progress: Math.max(0.05, st.progress) } });
      await new Promise((r) => setTimeout(r, 350));
    }
  } catch (e) {
    if (token === buildToken) fail(e);
  }
}

async function show(result: BuildResult, token: number): Promise<void> {
  store.set({ busy: { text: "loading meshes...", progress: 0.97 } });
  await Promise.all([avatar.load(fileUrl(result.id, result.avatar.mesh), store.get().avatarOpacity), pieces.load(result.id, result.pieces)]);
  if (token !== buildToken) return;
  points.setPoints(result.avatar.points);
  avatar.setVolumes(result.avatar.volumes);
  occlusion.setSurface(avatar.surface);
  store.set({ buildId: result.id, build: result, busy: null });
  actions.viewAvatar();
  refreshOcclusion();
}

// -- user interaction on the canvas ----------------------------------------------------------------------------------------
const canvas = kit.renderer.domElement;
let down: { x: number; y: number } | null = null;
const local = (ev: PointerEvent): { x: number; y: number } => {
  const r = canvas.getBoundingClientRect();
  return { x: ev.clientX - r.left, y: ev.clientY - r.top };
};
canvas.addEventListener("pointerdown", (ev) => {
  down = local(ev);
});
canvas.addEventListener("pointerup", (ev) => {
  const p = local(ev);
  const moved = down ? Math.hypot(p.x - down.x, p.y - down.y) : 0;
  down = null;
  if (moved > 5) return; // that was an orbit drag
  const r = canvas.getBoundingClientRect();
  const ndc = new Vector2((p.x / r.width) * 2 - 1, -(p.y / r.height) * 2 + 1);
  if (store.get().arranged) {
    const f = pieces.pickFeature(ndc, kit.camera);
    if (f) {
      store.set({
        selection:
          f.kind === "edge"
            ? { kind: "edge", piece: f.piece, id: f.edge.id, from: f.edge.from, to: f.edge.to, length_m: f.edge.length_m }
            : { kind: "point", piece: f.piece, name: f.point.name, world_m: f.point.world_m },
      });
      return;
    }
  }
  const point = points.pick(p.x, p.y);
  if (point) return actions.choosePoint(point);
  const hit = pieces.pick(ndc, kit.camera);
  if (hit) actions.selectPiece(store.get().selectedPiece === hit ? null : hit);
});
canvas.addEventListener("pointermove", (ev) => {
  const p = local(ev);
  const hover = points.pick(p.x, p.y);
  if (hover !== store.get().hoverPoint) store.set({ hoverPoint: hover });
  canvas.style.cursor = hover ? "pointer" : "";
  if (store.get().arranged) {
    const r = canvas.getBoundingClientRect();
    const over = pieces.pick(new Vector2((p.x / r.width) * 2 - 1, -(p.y / r.height) * 2 + 1), kit.camera);
    const near = over ? nearestPiece(store.get().placed.get(over)?.arrange?.min_distance_to_pieces_m ?? {}) : null;
    const text = over && near ? `${over}: nearest piece ${near.name} ${formatMm(near.distance_m)} mm` : null;
    if (text !== store.get().hoverText) store.set({ hoverText: text });
  }
});

// -- keep the views in sync with the state ----------------------------------------------------------------------------------
store.subscribe((s, changed) => {
  if (changed.includes("selectedPiece")) pieces.setSelected(s.selectedPiece);
  if (changed.includes("showLabels")) points.setLabelsVisible(s.showLabels);
  if (changed.includes("showVolumes")) avatar.setVolumesVisible(s.showVolumes);
  if (changed.includes("avatarOpacity")) avatar.setOpacity(s.avatarOpacity);
  if (changed.includes("hoverPoint")) points.setHover(s.hoverPoint);
  if (changed.includes("showMeshLines")) pieces.setMeshLines(s.showMeshLines);
  if (changed.includes("hiddenPieces")) pieces.setHidden(s.hiddenPieces);
  if (changed.includes("highlight")) pieces.setHighlight(s.highlight);
  if (changed.includes("selection")) pieces.setSelection(s.selection);
  if (changed.includes("xray")) points.setXray(s.xray);
  if (changed.includes("pointFilter")) points.setFilter(s.pointFilter);
  if (changed.includes("occluded")) points.setOccluded(s.occluded);
  if (["selectedPiece", "pendingPoint", "placements"].some((k) => changed.includes(k as never))) {
    const own = s.selectedPiece ? s.placements.get(s.selectedPiece)?.point : undefined;
    points.setSelected(s.pendingPoint ?? own ?? null);
  }
});

mountTopbar(must("topbar"), store, actions);
mountViewbar(must("viewbar"), store, actions);
mountPieceList(must("left"), store, actions);
mountArrange(must("left"), store, actions);
mountMeasurements(must("left"), store);
const inspectorHost = document.createElement("div");
inspectorHost.className = "inspector-host";
must("right").append(inspectorHost);
const inspector = mountInspector(inspectorHost, store, actions, getBundle);
mountControls(must("right"), store, actions);
mountErrors(must("view"), store);
mountStatus(must("view"), store, actions);
const openCloneDialog = mountCloneDialog(store, actions, inspector);
const openCopyDialog = mountCopyDialog(store, actions);

window.__fit = {
  pointClient(name) {
    const p = points.screenOf(name);
    if (!p) return null;
    const r = canvas.getBoundingClientRect();
    return { x: r.left + p.x, y: r.top + p.y };
  },
  pointInfo(name) {
    const i = points.info(name);
    if (!i) return null;
    const r = canvas.getBoundingClientRect();
    return { ...i, x: r.left + i.x, y: r.top + i.y };
  },
  ready: () => store.get().build !== null && store.get().busy === null,
  edgeClient(piece, edgeId) {
    const m = pieces.edgeMiddle(piece, edgeId);
    if (!m) return null;
    const p = kit.screenOf(m);
    const r = canvas.getBoundingClientRect();
    return { x: r.left + p.x, y: r.top + p.y };
  },
  arrangedEdges: () => pieces.arrangedEdges(),
  selection: () => store.get().selection,
  arrangeState: () => {
    const s = store.get();
    return {
      arranged: s.arranged,
      rings: s.arrangement?.rings.length ?? 0,
      violations: s.lastAlign?.validity?.violations.length ?? 0,
      hidden: [...s.hiddenPieces],
      meshLines: s.showMeshLines,
    };
  },
  status: () => {
    const el = document.querySelector<HTMLElement>("[data-testid=status-strip]");
    const v = store.get().lastAlign?.validity;
    return { visible: !!el && !el.hidden, text: el?.textContent ?? "", valid: v?.valid ?? null, className: el?.className ?? "" };
  },
  state: () => {
    const s = store.get();
    return {
      selectedPiece: s.selectedPiece,
      pendingPoint: s.pendingPoint,
      pendingPair: s.pendingPair ? `${s.pendingPair.source}+${s.pendingPair.derivedId}` : null,
      placements: [...s.placements.values()],
      derived: s.derived.map((d) => d.id),
      xray: s.xray,
    };
  },
};

listJobs()
  .then((jobs) => store.set({ jobs }))
  .catch(fail);

