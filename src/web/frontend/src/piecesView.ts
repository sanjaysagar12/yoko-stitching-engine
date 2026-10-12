// Pattern pieces: flat meshes laid out left to right beside the avatar, moved onto the avatar by backend answers.
// Display only: the matrices and wrapped vertices come from /api/align; nothing is scaled or recomputed here.
import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  Camera,
  Color,
  DoubleSide,
  Float32BufferAttribute,
  Group,
  Line,
  LineBasicMaterial,
  LineLoop,
  LineSegments,
  Matrix4,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  Points,
  PointsMaterial,
  Quaternion,
  Raycaster,
  Vector2,
  Vector3,
  WireframeGeometry,
} from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { fetchText, fileUrl } from "./api";
import { matrixFromRowMajor } from "./lib";
import type { PieceInfo, PlacedResult, Selection, WorldEdge, WorldPoint } from "./types";

const LAYOUT_START_M = 1.2; // right of the avatar's hand tips
const LAYOUT_GAP_M = 0.05;
const TWEEN_S = 0.45;
const ARRANGED_OPACITY = 0.62; // pieces are semi-transparent in an arrangement so pieces behind them and the body stay visible
const OUTLINE_COLOR = 0x111827;
const EDGE_COLOR = 0x374151;
const SELECT_COLOR = 0xea580c;
const HIGHLIGHT_COLOR = 0xdc2626;

/** What a click on an arranged piece found: an outline edge or a named point. */
export type Feature =
  | { kind: "edge"; piece: string; edge: WorldEdge }
  | { kind: "point"; piece: string; point: WorldPoint };

interface Tween {
  t: number;
  from: { p: Vector3; q: Quaternion; s: Vector3 };
  to: { p: Vector3; q: Quaternion; s: Vector3 };
  fade: boolean;
}

interface Entry {
  info: PieceInfo;
  mesh: Mesh;
  flat: BufferGeometry;
  layout: Matrix4;
  tween: Tween | null;
  placed: boolean;
  /** Arrangement display: outline, edges and named points in world space (hidden unless the piece is arranged). */
  decor: Group;
  arranged: boolean;
  outline: LineLoop | null;
  edges: Array<{ line: Line; edge: WorldEdge }>;
  points: Points | null;
  pointData: WorldPoint[];
  wire: LineSegments | null;
  marks: Object3D[];
}

const decompose = (m: Matrix4) => {
  const p = new Vector3();
  const q = new Quaternion();
  const s = new Vector3();
  m.decompose(p, q, s);
  return { p, q, s };
};

export class PiecesView {
  readonly root = new Group();
  private readonly entries = new Map<string, Entry>();
  private selected: string | null = null;
  private cursor = LAYOUT_START_M;
  private colorIndex = 0;
  private readonly ray = new Raycaster();

  constructor(parent: Object3D) {
    this.root.name = "pieces";
    parent.add(this.root);
  }

  /** Replace everything with the pieces of a build. */
  async load(buildId: string, pieces: PieceInfo[]): Promise<void> {
    for (const e of this.entries.values()) this.root.remove(e.mesh, e.decor);
    this.entries.clear();
    this.cursor = LAYOUT_START_M;
    this.colorIndex = 0;
    const ok = pieces.filter((p) => p.status === "ok" && p.obj);
    const texts = await Promise.all(ok.map((p) => fetchText(fileUrl(buildId, p.obj as string))));
    ok.forEach((info, i) => this.add(info, texts[i] as string));
  }

  /** Add one more piece (a derived clone) at the end of the layout row. */
  async addPiece(buildId: string, info: PieceInfo): Promise<void> {
    if (!info.obj || this.entries.has(info.name)) return;
    this.add(info, await fetchText(fileUrl(buildId, info.obj)));
  }

  private add(info: PieceInfo, objText: string): void {
    const group = new OBJLoader().parse(objText);
    let src: Mesh | null = null;
    group.traverse((o) => {
      if (!src && (o as Mesh).isMesh) src = o as Mesh;
    });
    const flat = (src as Mesh | null)?.geometry;
    if (!flat) return;
    flat.computeVertexNormals();
    const mat = new MeshStandardMaterial({
      color: new Color().setHSL((this.colorIndex++ * 0.137) % 1, 0.55, 0.6),
      roughness: 0.9,
      side: DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const mesh = new Mesh(flat, mat);
    mesh.name = info.name;
    mesh.matrixAutoUpdate = false;
    const layout = new Matrix4().makeTranslation(this.cursor, 0, 0);
    this.cursor += (info.width_cm ?? 0) / 100 + LAYOUT_GAP_M;
    mesh.matrix.copy(layout);
    const decor = new Group();
    decor.name = `decor-${info.name}`;
    decor.visible = false;
    this.entries.set(info.name, {
      info, mesh, flat, layout, tween: null, placed: false, decor, arranged: false, outline: null, edges: [], points: null,
      pointData: [], wire: null, marks: [],
    });
    this.root.add(mesh, decor);
  }

  has(name: string): boolean {
    return this.entries.has(name);
  }

  /** Remove a piece (a deleted clone). */
  remove(name: string): void {
    const e = this.entries.get(name);
    if (!e) return;
    this.root.remove(e.mesh, e.decor);
    this.clearDecor(e);
    if (e.mesh.geometry !== e.flat) e.mesh.geometry.dispose();
    e.flat.dispose();
    this.entries.delete(name);
  }

  setSelected(name: string | null): void {
    this.selected = name;
    for (const [n, e] of this.entries) {
      const m = e.mesh.material as MeshStandardMaterial;
      m.emissive.set(this.highlighted.has(n) ? HIGHLIGHT_COLOR : n === name ? 0x2a4a7a : 0x000000);
    }
  }

  /** Piece under a pointer position in normalised device coordinates, or null. */
  pick(ndc: Vector2, camera: Camera): string | null {
    this.ray.setFromCamera(ndc, camera);
    const hits = this.ray.intersectObjects([...this.entries.values()].map((e) => e.mesh), false);
    return hits[0]?.object.name ?? null;
  }

  /** World centre of a piece (for tests and camera framing). */
  centre(name: string): Vector3 | null {
    const e = this.entries.get(name);
    if (!e) return null;
    return new Box3().setFromObject(e.mesh).getCenter(new Vector3());
  }

  layoutBounds(): Box3 {
    const b = new Box3();
    for (const e of this.entries.values()) {
      if (!e.placed) b.union(new Box3().setFromObject(e.mesh));
    }
    return b;
  }

  /** Apply a backend placement result. Rigid: tween the matrix. Wrapped: swap in the wrapped mesh and fade it in. */
  apply(r: PlacedResult, animate = true, arranged = false): void {
    const e = this.entries.get(r.piece);
    if (!e) return;
    e.placed = true;
    const mat = e.mesh.material as MeshStandardMaterial;
    if (arranged && r.vertexArray && r.faceArray) {
      this.applyArranged(e, r);
      return;
    }
    if (e.arranged) this.leaveArranged(e);
    if (r.vertexArray && r.faceArray) {
      const g = new BufferGeometry();
      g.setAttribute("position", new Float32BufferAttribute(r.vertexArray, 3));
      g.setIndex(new BufferAttribute(r.faceArray, 1));
      g.computeVertexNormals();
      if (e.mesh.geometry !== e.flat) e.mesh.geometry.dispose();
      e.mesh.geometry = g;
      e.mesh.matrix.identity();
      e.mesh.matrixWorldNeedsUpdate = true;
      e.tween = null;
      if (animate) {
        mat.transparent = true;
        mat.opacity = 0.15;
        e.tween = { t: 0, fade: true, from: decompose(new Matrix4()), to: decompose(new Matrix4()) };
      }
      return;
    }
    if (e.mesh.geometry !== e.flat) {
      e.mesh.geometry.dispose();
      e.mesh.geometry = e.flat;
    }
    mat.transparent = false;
    mat.opacity = 1;
    this.moveTo(e, matrixFromRowMajor(r.matrix), animate);
  }

  /** Put a piece back into the flat layout. */
  clear(name: string, animate = true): void {
    const e = this.entries.get(name);
    if (!e) return;
    if (e.arranged) this.leaveArranged(e);
    e.placed = false;
    if (e.mesh.geometry !== e.flat) {
      e.mesh.geometry.dispose();
      e.mesh.geometry = e.flat;
    }
    const mat = e.mesh.material as MeshStandardMaterial;
    mat.transparent = false;
    mat.opacity = 1;
    this.moveTo(e, e.layout, animate);
  }

  private moveTo(e: Entry, target: Matrix4, animate: boolean): void {
    if (!animate) {
      e.mesh.matrix.copy(target);
      e.mesh.matrixWorldNeedsUpdate = true;
      e.tween = null;
      return;
    }
    e.tween = { t: 0, fade: false, from: decompose(e.mesh.matrix), to: decompose(target) };
  }

  update(dt: number): void {
    for (const e of this.entries.values()) {
      const tw = e.tween;
      if (!tw) continue;
      tw.t = Math.min(1, tw.t + dt / TWEEN_S);
      const k = tw.t * tw.t * (3 - 2 * tw.t);
      if (tw.fade) {
        (e.mesh.material as MeshStandardMaterial).opacity = 0.15 + 0.85 * k;
      } else {
        const p = tw.from.p.clone().lerp(tw.to.p, k);
        const q = tw.from.q.clone().slerp(tw.to.q, k);
        const s = tw.from.s.clone().lerp(tw.to.s, k);
        e.mesh.matrix.compose(p, q, s);
        e.mesh.matrixWorldNeedsUpdate = true;
      }
      if (tw.t >= 1) {
        e.tween = null;
        if (tw.fade) {
          const m = e.mesh.material as MeshStandardMaterial;
          m.opacity = 1;
          m.transparent = false;
        }
      }
    }
  }

  // -- arrangement display ---------------------------------------------------------------------------------------------
  private showMesh = false;
  private hidden: ReadonlySet<string> = new Set();
  private highlighted: ReadonlySet<string> = new Set();

  /** Arranged piece: wrapped mesh from the backend, semi-transparent and double sided, with outline, edges and points. */
  private applyArranged(e: Entry, r: PlacedResult): void {
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(r.vertexArray ?? new Float32Array(0), 3));
    g.setIndex(new BufferAttribute(r.faceArray ?? new Uint32Array(0), 1));
    g.computeVertexNormals();
    if (e.mesh.geometry !== e.flat) e.mesh.geometry.dispose();
    e.mesh.geometry = g;
    e.mesh.matrix.identity();
    e.mesh.matrixWorldNeedsUpdate = true;
    e.tween = null;
    const mat = e.mesh.material as MeshStandardMaterial;
    mat.transparent = true;
    mat.opacity = ARRANGED_OPACITY;
    mat.depthWrite = false;
    mat.polygonOffsetFactor = 1; // the outline lines are drawn in front of the surface
    mat.polygonOffsetUnits = 1;
    mat.needsUpdate = true;
    e.arranged = true;
    this.clearDecor(e);
    if (r.world_outline_m && r.world_outline_m.length > 2) {
      const og = new BufferGeometry();
      og.setAttribute("position", new Float32BufferAttribute(r.world_outline_m.flat(), 3));
      e.outline = new LineLoop(og, new LineBasicMaterial({ color: OUTLINE_COLOR }));
      e.outline.name = `outline-${e.info.name}`;
      e.decor.add(e.outline);
    }
    for (const edge of r.world_edges ?? []) {
      const lg = new BufferGeometry();
      lg.setAttribute("position", new Float32BufferAttribute(edge.polyline_m.flat(), 3));
      const line = new Line(lg, new LineBasicMaterial({ color: EDGE_COLOR }));
      line.userData = { piece: e.info.name, edgeId: edge.id };
      e.edges.push({ line, edge });
      e.decor.add(line);
    }
    e.pointData = r.world_named_points ?? [];
    if (e.pointData.length > 0) {
      const pg = new BufferGeometry();
      pg.setAttribute("position", new Float32BufferAttribute(e.pointData.flatMap((p) => p.world_m), 3));
      e.points = new Points(pg, new PointsMaterial({ color: 0x1d4ed8, size: 4, sizeAttenuation: false }));
      e.points.name = `points-${e.info.name}`;
      e.decor.add(e.points);
    }
    if (this.showMesh) this.buildWire(e);
    this.applyFlags(e);
  }

  private leaveArranged(e: Entry): void {
    e.arranged = false;
    this.clearDecor(e);
    e.decor.visible = false;
    const mat = e.mesh.material as MeshStandardMaterial;
    mat.transparent = false;
    mat.opacity = 1;
    mat.depthWrite = true;
    mat.polygonOffsetFactor = -1;
    mat.polygonOffsetUnits = -1;
    mat.emissive.set(this.selected === e.info.name ? 0x2a4a7a : 0x000000);
    mat.needsUpdate = true;
    e.mesh.visible = true;
  }

  private clearDecor(e: Entry): void {
    for (const o of [...e.decor.children]) {
      e.decor.remove(o);
      const m = o as unknown as { geometry?: BufferGeometry; material?: { dispose(): void } };
      m.geometry?.dispose();
      m.material?.dispose();
    }
    e.outline = null;
    e.edges = [];
    e.points = null;
    e.pointData = [];
    e.wire = null;
    e.marks = [];
  }

  private buildWire(e: Entry): void {
    if (e.wire) return;
    e.wire = new LineSegments(
      new WireframeGeometry(e.mesh.geometry),
      new LineBasicMaterial({ color: 0x6b7280, transparent: true, opacity: 0.4 }),
    );
    e.wire.name = `mesh-lines-${e.info.name}`;
    e.decor.add(e.wire);
  }

  /** Interior triangle edges of the arranged pieces (off by default: the outlines are what matters). */
  setMeshLines(on: boolean): void {
    this.showMesh = on;
    for (const e of this.entries.values()) {
      if (!e.arranged) continue;
      if (on) this.buildWire(e);
      else if (e.wire) {
        e.decor.remove(e.wire);
        e.wire.geometry.dispose();
        (e.wire.material as LineBasicMaterial).dispose();
        e.wire = null;
      }
    }
  }

  /** Switch pieces off and on (an eye button in the list). */
  setHidden(names: ReadonlySet<string>): void {
    this.hidden = names;
    for (const e of this.entries.values()) this.applyFlags(e);
  }

  /** Emphasise pieces (the pieces of a violation the user clicked). */
  setHighlight(names: readonly string[]): void {
    this.highlighted = new Set(names);
    for (const e of this.entries.values()) this.applyFlags(e);
  }

  private applyFlags(e: Entry): void {
    const hide = this.hidden.has(e.info.name);
    e.mesh.visible = !hide;
    e.decor.visible = e.arranged && !hide;
    const lit = this.highlighted.has(e.info.name);
    const mat = e.mesh.material as MeshStandardMaterial;
    mat.emissive.set(lit ? HIGHLIGHT_COLOR : e.info.name === this.selected ? 0x2a4a7a : 0x000000);
    if (e.outline) (e.outline.material as LineBasicMaterial).color.set(lit ? HIGHLIGHT_COLOR : OUTLINE_COLOR);
  }

  /** The outline edge or named point under a pointer position (normalised device coordinates), or null. */
  pickFeature(ndc: Vector2, camera: Camera): Feature | null {
    this.ray.setFromCamera(ndc, camera);
    const prevLine = this.ray.params.Line;
    const prevPoints = this.ray.params.Points;
    this.ray.params.Line = { threshold: 0.007 };
    this.ray.params.Points = { threshold: 0.012 };
    try {
      const objs: Object3D[] = [];
      for (const e of this.entries.values()) {
        if (!e.arranged || !e.decor.visible) continue;
        if (e.points) objs.push(e.points);
        for (const x of e.edges) objs.push(x.line);
      }
      const hit = this.ray.intersectObjects(objs, false)[0];
      if (!hit) return null;
      if (hit.object instanceof Points) {
        const name = hit.object.name.replace(/^points-/, "");
        const point = this.entries.get(name)?.pointData[hit.index ?? 0];
        return point ? { kind: "point", piece: name, point } : null;
      }
      const piece = (hit.object.userData as { piece?: string }).piece;
      const e = piece ? this.entries.get(piece) : undefined;
      const found = e?.edges.find((x) => x.line === hit.object);
      return piece && found ? { kind: "edge", piece, edge: found.edge } : null;
    } finally {
      this.ray.params.Line = prevLine;
      this.ray.params.Points = prevPoints;
    }
  }

  /** Mark the picked edge or point (orange line plus markers); null clears the mark. */
  setSelection(sel: Selection | null): void {
    for (const e of this.entries.values()) {
      for (const m of e.marks) {
        e.decor.remove(m);
        (m as Points).geometry.dispose();
      }
      e.marks = [];
      for (const x of e.edges) (x.line.material as LineBasicMaterial).color.set(EDGE_COLOR);
    }
    if (!sel) return;
    const e = this.entries.get(sel.piece);
    if (!e) return;
    const mark = (flat: number[], size: number): void => {
      const pg = new BufferGeometry();
      pg.setAttribute("position", new Float32BufferAttribute(flat, 3));
      const pts = new Points(pg, new PointsMaterial({ color: SELECT_COLOR, size, sizeAttenuation: false, depthTest: false }));
      pts.renderOrder = 5;
      e.marks.push(pts);
      e.decor.add(pts);
    };
    if (sel.kind === "edge") {
      const x = e.edges.find((q) => q.edge.id === sel.id);
      if (!x) return;
      (x.line.material as LineBasicMaterial).color.set(SELECT_COLOR);
      mark(x.edge.polyline_m.flat(), 7);
    } else {
      mark([...sel.world_m], 12);
    }
  }

  /** World position of the middle of an arranged edge (tests and camera framing). */
  edgeMiddle(piece: string, edgeId: string): Vector3 | null {
    const x = this.entries.get(piece)?.edges.find((q) => q.edge.id === edgeId);
    if (!x) return null;
    const p = x.edge.polyline_m[Math.floor(x.edge.polyline_m.length / 2)];
    return p ? new Vector3(p[0], p[1], p[2]) : null;
  }

  /** The arranged pieces and the edge ids they offer. */
  arrangedEdges(): Array<{ piece: string; ids: string[] }> {
    return [...this.entries.values()]
      .filter((e) => e.arranged)
      .map((e) => ({ piece: e.info.name, ids: e.edges.map((q) => q.edge.id) }));
  }

  /** World bounding box of some pieces (to frame a violation that has no point). */
  boundsOf(names: readonly string[]): Box3 {
    const b = new Box3();
    for (const n of names) {
      const e = this.entries.get(n);
      if (e) b.union(new Box3().setFromObject(e.mesh));
    }
    return b;
  }

  isArranged(name: string): boolean {
    return this.entries.get(name)?.arranged ?? false;
  }

  /** Leave the arrangement display: every piece back to its plain material (the caller applies plain results). */
  endArrangement(): void {
    for (const e of this.entries.values()) if (e.arranged) this.leaveArranged(e);
  }

  selectedName(): string | null {
    return this.selected;
  }
}
