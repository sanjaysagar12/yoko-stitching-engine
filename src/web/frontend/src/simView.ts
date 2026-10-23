// The simulated garment: every placed piece's own mesh (its faces, vertex for vertex), moved to the cloth solver's
// positions. The backend sends the embedding once (per placed-mesh vertex: three simulation vertices and weights) and
// then frames of the simulation mesh; the placed position of a vertex is sum_k w_k * x[idx_k]. Display only.
import { BufferAttribute, BufferGeometry, Color, DoubleSide, Group, Mesh, MeshStandardMaterial, Object3D } from "three";
import type { SimulationEmbedding } from "./api";

export const FABRIC_COLOUR = "#cfdcef"; // a light blue shirting

interface SimPiece {
  name: string;
  idx: Uint16Array | Uint32Array;
  w: Float32Array;
  mesh: Mesh;
}

export function decodeF32(b64: string): Float32Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return new Float32Array(out.buffer);
}

function decodeIdx(b64: string, dtype: "uint16" | "uint32"): Uint16Array | Uint32Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return dtype === "uint16" ? new Uint16Array(out.buffer) : new Uint32Array(out.buffer);
}

/** Positions of the placed-mesh vertices (`count` x 3) from the simulation mesh `x` (sim vertices x 3). */
export function embed(idx: ArrayLike<number>, w: ArrayLike<number>, x: ArrayLike<number>, out: Float32Array): Float32Array {
  const n = out.length / 3;
  for (let i = 0; i < n; i++) {
    let px = 0;
    let py = 0;
    let pz = 0;
    for (let k = 0; k < 3; k++) {
      const j = 3 * i + k;
      const v = 3 * (idx[j] as number);
      const wk = w[j] as number;
      px += wk * (x[v] as number);
      py += wk * (x[v + 1] as number);
      pz += wk * (x[v + 2] as number);
    }
    out[3 * i] = px;
    out[3 * i + 1] = py;
    out[3 * i + 2] = pz;
  }
  return out;
}

export class SimView {
  readonly group = new Group();
  private pieces: SimPiece[] = [];
  private frames: Float32Array | null = null;
  private frameIds: number[] = [];
  private vertices = 0;
  private index = 0;
  private playing = false;
  private clock = 0;
  private readonly material = new MeshStandardMaterial({ color: new Color(FABRIC_COLOUR), roughness: 0.92, metalness: 0, side: DoubleSide });
  /** Called when the shown frame changes (the panel's slider). */
  onFrame: ((index: number, frame: number) => void) | null = null;

  constructor(parent: Object3D) {
    this.group.name = "simulation";
    this.group.visible = false;
    parent.add(this.group);
  }

  get loaded(): boolean {
    return this.pieces.length > 0;
  }

  get frameCount(): number {
    return this.frameIds.length;
  }

  /** Build the draped meshes: the embedding plus each piece's placed faces (missing pieces are skipped). */
  setEmbedding(emb: SimulationEmbedding, faces: (piece: string) => Uint32Array | undefined, colour: (piece: string) => Color | null): void {
    this.clear();
    this.vertices = emb.sim_vertices;
    for (const p of emb.pieces) {
      const f = faces(p.name);
      if (!f) continue;
      const g = new BufferGeometry();
      g.setAttribute("position", new BufferAttribute(new Float32Array(p.count * 3), 3));
      g.setIndex(new BufferAttribute(f, 1));
      const own = colour(p.name);
      const mat = own ? this.material.clone() : this.material;
      if (own) mat.color.copy(own);
      const mesh = new Mesh(g, mat);
      mesh.name = `sim-${p.name}`;
      mesh.userData = { pieceColour: own?.getHex() ?? null };
      this.group.add(mesh);
      this.pieces.push({ name: p.name, idx: decodeIdx(p.idx, emb.idx_dtype), w: decodeF32(p.w), mesh });
    }
    this.show(decodeF32(emb.start));
  }

  /** One fabric colour for the whole garment, or each piece in its own colour (as in the layout). */
  setPieceColours(on: boolean): void {
    for (const p of this.pieces) {
      const own = (p.mesh.userData as { pieceColour: number | null }).pieceColour;
      const mat = p.mesh.material as MeshStandardMaterial;
      if (mat !== this.material) mat.color.set(on && own !== null ? own : FABRIC_COLOUR);
    }
  }

  /** Show the simulation mesh positions `x` (sim vertices x 3, metres). */
  show(x: Float32Array): void {
    if (x.length !== this.vertices * 3) return;
    for (const p of this.pieces) {
      const g = p.mesh.geometry;
      const pos = g.getAttribute("position") as BufferAttribute;
      embed(p.idx, p.w, x, pos.array as Float32Array);
      pos.needsUpdate = true;
      g.computeVertexNormals();
      g.computeBoundingSphere();
    }
  }

  /** Frames of the animation (T x sim vertices x 3) and their frame numbers; shows the last one. */
  setFrames(x: Float32Array, ids: number[]): void {
    this.frames = x;
    this.frameIds = ids;
    this.showIndex(ids.length - 1);
  }

  showIndex(i: number): void {
    if (!this.frames || this.frameIds.length === 0) return;
    this.index = Math.max(0, Math.min(this.frameIds.length - 1, i));
    const n = this.vertices * 3;
    this.show(this.frames.subarray(this.index * n, (this.index + 1) * n));
    this.onFrame?.(this.index, this.frameIds[this.index] ?? 0);
  }

  play(on: boolean): void {
    this.playing = on && this.frameIds.length > 1;
    if (this.playing && this.index >= this.frameIds.length - 1) this.showIndex(0);
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  /** Advance the animation (15 kept frames per second: real time for every 4th frame at 60 fps). */
  update(dt: number): void {
    if (!this.playing) return;
    this.clock += dt;
    if (this.clock < 1 / 15) return;
    this.clock = 0;
    if (this.index >= this.frameIds.length - 1) {
      this.playing = false;
      this.onFrame?.(this.index, this.frameIds[this.index] ?? 0);
      return;
    }
    this.showIndex(this.index + 1);
  }

  clear(): void {
    for (const p of this.pieces) {
      this.group.remove(p.mesh);
      p.mesh.geometry.dispose();
      if (p.mesh.material !== this.material) (p.mesh.material as MeshStandardMaterial).dispose();
    }
    this.pieces = [];
    this.frames = null;
    this.frameIds = [];
    this.playing = false;
  }
}
