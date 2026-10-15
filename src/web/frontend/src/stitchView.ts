// Stitches in the viewer. `setSegments`: the alignment engine's suggested mirror seams (thin magenta pairs).
// `setSeams`: the stitcher's result: each seam's sewing lines on both pieces in its colour and a few correspondence
// lines between them; a selected seam is drawn bright, the others faded.
import { BufferGeometry, Color, Float32BufferAttribute, Group, LineBasicMaterial, LineSegments, Object3D } from "three";
import type { StitchSeam } from "./api";

export const SEAM_COLOURS = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
  "#3b4cc0", "#d35400", "#16a085", "#8e44ad", "#c0392b", "#2c3e50", "#7f8c00", "#b8860b",
];

export const seamColour = (i: number): string => SEAM_COLOURS[i % SEAM_COLOURS.length] ?? "#1f77b4";

function segments(pts: number[], colour: string, opacity: number): LineSegments {
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(pts, 3));
  const m = new LineBasicMaterial({ color: new Color(colour), depthTest: false, transparent: opacity < 1, opacity });
  const l = new LineSegments(g, m);
  l.renderOrder = 12;
  return l;
}

function polyline(p: number[][]): number[] {
  const out: number[] = [];
  for (let i = 1; i < p.length; i++) {
    const a = p[i - 1];
    const b = p[i];
    if (a && b) out.push(...a, ...b);
  }
  return out;
}

export class StitchView {
  private readonly group = new Group();
  private readonly seams = new Group();
  private lines: LineSegments | null = null;

  constructor(parent: Object3D) {
    this.group.name = "stitches";
    this.seams.name = "stitcher-seams";
    parent.add(this.group, this.seams);
  }

  /** segments: for each stitch, a list of [pointOnA, pointOnB] pairs in world metres. */
  setSegments(stitches: ReadonlyArray<ReadonlyArray<ReadonlyArray<ReadonlyArray<number>>>>): void {
    if (this.lines) {
      this.group.remove(this.lines);
      this.lines.geometry.dispose();
      this.lines = null;
    }
    const pts: number[] = [];
    for (const st of stitches) for (const [a, b] of st) pts.push(...(a ?? []), ...(b ?? []));
    if (pts.length === 0) return;
    this.lines = segments(pts, "#d946ef", 1);
    this.group.add(this.lines);
  }

  /** The stitcher's seams (empty list clears); ``selected`` highlights one seam. */
  setSeams(seams: ReadonlyArray<StitchSeam>, selected: string | null = null): void {
    for (const c of [...this.seams.children]) {
      this.seams.remove(c);
      (c as LineSegments).geometry.dispose();
    }
    seams.forEach((s, i) => {
      const dim = selected !== null && selected !== s.id ? 0.18 : 1;
      const colour = s.status === "fail" && dim === 1 ? "#dc2626" : seamColour(i);
      this.seams.add(segments([...polyline(s.a), ...polyline(s.b)], colour, dim));
      this.seams.add(segments(s.pairs.flatMap(([a, b]) => [...(a ?? []), ...(b ?? [])]), colour, dim * 0.6));
    });
  }
}
