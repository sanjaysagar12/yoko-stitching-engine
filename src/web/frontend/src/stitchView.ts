// Stitch segments from the alignment engine (pairs of world points along two seam edges), drawn as thin lines.
import { BufferGeometry, Float32BufferAttribute, Group, LineBasicMaterial, LineSegments, Object3D } from "three";

export class StitchView {
  private readonly group = new Group();
  private lines: LineSegments | null = null;

  constructor(parent: Object3D) {
    this.group.name = "stitches";
    parent.add(this.group);
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
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(pts, 3));
    this.lines = new LineSegments(g, new LineBasicMaterial({ color: 0xd946ef, depthTest: false }));
    this.lines.renderOrder = 12;
    this.group.add(this.lines);
  }
}
