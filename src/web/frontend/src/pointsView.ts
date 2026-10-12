// Arrangement points: small spheres coloured by body part, optional name labels, hover (with the mirrored point),
// selection, a name filter, and dimming of points hidden behind the avatar (pickable again with the X-ray toggle).
import { Group, Mesh, MeshBasicMaterial, Object3D, RingGeometry, SphereGeometry, Vector3 } from "three";
import { mirrorName, partColor } from "./lib";
import type { SceneKit } from "./scene";
import type { ArrangementPoint } from "./types";

const PICK_RADIUS_PX = 14;
const HIDDEN_OPACITY = 0.15;

export class PointsView {
  private readonly group = new Group();
  private readonly spheres = new Map<string, Mesh>();
  private readonly labels = new Map<string, HTMLDivElement>();
  private names = new Set<string>();
  private labelsOn = false;
  private hover: string | null = null;
  private selected: string | null = null;
  private extra: ReadonlySet<string> = new Set();
  private occluded: ReadonlySet<string> = new Set();
  private xray = false;
  private filter = "";
  private readonly ring = new Mesh(new RingGeometry(0.02, 0.026, 32), new MeshBasicMaterial({ color: 0x111827, depthTest: false }));

  constructor(
    parent: Object3D,
    private readonly kit: SceneKit,
    private readonly overlay: HTMLElement,
  ) {
    this.group.name = "arrangement-points";
    this.ring.visible = false;
    this.ring.renderOrder = 20;
    this.group.add(this.ring);
    parent.add(this.group);
    kit.onFrame(() => this.update());
  }

  setPoints(points: ArrangementPoint[]): void {
    for (const s of this.spheres.values()) this.group.remove(s);
    for (const l of this.labels.values()) l.remove();
    this.spheres.clear();
    this.labels.clear();
    this.names = new Set(points.map((p) => p.name));
    const geo = new SphereGeometry(0.011, 14, 10);
    for (const p of points) {
      const mat = new MeshBasicMaterial({ color: partColor(p.bv), transparent: true, depthTest: false });
      const m = new Mesh(geo, mat); // depthTest off: visible through the semi-transparent avatar (hidden ones are dimmed)
      m.position.set(...p.position_m);
      m.name = p.name;
      m.renderOrder = 15;
      this.spheres.set(p.name, m);
      this.group.add(m);
      const label = document.createElement("div");
      label.className = "point-label";
      label.textContent = p.name;
      label.hidden = true;
      this.overlay.appendChild(label);
      this.labels.set(p.name, label);
    }
  }

  positions(): Map<string, Vector3> {
    return new Map([...this.spheres].map(([n, m]) => [n, m.position.clone()]));
  }

  setLabelsVisible(on: boolean): void {
    this.labelsOn = on;
  }

  setHover(name: string | null): void {
    this.hover = name;
  }

  /** Extra points to emphasise (for example both points of a mirrored-anchor pair). */
  setHighlights(names: Iterable<string>): void {
    this.extra = new Set(names);
  }

  setOccluded(names: ReadonlySet<string>): void {
    this.occluded = names;
  }

  setXray(on: boolean): void {
    this.xray = on;
  }

  setFilter(text: string): void {
    this.filter = text.trim().toLowerCase();
  }

  setSelected(name: string | null): void {
    this.selected = name;
    const m = name ? this.spheres.get(name) : undefined;
    this.ring.visible = Boolean(m);
    if (m) this.ring.position.copy(m.position);
  }

  private matches(name: string): boolean {
    return this.filter === "" || name.toLowerCase().includes(this.filter);
  }

  /** A point can be seen and clicked: it passes the name filter and is not behind the avatar (unless X-ray is on). */
  pickable(name: string): boolean {
    return this.spheres.has(name) && this.matches(name) && (this.xray || !this.occluded.has(name));
  }

  info(name: string): { x: number; y: number; hidden: boolean; pickable: boolean; filtered: boolean } | null {
    const m = this.spheres.get(name);
    if (!m) return null;
    const s = this.kit.screenOf(m.position);
    return { x: s.x, y: s.y, hidden: this.occluded.has(name), pickable: this.pickable(name), filtered: !this.matches(name) };
  }

  /** Canvas position of a point (for picking and for the end-to-end test). */
  screenOf(name: string): { x: number; y: number } | null {
    const i = this.info(name);
    return i ? { x: i.x, y: i.y } : null;
  }

  /** Pickable point nearest to a canvas position (CSS px relative to the canvas) within the pick radius. */
  pick(x: number, y: number): string | null {
    let best: { name: string; d: number } | null = null;
    for (const [name, m] of this.spheres) {
      if (!this.pickable(name)) continue;
      const s = this.kit.screenOf(m.position);
      if (!s.visible) continue;
      const d = Math.hypot(s.x - x, s.y - y);
      if (d <= PICK_RADIUS_PX && (!best || d < best.d)) best = { name, d };
    }
    return best ? best.name : null;
  }

  position(name: string): Vector3 | null {
    return this.spheres.get(name)?.position.clone() ?? null;
  }

  private update(): void {
    this.ring.quaternion.copy(this.kit.camera.quaternion);
    const twin = this.hover ? mirrorName(this.hover, this.names) : null;
    for (const [name, m] of this.spheres) {
      const mat = m.material as MeshBasicMaterial;
      const shown = this.matches(name);
      m.visible = shown;
      const hidden = this.occluded.has(name) && !this.xray;
      mat.opacity = hidden ? HIDDEN_OPACITY : 1;
      const hot = name === this.hover || name === this.selected || name === twin || this.extra.has(name);
      m.scale.setScalar(hot ? 1.8 : 1);
      const label = this.labels.get(name);
      if (!label) continue;
      const show = shown && (this.labelsOn ? !hidden : hot);
      const s = this.kit.screenOf(m.position);
      label.hidden = !show || !s.visible;
      if (!label.hidden) label.style.transform = `translate(${Math.round(s.x + 9)}px, ${Math.round(s.y - 8)}px)`;
    }
  }
}
