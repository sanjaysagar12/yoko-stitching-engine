// The avatar mesh (semi-transparent so pieces and points stay visible) and its bounding volumes as wireframe capsules.
import {
  Box3,
  CapsuleGeometry,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  Quaternion,
  Vector3,
} from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { fetchText } from "./api";
import { bodyPart, PART_COLORS } from "./lib";
import type { BoundingVolume } from "./types";

export class AvatarView {
  readonly root = new Group();
  private mesh: Mesh | null = null;
  private readonly volumes = new Group();

  constructor(parent: Object3D) {
    this.root.name = "avatar";
    this.volumes.visible = false;
    this.root.add(this.volumes);
    parent.add(this.root);
  }

  async load(url: string, opacity: number): Promise<void> {
    const text = await fetchText(url);
    const obj = new OBJLoader().parse(text);
    let found: Mesh | null = null;
    obj.traverse((o) => {
      if (!found && (o as Mesh).isMesh) found = o as Mesh;
    });
    const mesh = found as Mesh | null;
    if (!mesh) throw new Error("avatar OBJ has no mesh");
    mesh.geometry.computeVertexNormals();
    mesh.material = new MeshStandardMaterial({
      color: 0xb7bcc6,
      roughness: 0.85,
      metalness: 0,
      transparent: true,
      opacity,
      depthWrite: false,
    });
    if (this.mesh) this.root.remove(this.mesh);
    this.mesh = mesh;
    this.root.add(mesh);
  }

  /** The visible avatar surface mesh (null until loaded). */
  get surface(): Mesh | null {
    return this.mesh;
  }

  setOpacity(opacity: number): void {
    if (this.mesh) (this.mesh.material as MeshStandardMaterial).opacity = opacity;
  }

  setVolumes(volumes: BoundingVolume[]): void {
    this.volumes.clear();
    for (const v of volumes) {
      const a = new Vector3(...v.axis_start);
      const b = new Vector3(...v.axis_end);
      const dir = b.clone().sub(a);
      const len = dir.length();
      const geo = new CapsuleGeometry(v.radius_m, len, 4, 16);
      const color = PART_COLORS[bodyPart(v.name)] ?? 0x888888;
      const m = new Mesh(geo, new MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.35 }));
      m.position.copy(a.clone().add(b).multiplyScalar(0.5));
      m.quaternion.copy(new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), dir.normalize()));
      m.name = `bv-${v.name}`;
      this.volumes.add(m);
    }
  }

  setVolumesVisible(on: boolean): void {
    this.volumes.visible = on;
  }

  bounds(): Box3 {
    return this.mesh ? new Box3().setFromObject(this.mesh) : new Box3();
  }
}
