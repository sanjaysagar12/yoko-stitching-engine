// Which arrangement points are hidden behind the avatar from the current camera.
// The avatar is drawn into a small depth target (packed RGBA depth), read back once per camera change, and every point
// is compared with the depth stored at its screen position. Display only: nothing here feeds the placement.
import {
  Mesh,
  MeshDepthMaterial,
  RGBADepthPacking,
  Scene,
  Vector2,
  Vector3,
  WebGLRenderTarget,
  NearestFilter,
  Color,
} from "three";
import type { SceneKit } from "./scene";

const SCALE = 0.5; // the depth target has half the canvas resolution
const EPS = 1.5e-4; // depth margin (about 3.5 cm at 3.4 m): a point this far in front of the avatar counts as visible

export class Occlusion {
  private readonly scene = new Scene();
  private readonly material = new MeshDepthMaterial({ depthPacking: RGBADepthPacking });
  private target: WebGLRenderTarget | null = null;
  private proxy: Mesh | null = null;
  private pixels = new Uint8Array(0);
  private size = { w: 0, h: 0 };

  constructor(private readonly kit: SceneKit) {
    this.scene.background = new Color(0xffffff);
  }

  /** Use the avatar surface (shares its geometry; the depth material writes depth, the visible material does not). */
  setSurface(surface: Mesh | null): void {
    if (this.proxy) this.scene.remove(this.proxy);
    this.proxy = surface ? new Mesh(surface.geometry, this.material) : null;
    if (this.proxy) this.scene.add(this.proxy);
  }

  /** Names of the points hidden behind the avatar. Call after the camera moved. */
  hidden(points: ReadonlyMap<string, Vector3>): Set<string> {
    const out = new Set<string>();
    if (!this.proxy) return out;
    const r = this.kit.renderer;
    const buf = r.getDrawingBufferSize(new Vector2());
    const w = Math.max(16, Math.round(buf.x * SCALE));
    const h = Math.max(16, Math.round(buf.y * SCALE));
    if (!this.target || this.size.w !== w || this.size.h !== h) {
      this.target?.dispose();
      this.target = new WebGLRenderTarget(w, h, { minFilter: NearestFilter, magFilter: NearestFilter });
      this.pixels = new Uint8Array(w * h * 4);
      this.size = { w, h };
    }
    const prev = r.getRenderTarget();
    r.setRenderTarget(this.target);
    r.setClearColor(0xffffff, 1);
    r.clear();
    r.render(this.scene, this.kit.camera);
    r.readRenderTargetPixels(this.target, 0, 0, w, h, this.pixels);
    r.setRenderTarget(prev);
    for (const [name, p] of points) {
      const ndc = p.clone().project(this.kit.camera);
      if (ndc.z < -1 || ndc.z > 1) continue;
      const px = Math.min(w - 1, Math.max(0, Math.floor(((ndc.x + 1) / 2) * w)));
      const py = Math.min(h - 1, Math.max(0, Math.floor(((ndc.y + 1) / 2) * h))); // render targets are bottom-up
      const i = (py * w + px) * 4;
      const [r0, g0, b0, a0] = [this.pixels[i] ?? 255, this.pixels[i + 1] ?? 255, this.pixels[i + 2] ?? 255, this.pixels[i + 3] ?? 255];
      // RGBADepthPacking as read back here: the most significant byte of the depth is in r, the least in a
      const depth = r0 / 256 + g0 / 65536 + b0 / 16777216 + a0 / 4294967296;
      if (depth < ndc.z * 0.5 + 0.5 - EPS) out.add(name);
    }
    return out;
  }
}
