// three.js scene: renderer, camera, orbit controls, lights, a centimetre grid on the floor and a vertical ruler.
// World units are metres, y up, feet on y = 0, +z toward the viewer at the start, +x = the wearer's left.
import {
  AmbientLight,
  Box3,
  CanvasTexture,
  Color,
  DirectionalLight,
  GridHelper,
  Group,
  LineBasicMaterial,
  LineSegments,
  BufferGeometry,
  Float32BufferAttribute,
  PerspectiveCamera,
  Scene,
  Sprite,
  SpriteMaterial,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export class SceneKit {
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(40, 1, 0.05, 50);
  readonly renderer: WebGLRenderer;
  readonly controls: OrbitControls;
  private readonly frameCallbacks: Array<(dt: number) => void> = [];
  private last = performance.now();

  constructor(private readonly host: HTMLElement) {
    this.renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(this.renderer.domElement);
    this.scene.background = new Color(0xf3f4f6);

    this.camera.position.set(0, 1.25, 3.4);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.9, 0);
    this.controls.enableDamping = true;
    this.controls.update();

    this.scene.add(new AmbientLight(0xffffff, 1.1));
    const key = new DirectionalLight(0xffffff, 1.6);
    key.position.set(1.5, 3, 3);
    this.scene.add(key);
    const back = new DirectionalLight(0xffffff, 0.8);
    back.position.set(-2, 2, -3);
    this.scene.add(back);

    this.scene.add(this.makeGrid());
    this.scene.add(this.makeRuler(2.0));

    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  onFrame(cb: (dt: number) => void): void {
    this.frameCallbacks.push(cb);
  }

  private tick(): void {
    const now = performance.now();
    const dt = Math.min((now - this.last) / 1000, 0.1);
    this.last = now;
    this.controls.update();
    for (const cb of this.frameCallbacks) cb(dt);
    this.renderer.render(this.scene, this.camera);
  }

  /** Draw one frame now (for a screenshot taken right after moving the camera). */
  render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private resize(): void {
    const w = Math.max(this.host.clientWidth, 1);
    const h = Math.max(this.host.clientHeight, 1);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Canvas pixel position (CSS pixels, relative to the canvas) of a world point, plus whether it is in front. */
  screenOf(p: Vector3): { x: number; y: number; visible: boolean } {
    const v = p.clone().project(this.camera);
    const r = this.renderer.domElement.getBoundingClientRect();
    return { x: ((v.x + 1) / 2) * r.width, y: ((1 - v.y) / 2) * r.height, visible: v.z > -1 && v.z < 1 };
  }

  /** Move the camera to look at a box from the front. */
  frame(box: Box3): void {
    const size = box.getSize(new Vector3());
    const centre = box.getCenter(new Vector3());
    const dist = (Math.max(size.y, size.x / this.camera.aspect) / 2 / Math.tan((this.camera.fov * Math.PI) / 360)) * 1.25 + size.z / 2;
    this.controls.target.copy(centre);
    this.camera.position.set(centre.x, centre.y + size.y * 0.05, centre.z + dist);
    this.controls.update();
  }

  /** Look at a box from one side: front (+z), back (-z), left (the wearer's left, +x), right (-x) or top (+y). */
  view(side: "front" | "back" | "left" | "right" | "top", box: Box3): void {
    const size = box.getSize(new Vector3());
    const c = box.getCenter(new Vector3());
    const span = side === "left" || side === "right" ? size.z : size.x;
    const dist = (Math.max(size.y, span / this.camera.aspect) / 2 / Math.tan((this.camera.fov * Math.PI) / 360)) * 1.3 + 0.6;
    const dir = {
      front: new Vector3(0, 0, 1),
      back: new Vector3(0, 0, -1),
      left: new Vector3(1, 0, 0),
      right: new Vector3(-1, 0, 0),
      top: new Vector3(0, 1, 0.001),
    }[side];
    const d = side === "top" ? Math.max(size.x, size.z) * 1.2 + 1.2 : dist;
    this.controls.target.copy(c);
    this.camera.position.copy(c).addScaledVector(dir.normalize(), d);
    this.controls.update();
  }

  /** Look at a point from outside the body (away from the vertical axis through the origin) at a distance that shows ``radius``. */
  focus(centre: Vector3, radius: number): void {
    const out = new Vector3(centre.x, 0, centre.z);
    if (out.lengthSq() < 1e-4) out.set(0, 0, 1);
    out.normalize();
    const dist = (radius / Math.tan((this.camera.fov * Math.PI) / 360)) * 1.4 + radius;
    this.controls.target.copy(centre);
    this.camera.position.copy(centre).addScaledVector(out, dist);
    this.controls.update();
  }

  /** Call ``cb`` whenever the camera moves (orbit, zoom, pan, ``view``, ``frame``). */
  onCameraChange(cb: () => void): void {
    this.controls.addEventListener("change", cb);
  }

  /** 1 cm minor and 10 cm major lines on the floor (y = 0), 4 m x 4 m. */
  private makeGrid(): Group {
    const g = new Group();
    const minor = new GridHelper(4, 400, 0xd0d4da, 0xe2e5e9);
    const major = new GridHelper(4, 40, 0x8e96a3, 0x8e96a3);
    for (const grid of [minor, major]) {
      const m = grid.material as LineBasicMaterial;
      m.transparent = true;
      m.opacity = grid === minor ? 0.55 : 0.8;
      m.depthWrite = false;
    }
    major.position.y = 0.0005;
    g.add(minor, major);
    g.position.set(0.9, 0, 0);
    g.name = "floor-grid-cm";
    return g;
  }

  /** Vertical ruler with a tick every 10 cm (long every 50 cm) and labels in metres, beside the avatar. */
  private makeRuler(height: number): Group {
    const g = new Group();
    g.name = "ruler";
    const pts: number[] = [0, 0, 0, 0, height, 0];
    for (let cm = 0; cm <= height * 100 + 0.1; cm += 10) {
      const y = cm / 100;
      const len = cm % 50 === 0 ? 0.08 : 0.04;
      pts.push(0, y, 0, len, y, 0);
      if (cm % 50 === 0) {
        const s = labelSprite(`${(cm / 100).toFixed(1)} m`);
        s.position.set(0.17, y, 0);
        g.add(s);
      }
    }
    const geo = new BufferGeometry();
    geo.setAttribute("position", new Float32BufferAttribute(pts, 3));
    g.add(new LineSegments(geo, new LineBasicMaterial({ color: 0x374151 })));
    g.position.set(-1.15, 0, 0);
    return g;
  }
}

function labelSprite(text: string): Sprite {
  const c = document.createElement("canvas");
  c.width = 128;
  c.height = 48;
  const ctx = c.getContext("2d");
  if (ctx) {
    ctx.font = "bold 28px sans-serif";
    ctx.fillStyle = "#111827";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 4, 24);
  }
  const s = new Sprite(new SpriteMaterial({ map: new CanvasTexture(c), depthTest: false, transparent: true }));
  s.scale.set(0.2, 0.075, 1);
  s.center.set(0, 0.5);
  s.renderOrder = 10;
  return s;
}
