// One small observable store. Everything the UI shows is derived from this state.
import type {
  AlignResult,
  ApiError,
  ArrangeOptions,
  Arrangement,
  BuildResult,
  DerivedDef,
  Job,
  PairMode,
  PieceInfo,
  PlacedResult,
  Placement,
  Selection,
  WrapMode,
} from "./types";
import { DEFAULT_ARRANGE_OPTIONS } from "./types";

export interface State {
  jobs: Job[];
  job: string | null;
  buildId: string | null;
  build: BuildResult | null;
  /** Placements the user chose, in the order they were made. The backend computes where they land. */
  placements: Map<string, Placement>;
  /** Latest backend answer per placed piece. */
  placed: Map<string, PlacedResult>;
  /** Mirrored clones made by the user (saved with the placements). */
  derived: DerivedDef[];
  derivedInfos: Map<string, PieceInfo>;
  /** Reference point chosen for a piece before it is placed. */
  referenceDraft: Map<string, string>;
  selectedPiece: string | null;
  /** Point chosen before a piece ("point first, then piece"). */
  pendingPoint: string | null;
  /** A pair waiting for the user to click an arrangement point. */
  pendingPair: { source: string; derivedId: string; mode: PairMode } | null;
  hoverPoint: string | null;
  showLabels: boolean;
  showVolumes: boolean;
  xray: boolean;
  pointFilter: string;
  /** Arrangement points hidden behind the avatar from the current camera. */
  occluded: ReadonlySet<string>;
  avatarOpacity: number;
  /** Progress text while a build runs (null when idle). */
  busy: { text: string; progress: number } | null;
  error: ApiError | null;
  lastAlign: AlignResult | null;
  /** Auto-arrange: options, whether the arranged result is on show, and the report of the last arrangement. */
  arrangeOptions: ArrangeOptions;
  arranged: boolean;
  arrangement: Arrangement | null;
  /** Interior mesh lines of the pieces (hidden by default). */
  showMeshLines: boolean;
  /** Pieces switched off with their eye button (replace the Set to change it). */
  hiddenPieces: ReadonlySet<string>;
  /** Pieces emphasised after a click on a violation. */
  highlight: ReadonlyArray<string>;
  /** The outline edge or named point picked on an arranged piece. */
  selection: Selection | null;
  /** Text under the pointer while hovering an arranged piece (distance to its neighbours). */
  hoverText: string | null;
}

export type Listener = (state: State, changed: ReadonlyArray<keyof State>) => void;

export class Store {
  private state: State = {
    jobs: [],
    job: null,
    buildId: null,
    build: null,
    placements: new Map(),
    placed: new Map(),
    derived: [],
    derivedInfos: new Map(),
    referenceDraft: new Map(),
    selectedPiece: null,
    pendingPoint: null,
    pendingPair: null,
    hoverPoint: null,
    showLabels: false,
    showVolumes: false,
    xray: false,
    pointFilter: "",
    occluded: new Set(),
    avatarOpacity: 0.5,
    busy: null,
    error: null,
    lastAlign: null,
    arrangeOptions: { ...DEFAULT_ARRANGE_OPTIONS },
    arranged: false,
    arrangement: null,
    showMeshLines: false,
    hiddenPieces: new Set(),
    highlight: [],
    selection: null,
    hoverText: null,
  };
  private listeners = new Set<Listener>();

  get(): Readonly<State> {
    return this.state;
  }

  set(patch: Partial<State>): void {
    const changed = (Object.keys(patch) as Array<keyof State>).filter((k) => patch[k] !== this.state[k]);
    if (changed.length === 0) return;
    this.state = { ...this.state, ...patch };
    for (const l of this.listeners) l(this.state, changed);
  }

  /** Notify after mutating a Map or array held in the state (placements, placed, derived, ...). */
  touch(...keys: Array<keyof State>): void {
    for (const l of this.listeners) l(this.state, keys);
  }

  subscribe(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }
}

export const TORSO_PARTS = ["Chest", "Waist", "Hip", "Neck"];

/** Default wrap of a new placement: body_slice on the torso, around on limbs and the head. */
export const defaultWrap = (bv: string | undefined): WrapMode => (bv && TORSO_PARTS.includes(bv) ? "body_slice" : "around");

export const defaultPlacement = (piece: string, point: string, bv?: string, reference = "bbox_centre"): Placement => ({
  piece,
  point,
  offset_m: null,
  rotation_deg: 0,
  mirror: false,
  wrap: defaultWrap(bv),
  flip_facing: false,
  reference,
});
