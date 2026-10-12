// Shapes of the JSON the backend returns. Metres everywhere except fields named *_cm.

export interface Job {
  name: string;
  sm2d: string;
  smis: string;
  body: string;
  pieces: string[] | null;
}

export interface ApiError {
  code: string;
  message: string;
  piece?: string;
  tool?: string;
}

export interface NamedPoint {
  name: string;
  on_outline: boolean;
  xy_m: [number, number];
}

export type Axis =
  | { type: "bbox_center" }
  | { type: "edge"; from_point: string; to_point: string }
  | { type: "x"; value_m: number };

export interface DerivedDef {
  id: string;
  source: string;
  op: "mirror";
  axis: Axis;
}

export interface StitchEnd {
  piece: string;
  from_point: string;
  to_point: string;
}

export interface Stitch {
  a: StitchEnd;
  b: StitchEnd;
}

export interface PieceInfo {
  name: string;
  status: "ok" | "failed";
  reason?: string;
  code?: string;
  tool?: string;
  width_cm?: number;
  height_cm?: number;
  area_cm2?: number;
  obj?: string;
  bundle?: string;
  named_points?: NamedPoint[];
  default_reference?: [number, number];
  warnings?: string[];
  /** Derived (mirrored) pieces only. */
  mirror_of?: string;
  source?: string;
  axis?: Axis;
  suggested_stitch?: Stitch | null;
}

/** The parts of a piece bundle (<piece>.piece.json) the inspector draws. */
export interface PieceBundle {
  name: string;
  outline_m: Array<[number, number]>;
  bbox_m: { min: [number, number]; max: [number, number] };
  named_points: Array<{ name: string; label: string; xy_m: [number, number]; on_outline: boolean }>;
  default_reference: { xy_m: [number, number] };
}

export interface ArrangementPoint {
  name: string;
  bv: string;
  angle_deg: number;
  t: number;
  offset_m: number;
  wrap_direction: "around" | "flat";
  position_m: [number, number, number];
  normal: [number, number, number];
  up: [number, number, number];
  tangent: [number, number, number];
}

export interface BoundingVolume {
  name: string;
  kind: string;
  axis_start: [number, number, number];
  axis_end: [number, number, number];
  radius_m: number;
  length_m: number;
}

export interface MeasurementRow {
  kind: string;
  definition: string;
  target_cm: number | null;
  pre_correction_cm: number;
  post_correction_cm: number;
  pre_error_cm: number | null;
  post_error_cm: number | null;
}

export interface BuildResult {
  id: string;
  job: string;
  pieces: PieceInfo[];
  avatar: {
    bundle: { name: string; model_type: string; model_file: string; num_betas_used: number; convention: string };
    mesh: string;
    measurements: { measurements: Record<string, MeasurementRow>; warnings: string[] };
    points: ArrangementPoint[];
    volumes: BoundingVolume[];
  };
  warnings: string[];
}

export interface BuildStatus {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  stage: string;
  progress: number;
  result?: BuildResult;
  error?: ApiError;
}

export type WrapMode = "flat" | "around" | "body_slice";
export type PairMode = "same_anchor" | "mirror_anchor";
export type Side = "left" | "right" | "centre";

export interface Placement {
  piece: string;
  point: string;
  offset_m: number | null;
  rotation_deg: number;
  mirror: boolean;
  wrap: WrapMode;
  flip_facing: boolean;
  reference: string;
  /** Auto-arrange: body region of the shell ("torso", "arm_R", ...); absent = the region of the point. */
  ring?: string;
  /** Auto-arrange: layer above the body (0 = on the shell, 1 = floats over it); absent = automatic. */
  layer?: number;
  /** Auto-arrange: start shift of the piece on its shell, metres [around, along the axis]. */
  shift_m?: [number, number];
  /** Auto-arrange: keep the placement as it is and treat the piece as an obstacle. */
  locked?: boolean;
}

export const RING_NAMES = ["torso", "neck", "arm_L", "arm_R", "leg_L", "leg_R"] as const;

/** Options of /api/arrange (metres). */
export interface ArrangeOptions {
  clearance_body: number;
  gap_piece: number;
  gap_layer: number;
  max_standoff: number;
  max_shift: number;
  refine_max_edge: number;
  standoff_mode: "uniform" | "per_row";
}

export const DEFAULT_ARRANGE_OPTIONS: ArrangeOptions = {
  clearance_body: 0.03,
  gap_piece: 0.03,
  gap_layer: 0.015,
  max_standoff: 0.15,
  max_shift: 0.06,
  refine_max_edge: 0.01,
  standoff_mode: "uniform",
};

/** Per piece numbers of an arrangement. */
export interface ArrangePiece {
  ring: string;
  layer: number;
  locked: boolean;
  standoff_m: number | null;
  shift_m: [number, number];
  min_distance_to_avatar_m: number;
  min_distance_to_pieces_m: Record<string, number>;
  flags: string[];
  area_ratio: number | null;
  edge_change_p50_pct?: number | null;
  edge_change_p90_pct?: number | null;
  max_edge_change_pct: number | null;
  max_edge_change_body_pct: number | null;
  max_edge_change_shoulder_pct: number | null;
  self_intersections: number;
}

export interface WorldEdge {
  id: string;
  from: string;
  to: string;
  length_m: number;
  polyline_m: number[][];
}

export interface WorldPoint {
  name: string;
  label: string;
  on_outline: boolean;
  world_m: [number, number, number];
}

export interface RingRow {
  ring: string;
  pieces: string[];
  body_loop_m: number;
  garment_loop_m: number;
  true_ease_m: number;
  standoff_used_m: number;
  standoff_required_m: number;
  exceeded: boolean;
  binding_row_m: number | null;
  violating_rows: Array<{ row_m: number; needed_standoff_m: number }>;
}

export interface Violation {
  kind: "CLEARANCE_BODY" | "GAP_PIECES" | "STANDOFF_EXCEEDED";
  pieces: string[];
  value_m: number;
  required_m: number;
  at_m: [number, number, number] | null;
}

export interface SeamCandidate {
  a: { piece: string; edge: string; length_m: number; polyline_m: number[][] };
  b: { piece: string; edge: string; length_m: number; polyline_m: number[][] };
  distance_m: number;
  length_ratio: number;
}

export interface Arrangement {
  options: ArrangeOptions;
  rings: RingRow[];
  seam_candidates: SeamCandidate[];
  violations: Violation[];
  notes: string[];
  timing_s: number;
  convention: string;
}

export interface PlacedResult {
  piece: string;
  point: string;
  matrix: number[]; // 4x4 row-major, metres
  mirrored: boolean;
  extends: Side;
  wrapped_vertices: number[][] | null;
  wrapped_faces: number[][] | null;
  reference_position_m: [number, number, number];
  bbox_cm: { width: number; height: number };
  warnings: string[];
  fit: Record<string, unknown>;
  /** Present after /api/arrange. */
  arrange?: ArrangePiece;
  world_outline_m?: number[][];
  world_named_points?: WorldPoint[];
  world_edges?: WorldEdge[];
}

export interface DerivedEntry {
  id: string;
  source: string;
  axis: Axis;
  obj: string;
  bundle: string;
  bbox_cm: { width: number; height: number };
  suggested_stitch: Stitch | null;
}

export interface StitchReport {
  a: StitchEnd;
  b: StitchEnd;
  a_length_cm: number;
  b_length_cm: number;
  diff_cm: number;
  diff_pct: number;
  segments: number[][][] | null;
}

export interface AlignResult {
  placements: PlacedResult[];
  derived_pieces: DerivedEntry[];
  stitch_reports: StitchReport[];
  warnings: string[];
  arrangement?: Arrangement;
}

/** What the user picked on an arranged piece: an outline edge or a named point. */
export type Selection =
  | { kind: "edge"; piece: string; id: string; from: string; to: string; length_m: number }
  | { kind: "point"; piece: string; name: string; world_m: [number, number, number] };
