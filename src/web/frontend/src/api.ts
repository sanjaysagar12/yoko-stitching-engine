// Thin typed wrapper around the backend. The frontend never computes placement: it asks /api/align.
import type {
  AlignResult,
  ApiError,
  ArrangeOptions,
  AssemblyExpect,
  AssemblyPayload,
  Axis,
  BuildStatus,
  DerivedDef,
  ExportJobStatus,
  ImageExportOptions,
  Job,
  PairMode,
  PieceBundle,
  PieceInfo,
  Placement,
  Stitch,
} from "./types";

export class ApiFailure extends Error {
  constructor(
    public readonly status: number,
    public readonly error: ApiError,
  ) {
    super(error.message);
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  return (await (await requestRaw(url, init)).json()) as T;
}

async function requestRaw(url: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let err: ApiError = { code: "http_error", message: `${res.status} ${res.statusText}` };
    try {
      const body = (await res.json()) as { error?: ApiError };
      if (body.error) err = body.error;
    } catch {
      /* keep the generic error */
    }
    throw new ApiFailure(res.status, err);
  }
  return res;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const listJobs = (): Promise<Job[]> => request<Job[]>("/api/jobs");

export const startBuild = (job: string): Promise<{ build_id: string; status: string; cached: boolean }> =>
  request(`/api/jobs/${encodeURIComponent(job)}/build`, { method: "POST" });

export const buildStatus = (id: string): Promise<BuildStatus> => request<BuildStatus>(`/api/builds/${encodeURIComponent(id)}`);

/** Place the pieces (incremental: placements with `solved` stay where they are). Meshes come compact (base64). */
export const align = (
  build: string,
  placements: Placement[],
  derived: DerivedDef[],
  stitches: Stitch[] = [],
  options?: ArrangeOptions,
  knownMeshes: string[] = [],
  rings: unknown[] = [],
): Promise<AlignResult> =>
  request<AlignResult>(
    "/api/align",
    json("POST", { build, placements, derived_pieces: derived, stitches, rings, arrange_options: options, compact: true, known_meshes: knownMeshes }),
  );

/** Re-solve every piece that is not locked (Auto-arrange). Same answer as align. */
export const arrange = (
  build: string,
  placements: Placement[],
  derived: DerivedDef[],
  options: ArrangeOptions,
  stitches: Stitch[] = [],
  knownMeshes: string[] = [],
  rings: unknown[] = [],
): Promise<AlignResult> =>
  request<AlignResult>(
    "/api/arrange",
    json("POST", { build, placements, derived_pieces: derived, stitches, rings, arrange_options: options, compact: true, known_meshes: knownMeshes }),
  );

/** Make a linked mirrored copy of a piece (the backend asks the alignment engine). */
export const clonePiece = (build: string, source: string, axis: Axis, id: string): Promise<PieceInfo> =>
  request<PieceInfo>("/api/clone", json("POST", { build, source, axis, id }));

/** Make an exact copy of a piece (same orientation; ``derived`` lets it copy a derived piece). */
export const copyPiece = (build: string, source: string, id: string, derived: DerivedDef[]): Promise<PieceInfo> =>
  request<PieceInfo>("/api/clone", json("POST", { build, source, id, op: "copy", derived_pieces: derived }));

/** Placements for a piece and its clone (same anchor on the centre line, or mirrored anchor on the other side). */
export const pairPlacements = (
  build: string,
  base: Placement,
  derivedId: string,
  mode: PairMode,
  derived: DerivedDef[],
): Promise<{ placements: Placement[]; warnings?: string[] }> =>
  request("/api/pair", json("POST", { build, base, derived_id: derivedId, mode, derived_pieces: derived }));

export const loadPlacements = (job: string): Promise<{ placements: Placement[]; derived_pieces?: DerivedDef[] }> =>
  request(`/api/jobs/${encodeURIComponent(job)}/placements`);

export const savePlacements = (job: string, placements: Placement[], derived: DerivedDef[]): Promise<unknown> =>
  request(`/api/jobs/${encodeURIComponent(job)}/placements`, json("PUT", { placements, derived_pieces: derived }));

export const fileUrl = (build: string, path: string): string => `/api/builds/${encodeURIComponent(build)}/files/${path}`;

export const fetchBundle = (build: string, path: string): Promise<PieceBundle> => request<PieceBundle>(fileUrl(build, path));

export const exportUrl = (build: string, format: "obj" | "glb"): string =>
  `/api/export/${encodeURIComponent(build)}?format=${format}`;

export async function fetchText(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new ApiFailure(res.status, { code: "http_error", message: `${url}: ${res.status}` });
  return res.text();
}

// -- assembly files and image export -------------------------------------------------------------------------------------

/** Read an assembly file (text) on the server. Without `build` only the file and its job are checked. */
export const importAssembly = (text: string, q: { job?: string; build?: string; skipMissing?: boolean } = {}): Promise<AssemblyPayload> => {
  const params = new URLSearchParams();
  if (q.job) params.set("job", q.job);
  if (q.build) params.set("build", q.build);
  if (q.skipMissing) params.set("skip_missing", "true");
  return request<AssemblyPayload>(`/api/assembly/import?${params.toString()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: text,
  });
};

export interface AssemblyExportBody {
  job: string;
  placements: Placement[];
  derived_pieces: DerivedDef[];
  arrange_options: ArrangeOptions;
  stitches: Stitch[];
  rings: unknown[];
  garment: string;
  notes: string;
  mode: "arrange" | "incremental";
  include_solved: boolean;
  expect?: AssemblyExpect;
  views?: string[];
}

/** The assembly file text (validated by the server). */
export const exportAssembly = async (body: AssemblyExportBody): Promise<string> => (await requestRaw("/api/assembly/export", json("POST", body))).text();

export interface ImageExportBody extends Omit<AssemblyExportBody, "include_solved" | "job"> {
  build: string;
  options: ImageExportOptions;
  viewport?: Record<string, string>;
}

export const startImageExport = (body: ImageExportBody): Promise<ExportJobStatus> => request("/api/export/images", json("POST", body));
export const imageExportStatus = (id: string): Promise<ExportJobStatus> => request(`/api/export/images/${encodeURIComponent(id)}`);
export const cancelImageExport = (id: string): Promise<ExportJobStatus> => request(`/api/export/images/${encodeURIComponent(id)}`, { method: "DELETE" });
export const imageExportUrl = (id: string): string => `/api/export/images/${encodeURIComponent(id)}/download`;

// -- stitching (stitcher on the layout on screen) --------------------------------------------------------------------

export interface StitchSeam {
  id: string;
  label: string;
  kind: string;
  ease: "none" | "distribute" | "gather";
  declared_ratio: number;
  length_a_cm: number;
  length_b_cm: number;
  ratio: number;
  gap_m: { min: number; mean: number; max: number };
  crossings: number;
  status: "pass" | "fail";
  problems: string[];
  /** Sewing lines on both pieces (world metres). */
  a: number[][];
  b: number[][];
  /** A few correspondences [pointOnA, pointOnB]. */
  pairs: number[][][];
}

export interface StitchAnswer {
  pass: boolean;
  summary: Record<string, number | boolean>;
  seams: StitchSeam[];
  errors: { seam: string; error: string }[];
  unstitched_edges: { piece: string; edge: string; length_cm: number }[];
  failures: string[];
}

/** The layout the stitcher sews (placements carry their `solved` poses: nothing moves on the server). */
export interface StitchLayout {
  build: string;
  placements: Placement[];
  derived_pieces: DerivedDef[];
  arrange_options: ArrangeOptions;
  rings: unknown[];
}

export const importStitching = (text: string): Promise<{ stitching: Record<string, unknown>; warnings: string[] }> =>
  request("/api/stitching/import", { method: "POST", headers: { "Content-Type": "application/json" }, body: text });
export const runStitching = (body: StitchLayout & { stitching: unknown }): Promise<StitchAnswer> => request("/api/stitching/run", json("POST", body));
export const stitchingResultText = async (body: StitchLayout & { stitching: unknown }): Promise<string> =>
  (await requestRaw("/api/stitching/result", json("POST", body))).text();
export const stitchingZip = async (body: StitchLayout & { stitching: unknown }): Promise<Blob> =>
  (await requestRaw("/api/stitching/export", json("POST", body))).blob();

// -- simulation (the stitched layout sewn and draped by the cloth solver) ---------------------------------------------

export interface SimulationOptions {
  fabric: string;
  quality: string;
  frames: number;
  sew_time?: number;
}

export interface SimulationInfo {
  solver: { available: boolean; path: string | null; release: string | null; message: string };
  fabrics: { name: string; label: string; note: string }[];
  qualities: { name: string; mesh_mm: number }[];
  defaults: SimulationOptions & { sew_time: number; dt_after: number; keep_every: number };
}

export interface SimulationSummary {
  pass: boolean;
  finished: boolean;
  frames: number;
  wall_s: number | null;
  seam_gap_max_cm: number;
  seam_gap_mean_cm: number;
  seams_open: number;
  stretch_max_pct: number | null;
  inside_body: number | null;
  sim_vertices: number;
  sim_triangles: number;
  fails: string[];
}

export interface SimulationStatus {
  id: string;
  build: string;
  garment: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  stage: string;
  frame: number;
  frames: number;
  progress: number;
  gap_max_m: number | null;
  wall_s: number | null;
  options: Partial<SimulationOptions>;
  cached: boolean;
  has_scene: boolean;
  summary?: SimulationSummary;
  error?: ApiError;
}

/** Per piece: the simulation vertices (`idx`, 3 per placed-mesh vertex) and their weights (`w`), base64. */
export interface SimulationEmbedding {
  sim_vertices: number;
  idx_dtype: "uint16" | "uint32";
  pieces: { name: string; count: number; idx: string; w: string }[];
  start: string;
}

const sid = (id: string): string => `/api/simulation/${encodeURIComponent(id)}`;
export const simulationInfo = (): Promise<SimulationInfo> => request("/api/simulation/info");
export const startSimulation = (body: StitchLayout & { stitching: unknown; options: SimulationOptions }): Promise<SimulationStatus> =>
  request("/api/simulation/start", json("POST", body));
export const simulationStatus = (id: string): Promise<SimulationStatus> => request(sid(id));
export const cancelSimulation = (id: string): Promise<SimulationStatus> => request(sid(id), { method: "DELETE" });
export const simulationEmbedding = (id: string): Promise<SimulationEmbedding> => request(`${sid(id)}/embedding`);
export const simulationLatest = (id: string): Promise<{ frame: number; x: string | null }> => request(`${sid(id)}/latest`);
export const simulationFrames = (id: string): Promise<{ frames: number[]; vertices: number; x: string }> => request(`${sid(id)}/frames`);
export const simulationDownloadUrl = (id: string, format: "obj" | "zip"): string => `${sid(id)}/download?format=${format}`;
export const simulationImageUrl = (id: string, name: string): string => `${sid(id)}/images/${encodeURIComponent(name)}`;
