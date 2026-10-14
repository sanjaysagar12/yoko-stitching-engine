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
