// Thin typed wrapper around the backend. The frontend never computes placement: it asks /api/align.
import type {
  AlignResult,
  ApiError,
  ArrangeOptions,
  Axis,
  BuildStatus,
  DerivedDef,
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
  return (await res.json()) as T;
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
): Promise<AlignResult> =>
  request<AlignResult>(
    "/api/align",
    json("POST", { build, placements, derived_pieces: derived, stitches, arrange_options: options, compact: true, known_meshes: knownMeshes }),
  );

/** Re-solve every piece that is not locked (Auto-arrange). Same answer as align. */
export const arrange = (
  build: string,
  placements: Placement[],
  derived: DerivedDef[],
  options: ArrangeOptions,
  stitches: Stitch[] = [],
  knownMeshes: string[] = [],
): Promise<AlignResult> =>
  request<AlignResult>(
    "/api/arrange",
    json("POST", { build, placements, derived_pieces: derived, stitches, arrange_options: options, compact: true, known_meshes: knownMeshes }),
  );

/** Make a linked mirrored copy of a piece (the backend asks the alignment engine). */
export const clonePiece = (build: string, source: string, axis: Axis, id: string): Promise<PieceInfo> =>
  request<PieceInfo>("/api/clone", json("POST", { build, source, axis, id }));

/** Placements for a piece and its clone (same anchor on the centre line, or mirrored anchor on the other side). */
export const pairPlacements = (
  build: string,
  base: Placement,
  derivedId: string,
  mode: PairMode,
  derived: DerivedDef[],
): Promise<{ placements: Placement[] }> =>
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
