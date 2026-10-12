// Small pure helpers: display transforms and formatting only. No placement maths lives in the frontend.
import { Matrix4 } from "three";

/** Backend matrices are 4x4 row-major (column vectors); three.js Matrix4.set takes row-major arguments. */
export function matrixFromRowMajor(m: readonly number[]): Matrix4 {
  if (m.length !== 16) throw new Error(`expected 16 matrix entries, got ${m.length}`);
  const [a, b, c, d, e, f, g, h, i, j, k, l, mm, n, o, p] = m as [
    number, number, number, number, number, number, number, number,
    number, number, number, number, number, number, number, number,
  ];
  return new Matrix4().set(a, b, c, d, e, f, g, h, i, j, k, l, mm, n, o, p);
}

/** Left-to-right display layout: x offset of every piece for the given widths (cm), with a gap (cm), in metres. */
export function layoutOffsets(widthsCm: readonly number[], gapCm: number, startM: number): number[] {
  const out: number[] = [];
  let cursor = startM;
  for (const w of widthsCm) {
    out.push(cursor);
    cursor += (w + gapCm) / 100;
  }
  return out;
}

export const formatCm = (v: number | null | undefined, digits = 2): string =>
  v === null || v === undefined ? "-" : v.toFixed(digits);

export const formatSigned = (v: number | null | undefined, digits = 2): string =>
  v === null || v === undefined ? "-" : `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`;

/** Body part group of a bounding volume / point name: "UpperArm_L" -> "UpperArm". */
export const bodyPart = (bv: string): string => bv.replace(/_(L|R)$/, "");

export const PART_COLORS: Record<string, number> = {
  Head: 0xe6194b, Neck: 0xf58231, Chest: 0x3cb44b, Waist: 0x4363d8, Hip: 0x911eb4,
  UpperArm: 0x008080, Forearm: 0x46f0f0, Hand: 0x808000, Thigh: 0xf032e6, Shin: 0x9a6324, Foot: 0x111111,
};

export const partColor = (bv: string): number => PART_COLORS[bodyPart(bv)] ?? 0x888888;

/** Left/right counterpart of an arrangement point name (_L/_R or _Left/_Right); centre points map to themselves. */
export function mirrorName(name: string, known: ReadonlySet<string>): string | null {
  for (const [a, b] of [
    ["_L", "_R"],
    ["_Left", "_Right"],
  ] as const) {
    for (const [x, y] of [
      [a, b],
      [b, a],
    ] as const) {
      if (name.endsWith(x)) {
        const other = name.slice(0, -x.length) + y;
        return known.has(other) ? other : null;
      }
    }
  }
  return known.has(name) ? name : null;
}

/** Engine warnings are "CODE: text" strings. */
export function splitWarning(w: string): { code: string; text: string } {
  const m = /^([A-Za-z_]+):\s*([\s\S]*)$/.exec(w);
  return m ? { code: m[1] ?? "", text: m[2] ?? "" } : { code: "", text: w };
}

/** Default name of a clone: the suffix opposite to the source side when it is known (name or where it extends). */
export function cloneName(source: string, extendsSide?: "left" | "right" | "centre"): string {
  if (source.endsWith("_L")) return `${source.slice(0, -2)}_R`;
  if (source.endsWith("_R")) return `${source.slice(0, -2)}_L`;
  return `${source}_${extendsSide === "left" ? "R" : "L"}`;
}

/** Metres as millimetres with one decimal ("30.1"); "-" when unknown or not finite. */
export const formatMm = (m: number | null | undefined): string => (m === null || m === undefined || !Number.isFinite(m) ? "-" : (m * 1000).toFixed(1));

/** The nearest other piece in a ``min_distance_to_pieces_m`` table (null when there is none). */
export function nearestPiece(distances: Readonly<Record<string, number>>): { name: string; distance_m: number } | null {
  let best: { name: string; distance_m: number } | null = null;
  for (const [name, d] of Object.entries(distances)) {
    if (best === null || d < best.distance_m) best = { name, distance_m: d };
  }
  return best;
}

/** Bounding sphere radius used to frame a violation: 25 cm around the point. */
export const VIOLATION_FRAME_M = 0.25;
