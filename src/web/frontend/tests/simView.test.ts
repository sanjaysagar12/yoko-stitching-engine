import { describe, expect, it } from "vitest";
import { decodeF32, embed } from "../src/simView";

describe("simulation embedding", () => {
  it("puts every placed-mesh vertex at the weighted sum of its three simulation vertices", () => {
    // simulation mesh: one triangle; placed mesh: its corner 0, its centroid and the middle of edge 1-2
    const x = [0, 0, 0, 1, 0, 0, 0, 2, 0];
    const idx = [0, 1, 2, 0, 1, 2, 2, 1, 0];
    const w = [1, 0, 0, 1 / 3, 1 / 3, 1 / 3, 0.5, 0.5, 0];
    const out = embed(idx, w, x, new Float32Array(9));
    expect(Array.from(out.slice(0, 3))).toEqual([0, 0, 0]);
    expect(out[3]).toBeCloseTo(1 / 3);
    expect(out[4]).toBeCloseTo(2 / 3);
    expect(Array.from(out.slice(6, 9))).toEqual([0.5, 1, 0]);
  });

  it("decodes base64 little-endian float32", () => {
    const b64 = btoa(String.fromCharCode(...new Uint8Array(new Float32Array([1.5, -2]).buffer)));
    expect(Array.from(decodeF32(b64))).toEqual([1.5, -2]);
  });
});
