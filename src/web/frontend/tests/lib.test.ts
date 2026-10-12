import { Vector3 } from "three";
import { describe, expect, it } from "vitest";
import {
  baseOf,
  bodyPart,
  cloneName,
  copyNames,
  formatCm,
  formatSigned,
  layoutOffsets,
  matrixFromRowMajor,
  mirrorName,
  partColor,
  splitWarning,
  withDependents,
} from "../src/lib";
import { defaultPlacement, defaultWrap } from "../src/state";

describe("matrixFromRowMajor", () => {
  it("reads the backend's row-major 4x4 (translation in the last column)", () => {
    const m = matrixFromRowMajor([1, 0, 0, 0.1, 0, 1, 0, 0.2, 0, 0, 1, 0.3, 0, 0, 0, 1]);
    const p = new Vector3(1, 2, 3).applyMatrix4(m);
    expect(p.toArray()).toEqual([1.1, 2.2, 3.3]);
  });
  it("maps a rotation about z by 90 degrees x -> y", () => {
    const m = matrixFromRowMajor([0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const p = new Vector3(1, 0, 0).applyMatrix4(m);
    expect(p.x).toBeCloseTo(0);
    expect(p.y).toBeCloseTo(1);
  });
  it("keeps a reflection (mirrored placement) as a negative x scale", () => {
    const m = matrixFromRowMajor([-1, 0, 0, 0.3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    expect(new Vector3(0.1, 0, 0).applyMatrix4(m).x).toBeCloseTo(0.2);
    expect(m.determinant()).toBeLessThan(0);
  });
  it("rejects a matrix of the wrong size", () => {
    expect(() => matrixFromRowMajor([1, 2, 3])).toThrow();
  });
});

describe("layoutOffsets", () => {
  it("places pieces left to right with a gap, in metres", () => {
    const x = layoutOffsets([30, 20, 10], 5, 1.2);
    expect(x[0]).toBeCloseTo(1.2);
    expect(x[1]).toBeCloseTo(1.55);
    expect(x[2]).toBeCloseTo(1.8);
  });
});

describe("formatting", () => {
  it("formats cm and signed errors", () => {
    expect(formatCm(37.254)).toBe("37.25");
    expect(formatCm(null)).toBe("-");
    expect(formatSigned(0.004)).toBe("+0.00");
    expect(formatSigned(-1.5, 1)).toBe("-1.5");
  });
  it("groups left/right volumes into one body part colour", () => {
    expect(bodyPart("UpperArm_L")).toBe("UpperArm");
    expect(partColor("Thigh_R")).toBe(partColor("Thigh_L"));
  });
});

describe("point and piece names", () => {
  const known = new Set(["UpperArm_Outer_L", "UpperArm_Outer_R", "Chest_Left", "Chest_Right", "Chest_Front", "Hip_FrontSide_L"]);
  it("finds the mirrored point", () => {
    expect(mirrorName("UpperArm_Outer_L", known)).toBe("UpperArm_Outer_R");
    expect(mirrorName("Chest_Right", known)).toBe("Chest_Left");
    expect(mirrorName("Chest_Front", known)).toBe("Chest_Front");
    expect(mirrorName("Hip_FrontSide_L", known)).toBeNull();
  });
  it("splits engine warnings into a code and a text", () => {
    expect(splitWarning("SIDE_UNLIKELY: 'x' looks like a front piece")).toEqual({
      code: "SIDE_UNLIKELY",
      text: "'x' looks like a front piece",
    });
    expect(splitWarning("plain").code).toBe("");
  });
  it("names the clone after the opposite side", () => {
    expect(cloneName("FrontPanel_L")).toBe("FrontPanel_R");
    expect(cloneName("FrontPanel", "left")).toBe("FrontPanel_R");
    expect(cloneName("FrontPanel", "right")).toBe("FrontPanel_L");
    expect(cloneName("FrontPanel")).toBe("FrontPanel_L");
  });
  it("names copies X_copy, X_copy2, ... skipping used names, never _L / _R", () => {
    expect(copyNames("Pocket_copy", 3, new Set())).toEqual(["Pocket_copy", "Pocket_copy2", "Pocket_copy3"]);
    expect(copyNames("Pocket_copy", 1, new Set(["Pocket_copy"]))).toEqual(["Pocket_copy2"]);
    expect(copyNames("Pocket_copy", 2, new Set(["Pocket_copy2"]))).toEqual(["Pocket_copy", "Pocket_copy3"]);
  });
  it("deletes a derived piece with the copies made from it, and finds the base piece of a copy", () => {
    const defs = [
      { id: "S_L", source: "S", op: "mirror" as const },
      { id: "S_L_copy", source: "S_L", op: "copy" as const },
      { id: "S_L_copy_copy", source: "S_L_copy", op: "copy" as const },
      { id: "P_copy", source: "P", op: "copy" as const },
    ];
    expect([...withDependents(defs, "S_L")].sort()).toEqual(["S_L", "S_L_copy", "S_L_copy_copy"]);
    expect([...withDependents(defs, "P")].sort()).toEqual(["P", "P_copy"]);
    expect(baseOf(defs, "S_L_copy_copy")).toBe("S");
    expect(baseOf(defs, "P")).toBe("P");
  });
});

describe("default placement", () => {
  it("rolls pieces on the torso, neck, arms and legs (size exact) and wraps the rest around the body part", () => {
    expect(defaultWrap("Chest")).toBe("roll");
    expect(defaultWrap("Hip")).toBe("roll");
    expect(defaultWrap("Neck")).toBe("roll");
    expect(defaultWrap("UpperArm_L")).toBe("roll");
    expect(defaultWrap("Shin_R")).toBe("roll");
    expect(defaultWrap("Head")).toBe("around");
    expect(defaultWrap("Hand_L")).toBe("around");
    expect(defaultWrap(undefined)).toBe("around");
    const p = defaultPlacement("FrontPanel", "Chest_Front", "Chest", "A33");
    expect(p).toMatchObject({ wrap: "roll", reference: "A33", offset_m: null, mirror: false });
  });
});
