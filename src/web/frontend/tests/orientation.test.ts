import { describe, expect, it } from "vitest";
import { normAngle } from "../src/ui/orientation";

describe("normAngle (rotate buttons)", () => {
  it("keeps angles in (-180, 180]", () => {
    expect(normAngle(0)).toBe(0);
    expect(normAngle(190)).toBe(-170);
    expect(normAngle(-180)).toBe(180);
    expect(normAngle(180)).toBe(180);
    expect(normAngle(-95 - 90)).toBe(175);
    expect(normAngle(720 + 15)).toBe(15);
  });
});
