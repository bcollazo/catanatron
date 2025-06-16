import { describe, test, expect } from "vitest";
import {
  cubeToAxial,
  tilePixelVector,
  getNodeDelta,
  getEdgeTransform,
} from "./coordinates";
import { type Direction } from "./api.types";

describe("tilePixelVector", () => {
  test("zeroed", () => {
    const centerCoord = tilePixelVector([0, 0, 0], 10, 0, 0);
    expect(centerCoord).toEqual([0, 0]);
  });
  test("center offset by 1s", () => {
    const centerCoord = tilePixelVector([0, 0, 0], 10, 1, 1);
    expect(centerCoord).toEqual([1, 1]);
  });
  test("position shifted by 1s", () => {
    const centerCoord = tilePixelVector([1, 1, 1], 10, 0, 0);
    expect(centerCoord).toEqual([25.98076211355, 15]);
  });
});

describe("cubeToAxial", () => {
  test("zeroes", () => {
    expect(cubeToAxial([0, 0, 0])).toEqual({ q: 0, r: 0 });
  });
  test("ones", () => {
    expect(cubeToAxial([1, 1, 1])).toEqual({ q: 1, r: 1 });
  });
});

describe("getNodeDelta", () => {
  test("valid directions", () => {
    const directionCalcs = [
      ["NORTH", [0, -2.5]],
      ["SOUTH", [0, 2.5]],
      ["NORTHWEST", [-2.5, -1.25]],
      ["NORTHEAST", [2.5, -1.25]],
      ["SOUTHWEST", [-2.5, 1.25]],
      ["SOUTHEAST", [2.5, 1.25]],
    ] as const;
    for (const [direction, expected] of directionCalcs) {
      expect(getNodeDelta(direction, 5, 5)).toEqual(expected);
    }
  });
  test("invalid directions", () => {
    for (const direction of ["EAST", "WEST"] as const) {
      expect(() => getNodeDelta(direction, 5, 5)).toThrowError(
        "Unknown direction " + direction
      );
    }
  });
});

describe("getEdgeTransform", () => {
  test("valid directions", () => {
    expect(getEdgeTransform("NORTHEAST", 5)).toEqual(
      "translateX(-50%) translateY(-50%) rotate(30deg) translateY(-4.325px)"
    );
    expect(getEdgeTransform("EAST", 5)).toEqual(
      "translateX(-50%) translateY(-50%) rotate(90deg) translateY(-4.325px)"
    );
  });
  test("invalid direction", () => {
    for (const direction of ["SOUTH", "BLA" as Direction] as const) {
      expect(() => getEdgeTransform(direction, 5)).toThrowError(
        `Unknown direction ${direction}`
      );
    }
  });
});
