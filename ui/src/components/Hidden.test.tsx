import { describe, expect, test } from "vitest";

import { getCSSBoundaries } from "./Hidden";

describe("getCSSBoundaries", () => {
  test("hides md and below for down breakpoints", () => {
    expect(getCSSBoundaries({ size: "md", direction: "down" })).toEqual({
      xs: "none",
      sm: "none",
      md: "none",
      lg: "block",
      xl: "block",
    });
  });

  test("hides lg and above for up breakpoints", () => {
    expect(getCSSBoundaries({ size: "lg", direction: "up" })).toEqual({
      xs: "block",
      lg: "none",
    });
  });

  test("hides everything for xl down", () => {
    expect(getCSSBoundaries({ size: "xl", direction: "down" })).toBe("none");
  });
});
