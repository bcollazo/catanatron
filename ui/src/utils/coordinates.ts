// Helpers for implementing https://www.redblobgames.com/grids/hexagons/
// TODO - is this still needed with new CSS features?

import type { Direction } from "./api.types";

export type CubeCoordinate = [number, number, number];
type RadialCoordinate = {
  q: number;
  r: number;
};

// Gives center coordinate for tile.
export function tilePixelVector(
  coordinate: CubeCoordinate,
  size: number,
  centerX: number,
  centerY: number
) {
  const hex = cubeToAxial(coordinate);
  return [
    size * (SQRT3 * hex.q + (SQRT3 / 2) * hex.r) + centerX,
    size * (3 / 2) * hex.r + centerY,
  ];
}

export function cubeToAxial(cube: CubeCoordinate): RadialCoordinate {
  return { q: cube[0], r: cube[2] };
}

// TODO - why are EAST and WEST not included here?
export function getNodeDelta(direction: Direction, w: number, h: number) {
  switch (direction) {
    case "NORTH":
      return [0, -h / 2];
    case "NORTHEAST":
      return [w / 2, -h / 4];
    case "SOUTHEAST":
      return [w / 2, h / 4];
    case "SOUTH":
      return [0, h / 2];
    case "SOUTHWEST":
      return [-w / 2, h / 4];
    case "NORTHWEST":
      return [-w / 2, -h / 4];
    default:
      throw Error("Unknown direction " + direction);
  }
}

export function getEdgeTransform(direction: Direction, size: number) {
  const distanceToEdge = size * 0.865;
  const translate = (deg: number) =>
    `translateX(-50%) translateY(-50%) rotate(${deg}deg) translateY(${-distanceToEdge}px)`;
  switch (direction) {
    case "NORTHEAST":
      return `${translate(30)}`;
    case "EAST":
      return `${translate(90)}`;
    case "SOUTHEAST":
      return `${translate(150)}`;
    case "SOUTHWEST":
      return `${translate(210)}`;
    case "WEST":
      return `${translate(270)}`;
    case "NORTHWEST":
      return `${translate(330)}`;
    default:
      throw Error("Unknown direction " + direction);
  }
}

export const SQRT3 = 1.73205080757;
