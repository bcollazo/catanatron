// Helpers for implementing https://www.redblobgames.com/grids/hexagons/

// Gives center coordinate for tile.
export function tilePixelVector(coordinate, size, centerX, centerY) {
  const hex = cubeToAxial(coordinate);
  return [
    size * (SQRT3 * hex.q + (SQRT3 / 2) * hex.r) + centerX,
    size * (3 / 2) * hex.r + centerY,
  ];
}

export function cubeToAxial(cube) {
  return { q: cube[0], r: cube[2] };
}
export function getNodeDelta(direction, w, h) {
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
      throw Error("Unkown direction " + direction);
  }
}

export function getEdgeDeltaAndTransform(direction, w, h, scale=1) {
  switch (direction) {
    case "EAST":
      return [
        w / 2,
        -h / 4,
        `rotate(90deg) translateX(1.5rem) translateY(1.25rem) scale(${scale})`,
      ];
    case "NORTHEAST":
      return [
        w / 2,
        -h / 4,
        `rotate(-150deg) translateX(3rem) translateY(-0.25rem) scale(${scale})`,
      ];
    case "SOUTHEAST":
      return [
        w / 2,
        h / 4,
        `rotate(150deg) translateX(2.50rem) translateY(0.75rem) scale(${scale})`,
      ];
    case "WEST":
      return [
        -w / 2,
        h / 4,
        `rotate(-90deg) translateX(2rem) translateY(-1.25rem) scale(${scale})`,
      ];
    case "SOUTHWEST":
      return [
        -w / 2,
        h / 4,
        `rotate(30deg) translateX(0.50rem) translateY(0.50rem) scale(${scale})`,
      ];
    case "NORTHWEST":
      return [
        -w / 2,
        -h / 4,
        `rotate(-30deg) translateX(1rem) translateY(-1rem) scale(${scale})`,
      ];
    default:
      throw Error("Unkown direction " + direction);
  }
}

export const SQRT3 = 1.73205080757;
