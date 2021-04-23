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

const SMALL_BREAKPOINT = 576;

export function getEdgeTransform(direction, size, viewportWidth) {
  const stroke = viewportWidth < SMALL_BREAKPOINT ? 8 : 12;
  const traslate = `translate(-${(size * 0.8) / 2}px, -${size - stroke / 4}px)`;
  switch (direction) {
    case "EAST":
      return `rotate(90deg) ${traslate}`;
    case "NORTHEAST":
      return `rotate(30deg) ${traslate}`;
    case "SOUTHEAST":
      return `rotate(150deg) ${traslate}`;
    case "WEST":
      return `rotate(-90deg) ${traslate}`;
    case "SOUTHWEST":
      return `rotate(210deg) ${traslate}`;
    case "NORTHWEST":
      return `rotate(-30deg) ${traslate}`;
    default:
      throw Error("Unkown direction " + direction);
  }
}

export const SQRT3 = 1.73205080757;
