import React from "react";

import { Circle } from "./Tile";
import { SQRT3, tilePixelVector } from "../utils/coordinates";

export default function Robber({ center, size, coordinate }) {
  const [centerX, centerY] = center;
  const w = SQRT3 * size;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const [deltaX, deltaY] = [-w / 4, 0];
  const x = tileX + deltaX;
  const y = tileY + deltaY;

  return (
    <Circle
      className="bg-gray-900 text-white absolute"
      style={{
        left: x,
        top: y,
        transform: `translateY(-0.75rem) translateX(-0.75rem)`,
      }}
    >
      R
    </Circle>
  );
}
