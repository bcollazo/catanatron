import React from "react";

import { Circle } from "./Tile";
import { tilePixelVector } from "../utils/coordinates";

export default function Robber({ center, w, size, coordinate }) {
  console.log({ center, w, size, coordinate });
  const [centerX, centerY] = center;
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
