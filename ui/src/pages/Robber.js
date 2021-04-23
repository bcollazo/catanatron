import React from "react";

import { NumberToken } from "./Tile";
import { SQRT3, tilePixelVector } from "../utils/coordinates";

export default function Robber({ center, size, coordinate }) {
  const [centerX, centerY] = center;
  const w = SQRT3 * size;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const [deltaX, deltaY] = [-w / 2 + w / 8, 0];
  const x = tileX + deltaX;
  const y = tileY + deltaY;

  return (
    <NumberToken
      className="robber"
      size={size}
      style={{
        left: x,
        top: y,
      }}
    >
      R
    </NumberToken>
  );
}
