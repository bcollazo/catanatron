import React from "react";
import cn from "classnames";

import { tilePixelVector, getNodeDelta } from "../utils/coordinates";

export default function Node({
  centerX,
  centerY,
  w,
  h,
  size,
  coordinate,
  direction,
  building,
}) {
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const [deltaX, deltaY] = getNodeDelta(direction, w, h);
  const x = tileX + deltaX;
  const y = tileY + deltaY;

  const color = `bg-white bg-${building.color.toLowerCase()}-700`;
  return (
    <div
      className={cn(
        "node-building absolute w-6 h-6 border-2 border-black",
        color
      )}
      style={{
        left: x,
        top: y,
        transform: `translateY(-0.75rem) translateX(-0.75rem)`,
      }}
    ></div>
  );
}
