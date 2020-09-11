import React from "react";
import cn from "classnames";

import {
  tilePixelVector,
  getEdgeDeltaAndTransform,
} from "../utils/coordinates";

export default function Road({
  centerX,
  centerY,
  w,
  h,
  size,
  coordinate,
  direction,
  building,
}) {
  const color = `bg-white bg-${building.color.toLowerCase()}-700`;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const [deltaX, deltaY, transform] = getEdgeDeltaAndTransform(direction, w, h);
  const x = tileX + deltaX;
  const y = tileY + deltaY;
  return (
    <div
      className={cn(
        "road absolute border-2 border-black h-3 w-10",
        color,
        coordinate,
        direction
      )}
      style={{
        left: x,
        top: y,
        transform: transform,
      }}
    ></div>
  );
}
