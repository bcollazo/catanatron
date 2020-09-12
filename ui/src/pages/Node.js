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

  const color = `bg-white bg-${building[0].toLowerCase()}-700`;
  const city = building[1] === "CITY";
  const border = city ? "w-8 h-8 border-2" : "w-6 h-6 border-2";
  return (
    <>
      <div
        className={cn("node-building absolute border-black", color, border)}
        style={{
          left: x,
          top: y,
          transform: `translateY(-0.75rem) translateX(-0.75rem)`,
        }}
      ></div>
      {city && (
        <div
          className={cn(
            "node-building absolute border-black",
            color,
            "w-4 h-4 border-2"
          )}
          style={{
            left: x,
            top: y,
            transform: `translateY(-0.25rem) translateX(-0.25rem)`,
          }}
        ></div>
      )}
    </>
  );
}
