import React from "react";
import cn from "classnames";

import {
  tilePixelVector,
  getEdgeDeltaAndTransform,
} from "../utils/coordinates";

function Road({ color }) {
  const cssClass = `bg-white bg-${color.toLowerCase()}-700`;
  return (
    <div
      className={cn("road absolute border-2 border-black h-3 w-10", cssClass)}
    ></div>
  );
}

export default function Edge({
  id,
  centerX,
  centerY,
  w,
  h,
  size,
  coordinate,
  direction,
  color,
}) {
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const [deltaX, deltaY, transform] = getEdgeDeltaAndTransform(direction, w, h);
  const x = tileX + deltaX;
  const y = tileY + deltaY;

  return (
    <div
      className="edge absolute h-3 w-10"
      style={{
        left: x,
        top: y,
        transform: transform,
      }}
      onClick={() => console.log("Clicked edge", id)}
    >
      {color && <Road color={color} />}
    </div>
  );
}
