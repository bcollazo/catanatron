import React from "react";
import cn from "classnames";

import {
  tilePixelVector,
  getEdgeDeltaAndTransform,
} from "../utils/coordinates";

function Road({ building }) {
  const color = `bg-white bg-${building[0].toLowerCase()}-700`;
  return (
    <div
      className={cn("road absolute border-2 border-black h-3 w-10", color)}
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
  building,
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
      {building && <Road building={building} />}
    </div>
  );
}
