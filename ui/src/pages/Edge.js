import React from "react";
import cn from "classnames";

import {
  tilePixelVector,
  getEdgeDeltaAndTransform,
  SQRT3,
} from "../utils/coordinates";

function Road({ color }) {
  const cssClass = `bg-white bg-${color.toLowerCase()}-600`;
  return (
    <div
      className={cn(
        "road absolute rounded-sm border-2 border-black h-3 w-10",
        cssClass
      )}
    ></div>
  );
}

export default function Edge({
  id,
  center,
  size,
  coordinate,
  direction,
  color,
}) {
  const [centerX, centerY] = center;
  const w = SQRT3 * size;
  const h = 2 * size;
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
