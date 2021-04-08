import React from "react";
import cn from "classnames";

import { tilePixelVector, getNodeDelta, SQRT3 } from "../utils/coordinates";

function Building({ building, color }) {
  const cssClass = `bg-white bg-${color.toLowerCase()}-600`;
  const city = building === "CITY";
  const border = city ? "w-8 h-8 border-2" : "w-6 h-6 border-2";
  return (
    <>
      <div
        className={cn(
          "rounded-md node-building absolute border-black",
          cssClass,
          border
        )}
      ></div>
      {city && (
        <div
          className={cn(
            "rounded-md node-building absolute border-black",
            cssClass,
            "w-4 h-4 border-2"
          )}
          style={{ transform: "translateY(0.50rem) translateX(0.50rem)" }}
        ></div>
      )}
    </>
  );
}

export default function Node({
  id,
  center,
  size,
  coordinate,
  direction,
  building,
  color,
}) {
  const [centerX, centerY] = center;
  const w = SQRT3 * size;
  const h = 2 * size;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const [deltaX, deltaY] = getNodeDelta(direction, w, h);
  const x = tileX + deltaX;
  const y = tileY + deltaY;

  return (
    <div
      className="node w-8 h-8 absolute"
      style={{
        left: x,
        top: y,
        transform: `translateY(-50%) translateX(-50%)`,
      }}
      onClick={() => console.log("Clicked node", id)}
    >
      {color && <Building building={building} color={color} />}
    </div>
  );
}
