import React from "react";
import cn from "classnames";

import { tilePixelVector, getNodeDelta, SQRT3 } from "../utils/coordinates";

function Building({ building, color }) {
  const type = building === "CITY" ? "city" : "settlement";
  return <div className={cn(color, type)}></div>;
}

export default function Node({
  id,
  center,
  size,
  coordinate,
  direction,
  building,
  color,
  flashing,
  onClick,
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
      className="node"
      style={{
        width: size * 0.5,
        height: size * 0.5,
        left: x,
        top: y,
        transform: `translateY(-50%) translateX(-50%)`,
      }}
      onClick={onClick}
    >
      {color && <Building building={building} color={color} />}
      {flashing && <div className="pulse"></div>}
      {/* {id} */}
    </div>
  );
}
