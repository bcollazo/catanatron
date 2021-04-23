import React from "react";
import cn from "classnames";

import { tilePixelVector, getEdgeTransform, SQRT3 } from "../utils/coordinates";
import useWindowSize from "../utils/useWindowSize";

function Road({ color, size }) {
  const cssClass = `bg-white bg-${color.toLowerCase()}-600`;
  return (
    <div className={cn("road", cssClass)} style={{ width: size * 0.8 }}></div>
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
  const { width } = useWindowSize();
  const [centerX, centerY] = center;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const transform = getEdgeTransform(direction, size, width);

  return (
    <div
      className={"edge absolute " + direction}
      style={{
        left: tileX,
        top: tileY,
        transform: transform,
      }}
      onClick={() => console.log("Clicked edge", id)}
    >
      {color && <Road color={color} size={size} />}
    </div>
  );
}
