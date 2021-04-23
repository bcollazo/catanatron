import React from "react";
import cn from "classnames";

import { tilePixelVector, getEdgeTransform } from "../utils/coordinates";
import useWindowSize from "../utils/useWindowSize";

function Road({ color }) {
  return <div className={cn("road", color)}></div>;
}

const SMALL_BREAKPOINT = 576;

export default function Edge({
  id,
  center,
  size,
  coordinate,
  direction,
  color,
}) {
  const { width } = useWindowSize();
  const stroke = width < SMALL_BREAKPOINT ? 8 : 12;
  const [centerX, centerY] = center;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const transform = getEdgeTransform(direction, size, width);

  return (
    <div
      className={"edge " + direction}
      style={{
        left: tileX,
        top: tileY,
        width: size,
        height: stroke,
        transform: transform,
      }}
      onClick={() => console.log("Clicked edge", id)}
    >
      {color && <Road color={color} />}
    </div>
  );
}
