import React from "react";
import cn from "classnames";

import { tilePixelVector, getEdgeTransform } from "../utils/coordinates";
import useWindowSize from "../utils/useWindowSize";

function Road({ color }) {
  return <div className={cn("road", color)}></div>;
}

export default function Edge({
  id,
  center,
  size,
  coordinate,
  direction,
  color,
  flashing,
  onClick,
}) {
  const { width } = useWindowSize();
  const [centerX, centerY] = center;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const transform = getEdgeTransform(direction, size, width);

  console.log(size);
  return (
    <div
      id={id}
      className={"edge " + direction}
      style={{
        left: tileX,
        top: tileY,
        width: size,
        transform: transform,
      }}
      onClick={onClick}
    >
      {color && <Road color={color} />}
      {flashing && <div className="pulse"></div>}
    </div>
  );
}
