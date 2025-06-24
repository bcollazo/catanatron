import cn from "classnames";

import type { Color, Direction } from "../utils/api.types";
import { tilePixelVector, getEdgeTransform, type CubeCoordinate } from "../utils/coordinates";

function Road({ color }: { color: Color}) {
  return <div className={cn("road", color)}></div>;
}

export type EdgeId = `${number},${number}`;

export const toEdgeId = (id: [number, number]): EdgeId => `${id[0]},${id[1]}`;

type EdgeProps = {
  id: `${number},${number}`;
  center: [number, number];
  coordinate: CubeCoordinate;
  size: number;
  direction: Direction;
  color: Color | null;
  flashing: boolean;
  onClick: React.MouseEventHandler;
};

export default function Edge({
  id,
  center,
  size,
  coordinate,
  direction,
  color,
  flashing,
  onClick,
}: EdgeProps) {
  const [centerX, centerY] = center;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const transform = getEdgeTransform(direction, size); // TODO this used to include windowSize; should we bring that back?

  return (
    <div
      id={id}
      className={"edge " + direction}
      style={{
        left: tileX,
        top: tileY,
        width: size * 0.9,
        transform: transform,
      }}
      onClick={onClick}
    >
      {color && <Road color={color} />}
      {flashing && <div className="pulse"></div>}
    </div>
  );
}
