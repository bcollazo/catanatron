import cn from "classnames";

import { tilePixelVector, getNodeDelta, SQRT3, type CubeCoordinate } from "../utils/coordinates";
import type { Building, Color, Direction } from "../utils/api.types";

type BuildingProps<T = Color | null> = {
  color: T;
  building: T extends Color ? Building : null;
};

function Building({ building, color }: BuildingProps) {
  const type = building === "CITY" ? "city" : "settlement";
  return <div className={cn(color, type)}></div>;
}

type NodeProps = {
  center: [number, number];
  size: number;
  coordinate: CubeCoordinate;
  direction: Direction;
  building: Building | null;
  color: Color | null;
  flashing: boolean;
  onClick: React.MouseEventHandler<HTMLDivElement>;
};

export default function Node({
  center,
  size,
  coordinate,
  direction,
  building,
  color,
  flashing,
  onClick,
}: NodeProps) {
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
