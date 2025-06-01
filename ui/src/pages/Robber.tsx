import {
  SQRT3,
  tilePixelVector,
  type CubeCoordinate,
} from "../utils/coordinates";
import { Paper } from "@mui/material";

type RobberProps = {
  center: [number, number];
  size: number;
  coordinate: CubeCoordinate;
};

export default function Robber({ center, size, coordinate }: RobberProps) {
  const [centerX, centerY] = center;
  const w = SQRT3 * size;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const [deltaX, deltaY] = [-w / 2 + w / 8, 0];
  const x = tileX + deltaX;
  const y = tileY + deltaY;

  return (
    <Paper
      elevation={3}
      className="robber number-token"
      style={{
        left: x,
        top: y,
      }}
    >
      R
    </Paper>
  );
}
