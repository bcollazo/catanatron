import { render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";

import { makeGameState } from "../test/fixtures";
import Board from "./Board";

test("renders the nested tile, node, and edge shapes emitted by GameEncoder", () => {
  const onNodeClick = vi.fn();
  const onEdgeClick = vi.fn();
  const onTileClick = vi.fn();
  const gameState = makeGameState();

  const { container } = render(
    <Board
      width={1200}
      height={900}
      buildOnNodeClick={() => onNodeClick}
      buildOnEdgeClick={() => onEdgeClick}
      handleTileClick={onTileClick}
      replayMode={false}
      gameState={gameState}
      isMobile={false}
      show
      isMovingRobber={false}
    />
  );

  expect(container.querySelectorAll(".tile")).toHaveLength(3);
  expect(container.querySelectorAll(".node")).toHaveLength(1);
  expect(container.querySelectorAll(".edge")).toHaveLength(1);
  expect(container.querySelectorAll(".robber")).toHaveLength(1);
  expect(screen.getByText("11")).toBeInTheDocument();
  expect(screen.getByText("3:1")).toBeInTheDocument();
});
