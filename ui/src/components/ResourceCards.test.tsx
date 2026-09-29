import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

import ResourceCards from "./ResourceCards";

const OWN_HAND = {
  P0_WOOD_IN_HAND: 2,
  P0_BRICK_IN_HAND: 0,
  P0_SHEEP_IN_HAND: 1,
  P0_WHEAT_IN_HAND: 0,
  P0_ORE_IN_HAND: 0,
  P0_KNIGHT_IN_HAND: 1,
  P0_VICTORY_POINT_IN_HAND: 0,
  P0_MONOPOLY_IN_HAND: 0,
  P0_YEAR_OF_PLENTY_IN_HAND: 0,
  P0_ROAD_BUILDING_IN_HAND: 0,
};

/** What an opponent's seat looks like once the server knows who is asking. */
const THEIR_HAND = {
  P1_NUM_RESOURCES_IN_HAND: 5,
  P1_NUM_DEVELOPMENT_CARDS_IN_HAND: 2,
};

test("shows every card of a hand it is allowed to read", () => {
  const { container } = render(
    <ResourceCards playerState={OWN_HAND} playerKey="P0" />
  );

  expect(container.querySelectorAll(".wood-cards")).toHaveLength(1);
  expect(container.querySelectorAll(".sheep-cards")).toHaveLength(1);
  expect(container.querySelectorAll(".hidden-cards")).toHaveLength(0);
  expect(screen.getByTitle("1 Knight Card(s)")).toBeInTheDocument();
});

test("shows an opponent's hand as two face-down counts", () => {
  const { container } = render(
    <ResourceCards playerState={THEIR_HAND} playerKey="P1" />
  );

  expect(container.querySelectorAll(".wood-cards")).toHaveLength(0);
  expect(screen.getByTitle("5 Resource Card(s), face down")).toBeInTheDocument();
  expect(
    screen.getByTitle("2 Development Card(s), face down")
  ).toBeInTheDocument();
});

test("draws nothing for an empty hand", () => {
  const { container } = render(
    <ResourceCards
      playerState={{
        P1_NUM_RESOURCES_IN_HAND: 0,
        P1_NUM_DEVELOPMENT_CARDS_IN_HAND: 0,
      }}
      playerKey="P1"
    />
  );

  expect(container.querySelectorAll(".card")).toHaveLength(0);
});
