import { describe, test, expect } from "vitest";
import {
  getHumanColor,
  isPlayersTurn,
  playerKey,
  type GameState,
} from "./stateUtils";

describe("isPlayersTurn", () => {
  test("isTrue", () => {
    const gameState: GameState = {
      bot_colors: ["BLUE"],
      colors: ["BLUE", "RED"],
      current_color: "RED",
    };
    expect(isPlayersTurn(gameState)).toBeTruthy();
  });
  test("isFalse", () => {
    const gameState: GameState = {
      bot_colors: ["BLUE"],
      colors: ["BLUE", "RED"],
      current_color: "BLUE",
    };
    expect(isPlayersTurn(gameState)).toBeFalsy();
  });
});

describe("playerKey", () => {
  test("valid color", () => {
    const gameState: GameState = {
      bot_colors: ["BLUE"],
      colors: ["BLUE", "RED"],
      current_color: "BLUE",
    };
    expect(playerKey(gameState, "RED")).toBe("P1");
  });
  test("bot color", () => {
    const gameState: GameState = {
      bot_colors: ["BLUE"],
      colors: ["BLUE", "RED"],
      current_color: "BLUE",
    };
    expect(playerKey(gameState, "BLUE")).toBe("P0");
  });
});

describe("getHumanColor", () => {
  test("single human", () => {
    const gameState: GameState = {
      bot_colors: ["BLUE"],
      colors: ["BLUE", "RED"],
      current_color: "BLUE",
    };
    expect(getHumanColor(gameState)).toBe("RED");
  });
  test("only bots", () => {
    const gameState: GameState = {
      bot_colors: ["BLUE", "RED"],
      colors: ["BLUE", "RED"],
      current_color: "BLUE",
    };
    expect(getHumanColor(gameState)).toBeUndefined();
  });
});
