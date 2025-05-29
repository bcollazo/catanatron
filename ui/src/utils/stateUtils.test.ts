import { describe, test, expect } from "vitest";
import { isPlayersTurn } from "./stateUtils";


describe('isPlayersTurn', () => {
    test('isTrue', () => {
        const gameState = {
            bot_colors: ["BLUE"],
            colors: ["BLUE", "RED"],
            current_color: "RED"
        };
        expect(isPlayersTurn(gameState)).toBeTruthy();
    });
    test('isFalse', () => {
        const gameState = {
            bot_colors: ["BLUE"],
            colors: ["BLUE", "RED"],
            current_color: "BLUE",
        };
        expect(isPlayersTurn(gameState)).toBeFalsy();
    })
})
