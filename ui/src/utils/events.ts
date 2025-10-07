import { type KeyboardEvent } from "react";

export type InteractionEvent =
  | React.KeyboardEvent
  | React.MouseEvent
  | React.TouchEvent;
type KeydownEvent = React.KeyboardEvent & { type: "keydown" };

export const isKeyDownEvent = (
  event: InteractionEvent
): event is KeydownEvent => event && event.type === "keydown";

export const isTabOrShift = (event: InteractionEvent) =>
  isKeyDownEvent(event) && (event.key === "Tab" || event.key === "Shift");

export function allowOnlyNumberKeys(e: KeyboardEvent<HTMLInputElement>) {
  const allowedKeys = [
    "Backspace",
    "ArrowLeft",
    "ArrowRight",
    "Delete",
    "Tab",
    "Enter",
  ];

  if (!/[0-9]/.test(e.key) && !allowedKeys.includes(e.key)) {
    e.preventDefault();
  }
}
