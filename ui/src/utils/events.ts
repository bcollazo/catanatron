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
