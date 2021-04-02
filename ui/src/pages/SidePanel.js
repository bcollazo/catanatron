import React, { useEffect, useRef } from "react";
import cn from "classnames";

export default function SidePanel({ state }) {
  const actionRef = useRef(null);

  useEffect(() => {
    if (actionRef.current !== null) {
      actionRef.current.scrollTo(0, actionRef.current.scrollHeight);
    }
  }, [state.actions]);

  const actions = state.actions.map(([color, action_type, value], index) => {
    let text = action_type + " " + value;
    if (action_type === "ROLL") {
      const number = value[0] + value[1];
      text = `${action_type} (${value[0]}, ${value[1]}) = ${number}`;
    } else if (action_type === "MOVE_ROBBER") {
      const color_to_steal_from = value[1];
      text = `${action_type}`;
      if (color_to_steal_from !== null) {
        text += ` / STEAL ${color_to_steal_from}`;
      } else {
        text += ` / DIDNT STEAL`;
      }
    } else if (action_type === "END_TURN") {
      text = `${action_type}`;
    }

    const colorClass = `text-white text-${color.toLowerCase()}-600`;
    return (
      <div key={index} className={colorClass}>
        {color}: {text}
      </div>
    );
  });

  const players = state.players.map(
    ({
      color,
      resource_deck,
      development_deck,
      public_victory_points,
      actual_victory_points,
      played_development_cards,
      has_army,
      has_road,
    }) => {
      const playerClass = `bg-white bg-${color.toLowerCase()}-700 h-42 overflow-auto rounded-md p-2 m-1`;
      return (
        <div key={color} className={playerClass}>
          <div>{JSON.stringify(resource_deck, null, 2)}</div>
          <div>{JSON.stringify(development_deck, null, 2)}</div>
          <div>Victory Points: {public_victory_points}</div>
          {/* <div>AVPs: {actual_victory_points}</div> */}
          <div className={cn({ "font-bold": has_army })}>
            Knights: {played_development_cards["KNIGHT"]}
          </div>
          {has_road && <div className="font-bold">Longest Road</div>}
        </div>
      );
    }
  );

  return (
    <div className="h-full w-300 bg-gray-900 p-4">
      <div ref={actionRef} className="h-64 overflow-auto">
        {actions}
      </div>
      {players}
    </div>
  );
}
