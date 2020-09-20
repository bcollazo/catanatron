import React, { useEffect, useRef } from "react";

export default function SidePanel({ state }) {
  const actionRef = useRef(null);

  useEffect(() => {
    if (actionRef.current !== null) {
      actionRef.current.scrollTo(0, actionRef.current.scrollHeight);
    }
  }, [state.actions]);

  const actions = state.actions.map(([player, action_type, value], index) => {
    const { color } = player;
    let text = action_type + " " + value;
    if (action_type === "ROLL") {
      const number = value[0] + value[1];
      text = `${action_type} (${value[0]}, ${value[1]}) = ${number}`;
    } else if (action_type === "MOVE_ROBBER") {
      const player = value[1];
      text = `${action_type}`;
      if (player !== null) {
        text += ` / STEAL ${player.color}`;
      } else {
        text += ` / DIDNT STEAL`;
      }
    } else if (action_type === "END_TURN") {
      text = `${action_type}`;
    }

    const colorClass = `text-white text-${color.toLowerCase()}-700`;
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
      public_victory_points,
      actual_victory_points,
    }) => {
      const colorClass = `bg-white bg-${color.toLowerCase()}-700 h-24`;
      return (
        <div key={color} className={colorClass}>
          <div>{JSON.stringify(resource_deck, null, 2)}</div>
          <div>VPs: {public_victory_points}</div>
          <div>AVPs: {actual_victory_points}</div>
        </div>
      );
    }
  );

  return (
    <div className="h-full lg:w-1/3 w-1/2 bg-gray-900 p-6">
      <div ref={actionRef} className="h-64 overflow-auto">
        {actions}
      </div>
      {players}
    </div>
  );
}
