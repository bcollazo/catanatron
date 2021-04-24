import React from "react";
import cn from "classnames";

import "./PlayerStateBox.scss";

export function ResourceCards({ playerState }) {
  const numDevCards = Object.values(playerState.development_deck).reduce(
    (a, b) => a + b,
    0
  );
  const resdeck = playerState.resource_deck;
  return (
    <div className="resource-cards" title="Resource Cards">
      {resdeck.WOOD !== 0 && (
        <div className="wood-cards center-text">{resdeck.WOOD}</div>
      )}
      {resdeck.BRICK !== 0 && (
        <div className="brick-cards center-text">{resdeck.BRICK}</div>
      )}
      {resdeck.SHEEP !== 0 && (
        <div className="sheep-cards center-text">{resdeck.SHEEP}</div>
      )}
      {resdeck.WHEAT !== 0 && (
        <div className="wheat-cards center-text">{resdeck.WHEAT}</div>
      )}
      {resdeck.ORE !== 0 && (
        <div className="ore-cards center-text">{resdeck.ORE}</div>
      )}
      {numDevCards !== 0 && (
        <div className="dev-cards center-text" title="Development Cards">
          {numDevCards}
        </div>
      )}
    </div>
  );
}

export default function PlayerStateBox({ playerState, longestRoad }) {
  return (
    <div className="player-state-box">
      <ResourceCards playerState={playerState} />
      <div className="scores">
        <div
          className={cn("num-knights center-text", {
            has_army: playerState.has_army,
          })}
          title="Knights Played"
        >
          <span>{playerState.played_development_cards.KNIGHT}</span>
          <small>knights</small>
        </div>
        <div
          className={cn("num-roads center-text", {
            has_road: playerState.has_road,
          })}
          title="Longest Road"
        >
          {longestRoad}
          <small>roads</small>
        </div>
        <div
          className={cn("victory-points center-text", {
            has_road: playerState.actual_victory_points >= 10,
          })}
          title="Victory Points"
        >
          {playerState.actual_victory_points}
          <small>VPs</small>
        </div>
      </div>
    </div>
  );
}
