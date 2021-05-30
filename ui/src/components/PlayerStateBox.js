import React from "react";
import cn from "classnames";

import "./PlayerStateBox.scss";

export function ResourceCards({ playerState, playerKey }) {
  const amount = (card) => playerState[`${playerKey}_${card}_IN_HAND`];
  return (
    <div className="resource-cards" title="Resource Cards">
      {amount("WOOD") !== 0 && (
        <div className="wood-cards center-text">{amount("WOOD")}</div>
      )}
      {amount("BRICK") !== 0 && (
        <div className="brick-cards center-text">{amount("BRICK")}</div>
      )}
      {amount("SHEEP") !== 0 && (
        <div className="sheep-cards center-text">{amount("SHEEP")}</div>
      )}
      {amount("WHEAT") !== 0 && (
        <div className="wheat-cards center-text">{amount("WHEAT")}</div>
      )}
      {amount("ORE") !== 0 && (
        <div className="ore-cards center-text">{amount("ORE")}</div>
      )}
      {/* {numDevCards !== 0 && (
        <div className="dev-cards center-text" title="Development Cards">
          {numDevCards}
        </div>
      )} */}
    </div>
  );
}

export default function PlayerStateBox({ playerState, playerKey }) {
  const actualVps = playerState[`${playerKey}_ACTUAL_VPS`];
  return (
    <div className="player-state-box">
      <ResourceCards playerState={playerState} playerKey={playerKey} />
      <div className="scores">
        <div
          className={cn("num-knights center-text", {
            bold: playerState[`${playerKey}_HAS_ARMY`],
          })}
          title="Knights Played"
        >
          <span>{playerState[`${playerKey}_KNIGHT_PLAYED`]}</span>
          <small>knights</small>
        </div>
        <div
          className={cn("num-roads center-text", {
            bold: playerState[`${playerKey}_HAS_ROAD`],
          })}
          title="Longest Road"
        >
          {playerState[`${playerKey}_LONGEST_ROAD_LENGTH`]}
          <small>roads</small>
        </div>
        <div
          className={cn("victory-points center-text", {
            bold: actualVps >= 10,
          })}
          title="Victory Points"
        >
          {actualVps}
          <small>VPs</small>
        </div>
      </div>
    </div>
  );
}
