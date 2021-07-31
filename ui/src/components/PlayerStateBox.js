import React from "react";
import cn from "classnames";

import "./PlayerStateBox.scss";
import { Paper } from "@material-ui/core";

export function ResourceCards({ playerState, playerKey }) {
  const amount = (card) => playerState[`${playerKey}_${card}_IN_HAND`];
  return (
    <div className="resource-cards" title="Resource Cards">
      {amount("WOOD") !== 0 && (
        <div className="wood-cards center-text card">
          <Paper>{amount("WOOD")}</Paper>
        </div>
      )}
      {amount("BRICK") !== 0 && (
        <div className="brick-cards center-text card">
          <Paper>{amount("BRICK")}</Paper>
        </div>
      )}
      {amount("SHEEP") !== 0 && (
        <div className="sheep-cards center-text card">
          <Paper>{amount("SHEEP")}</Paper>
        </div>
      )}
      {amount("WHEAT") !== 0 && (
        <div className="wheat-cards center-text card">
          <Paper>{amount("WHEAT")}</Paper>
        </div>
      )}
      {amount("ORE") !== 0 && (
        <div className="ore-cards center-text card">
          <Paper>{amount("ORE")}</Paper>
        </div>
      )}
      {/* {numDevCards !== 0 && (
        <div className="dev-cards center-text" title="Development Cards">
          {numDevCards}
        </div>
      )} */}
    </div>
  );
}

export default function PlayerStateBox({ playerState, playerKey, color }) {
  const actualVps = playerState[`${playerKey}_ACTUAL_VICTORY_POINTS`];
  return (
    <div className={cn("player-state-box foreground", color)}>
      <ResourceCards playerState={playerState} playerKey={playerKey} />
      <div className="scores">
        <div
          className={cn("num-knights center-text", {
            bold: playerState[`${playerKey}_HAS_ARMY`],
          })}
          title="Knights Played"
        >
          <span>{playerState[`${playerKey}_PLAYED_KNIGHT`]}</span>
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
