import { Paper } from "@mui/material";
import { type PlayerState } from "../utils/api.types";
import { type Card } from "../utils/api.types";

// TODO - do we need to split the SCSS for this component?
import "./PlayerStateBox.scss";

export default function ResourceCards({ playerState, playerKey }: { playerState: PlayerState; playerKey: string }) {
  const amount = (card: Card) => playerState[`${playerKey}_${card}_IN_HAND`];

  // Seen from another seat, this hand arrives as two counts instead of cards.
  if (amount("WOOD") === undefined) {
    return (
      <FaceDownCards
        resources={playerState[`${playerKey}_NUM_RESOURCES_IN_HAND`] ?? 0}
        developments={
          playerState[`${playerKey}_NUM_DEVELOPMENT_CARDS_IN_HAND`] ?? 0
        }
      />
    );
  }

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
      <div className="separator"></div>
      {amount("VICTORY_POINT") !== 0 && (
        <div
          className="dev-cards center-text card"
          title={amount("VICTORY_POINT") + " Victory Point Card(s)"}
        >
          <Paper>
            <span>{amount("VICTORY_POINT")}</span>
            <span>VP</span>
          </Paper>
        </div>
      )}
      {amount("KNIGHT") !== 0 && (
        <div
          className="dev-cards center-text card"
          title={amount("KNIGHT") + " Knight Card(s)"}
        >
          <Paper>
            <span>{amount("KNIGHT")}</span>
            <span>KN</span>
          </Paper>
        </div>
      )}
      {amount("MONOPOLY") !== 0 && (
        <div
          className="dev-cards center-text card"
          title={amount("MONOPOLY") + " Monopoly Card(s)"}
        >
          <Paper>
            <span>{amount("MONOPOLY")}</span>
            <span>MO</span>
          </Paper>
        </div>
      )}
      {amount("YEAR_OF_PLENTY") !== 0 && (
        <div
          className="dev-cards center-text card"
          title={amount("YEAR_OF_PLENTY") + " Year of Plenty Card(s)"}
        >
          <Paper>
            <span>{amount("YEAR_OF_PLENTY")}</span>
            <span>YP</span>
          </Paper>
        </div>
      )}
      {amount("ROAD_BUILDING") !== 0 && (
        <div
          className="dev-cards center-text card"
          title={amount("ROAD_BUILDING") + " Road Building Card(s)"}
        >
          <Paper>
            <span>{amount("ROAD_BUILDING")}</span>
            <span>RB</span>
          </Paper>
        </div>
      )}
    </div>
  );
}

function FaceDownCards({
  resources,
  developments,
}: {
  resources: number;
  developments: number;
}) {
  return (
    <div className="resource-cards" title="Resource Cards">
      {resources !== 0 && (
        <div
          className="hidden-cards center-text card"
          title={resources + " Resource Card(s), face down"}
        >
          <Paper>{resources}</Paper>
        </div>
      )}
      {developments !== 0 && (
        <div
          className="dev-cards center-text card"
          title={developments + " Development Card(s), face down"}
        >
          <Paper>
            <span>{developments}</span>
            <span>DEV</span>
          </Paper>
        </div>
      )}
    </div>
  );
}
