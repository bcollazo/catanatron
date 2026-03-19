import { Paper } from "@mui/material";
import { type PlayerState } from "../utils/api.types";
import { type Card } from "../utils/api.types";
import { useSkin } from "../SkinContext";

// TODO - do we need to split the SCSS for this component?
import "./PlayerStateBox.scss";

type ResourceType = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";

const RESOURCE_SYMBOL_KEYS: Record<ResourceType, "WOOD_SYMBOL" | "BRICK_SYMBOL" | "SHEEP_SYMBOL" | "WHEAT_SYMBOL" | "ORE_SYMBOL"> = {
  WOOD: "WOOD_SYMBOL",
  BRICK: "BRICK_SYMBOL",
  SHEEP: "SHEEP_SYMBOL",
  WHEAT: "WHEAT_SYMBOL",
  ORE: "ORE_SYMBOL",
};

export default function ResourceCards({ playerState, playerKey }: { playerState: PlayerState; playerKey: string }) {
  const { skin, assets } = useSkin();
  const amount = (card: Card) => playerState[`${playerKey}_${card}_IN_HAND`];
  const isClassic = skin === "classic";

  const renderResourceCard = (resource: ResourceType, cssClass: string) => {
    const count = amount(resource);
    if (count === 0) return null;

    if (isClassic) {
      const symbol = assets[RESOURCE_SYMBOL_KEYS[resource]];
      return (
        <div className="classic-card card" key={resource}>
          <Paper>
            <img src={symbol} alt={resource} className="card-symbol" />
            <span className="card-count">{count}</span>
          </Paper>
        </div>
      );
    }

    return (
      <div className={`${cssClass} center-text card`} key={resource}>
        <Paper>{count}</Paper>
      </div>
    );
  };

  return (
    <div className="resource-cards" title="Resource Cards">
      {renderResourceCard("WOOD", "wood-cards")}
      {renderResourceCard("BRICK", "brick-cards")}
      {renderResourceCard("SHEEP", "sheep-cards")}
      {renderResourceCard("WHEAT", "wheat-cards")}
      {renderResourceCard("ORE", "ore-cards")}
      <div className="separator"></div>
      {amount("VICTORY_POINT") !== 0 && (
        <div
          className={`dev-cards center-text card${isClassic ? " classic-dev-card" : ""}`}
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
          className={`dev-cards center-text card${isClassic ? " classic-dev-card" : ""}`}
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
          className={`dev-cards center-text card${isClassic ? " classic-dev-card" : ""}`}
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
          className={`dev-cards center-text card${isClassic ? " classic-dev-card" : ""}`}
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
          className={`dev-cards center-text card${isClassic ? " classic-dev-card" : ""}`}
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
