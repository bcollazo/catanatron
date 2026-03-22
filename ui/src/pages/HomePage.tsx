import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Button,
  Checkbox,
  IconButton,
  MenuItem,
  Select,
  Slider,
  Tooltip,
} from "@mui/material";
import HelpOutlineRoundedIcon from "@mui/icons-material/HelpOutlineRounded";
import { GridLoader } from "react-spinners";
import {
  createGame,
  type MapTemplate,
  type PlayerArchetype,
} from "../utils/apiClient";

import "./HomePage.scss";

const PLAYER_ARCHETYPES: Array<{
  value: PlayerArchetype;
  label: string;
}> = [
  { value: "HUMAN", label: "Human" },
  { value: "RANDOM", label: "Random" },
  { value: "CATANATRON", label: "Catanatron" },
  { value: "WEIGHTED_RANDOM", label: "Weighted Random" },
];

const MAP_TEMPLATES: MapTemplate[] = ["BASE", "MINI", "TOURNAMENT"];
const PLAYER_COLORS = ["RED", "BLUE", "ORANGE", "WHITE"] as const;

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const [mapTemplate, setMapTemplate] = useState<MapTemplate>("BASE");
  const [vpsToWin, setVpsToWin] = useState(10);
  const [discardLimit, setDiscardLimit] = useState(7);
  const [friendlyRobber, setFriendlyRobber] = useState(false);
  const [players, setPlayers] = useState<PlayerArchetype[]>([
    "CATANATRON",
    "RANDOM",
  ]);
  const navigate = useNavigate();
  const humanCount = players.filter((player) => player === "HUMAN").length;
  const hasTooManyHumans = humanCount > 1;

  const handlePlayerChange = (index: number, value: PlayerArchetype) => {
    if (
      value === "HUMAN" &&
      players[index] !== "HUMAN" &&
      humanCount >= 1
    ) {
      return;
    }

    setPlayers((current) =>
      current.map((player, playerIndex) =>
        playerIndex === index ? value : player
      )
    );
  };

  const handleAddPlayer = () => {
    setPlayers((current) =>
      current.length >= 4 ? current : [...current, "WEIGHTED_RANDOM"]
    );
  };

  const handleRemovePlayer = (index: number) => {
    setPlayers((current) =>
      current.length <= 2
        ? current
        : current.filter((_, playerIndex) => playerIndex !== index)
    );
  };

  const handleCreateGame = async () => {
    if (hasTooManyHumans) {
      return;
    }

    setLoading(true);
    const gameId = await createGame({
      players,
      mapTemplate,
      vpsToWin,
      discardLimit,
      friendlyRobber,
    });
    setLoading(false);
    navigate("/games/" + gameId);
  };

  return (
    <div className="home-page">
      <h1 className="logo">Catanatron</h1>

      <div className="switchable">
        {!loading ? (
          <>
            <div className="setup-card">
              <p className="setup-note">Open hands. Random discard choice.</p>

              <div className="control-group">
                <div className="control-header">
                  <span>Map Template</span>
                  <strong>{mapTemplate}</strong>
                </div>
                <div className="map-template-buttons">
                  {MAP_TEMPLATES.map((value) => (
                    <Button
                      key={value}
                      variant="contained"
                      onClick={() => setMapTemplate(value)}
                      className={`choice-button ${
                        mapTemplate === value ? "selected" : ""
                      }`}
                    >
                      {value}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="control-row">
                <div className="control-group compact-control">
                  <div className="control-header">
                    <span>Points to Win</span>
                    <strong>{vpsToWin}</strong>
                  </div>
                  <Slider
                    value={vpsToWin}
                    min={3}
                    max={20}
                    step={1}
                    marks
                    valueLabelDisplay="auto"
                    onChange={(_, value) => setVpsToWin(value as number)}
                  />
                </div>

                <div className="control-group compact-control">
                  <div className="control-header">
                    <span>Card Discard Limit</span>
                    <strong>{discardLimit}</strong>
                  </div>
                  <Slider
                    value={discardLimit}
                    min={5}
                    max={20}
                    step={1}
                    marks
                    valueLabelDisplay="auto"
                    onChange={(_, value) => setDiscardLimit(value as number)}
                  />
                </div>

                <div className="control-group compact-control switch-control">
                  <div className="control-header">
                    <span className="inline-title">
                      Friendly Robber
                      <Tooltip
                        title="Prevent robber placement on tiles touching opponents with 2 victory points."
                        arrow
                        enterTouchDelay={0}
                        leaveTouchDelay={3000}
                      >
                        <IconButton
                          size="small"
                          className="help-button"
                          aria-label="Friendly Robber help"
                        >
                          <HelpOutlineRoundedIcon fontSize="inherit" />
                        </IconButton>
                      </Tooltip>
                    </span>
                    <strong>{friendlyRobber ? "On" : "Off"}</strong>
                  </div>
                  <Checkbox
                    className="inline-switch"
                    checked={friendlyRobber}
                    onChange={(event) =>
                      setFriendlyRobber(event.target.checked)
                    }
                  />
                </div>
              </div>

              <div className="control-group">
                <div className="control-header">
                  <span className="players-heading">
                    Players
                    <span className="players-hint">(At most one Human player)</span>
                  </span>
                  <strong>{players.length}/4</strong>
                </div>
                {hasTooManyHumans && (
                  <Alert severity="error" className="players-alert">
                    Only one Human player is allowed.
                  </Alert>
                )}
                <div className="players-list">
                  {players.map((player, index) => (
                    <div className="player-row" key={`${player}-${index}`}>
                      <div className="player-meta">
                        <span className="player-label">Player {index + 1}</span>
                        <span
                          className={`player-color-chip ${PLAYER_COLORS[index].toLowerCase()}`}
                        >
                          {PLAYER_COLORS[index]}
                        </span>
                      </div>
                      <Select
                        size="small"
                        value={player}
                        onChange={(event) =>
                          handlePlayerChange(
                            index,
                            event.target.value as PlayerArchetype
                          )
                        }
                      >
                        {PLAYER_ARCHETYPES.map((option) => (
                          <MenuItem
                            key={option.value}
                            value={option.value}
                            disabled={
                              option.value === "HUMAN" &&
                              humanCount >= 1 &&
                              player !== "HUMAN"
                            }
                          >
                            {option.label}
                          </MenuItem>
                        ))}
                      </Select>
                      <Button
                        variant="text"
                        className="remove-player-btn"
                        disabled={players.length <= 2}
                        onClick={() => handleRemovePlayer(index)}
                      >
                        Remove
                      </Button>
                    </div>
                  ))}
                </div>

                <Button
                  variant="contained"
                  className="add-player-btn"
                  disabled={players.length >= 4}
                  onClick={handleAddPlayer}
                >
                  Add Player
                </Button>
              </div>

              <Button
                variant="contained"
                color="primary"
                className="start-btn"
                disabled={hasTooManyHumans}
                onClick={handleCreateGame}
              >
                Start
              </Button>
            </div>
          </>
        ) : (
          <GridLoader
            className="loader"
            color="#ffffff"
            size={60}
          />
        )}
      </div>
    </div>
  );
}
