import React, { createContext, useReducer } from "react";
import ACTIONS from "./actions";

const initialState = {
  gameState: null,
  // UI
  isBuildingRoad: false,
  isBuildingSettlement: false,
  isBuildingCity: false,
  isLeftDrawerOpen: false,
  isPlayingYearOfPlenty: false,
  isRoadBuilding: false,
  freeRoadsAvailable: 0,
  isMovingRobber: false,
};
const store = createContext(initialState);
const { Provider } = store;

const StateProvider = ({ children }) => {
  const [state, dispatch] = useReducer((state, action) => {
    switch (action.type) {
      case ACTIONS.SET_LEFT_DRAWER_OPENED:
        return { ...state, isLeftDrawerOpen: action.data };
      case ACTIONS.SET_GAME_STATE:
        return {
          ...state,
          gameState: action.data,
          // Lazy way of turning these off
          isBuildingRoad: false,
          isBuildingSettlement: false,
          isBuildingCity: false,
          isRoadBuilding: state.isRoadBuilding && state.freeRoadsAvailable > 0,
          freeRoadsAvailable: state.isRoadBuilding ? state.freeRoadsAvailable - 1 : 0,
          isPlayingYearOfPlenty: false,
          isMovingRobber: false,
        };
      case ACTIONS.SET_IS_BUILDING_ROAD:
        return { ...state, isBuildingRoad: true };
      case ACTIONS.SET_IS_BUILDING_SETTLEMENT:
        return { ...state, isBuildingSettlement: true };
      case ACTIONS.SET_IS_BUILDING_CITY:
        return { ...state, isBuildingCity: true };
      case ACTIONS.SET_IS_PLAYING_YEAR_OF_PLENTY:
        return { ...state, isPlayingYearOfPlenty: true }
      case ACTIONS.PLAY_YEAR_OF_PLENTY:
        return { ...state, isPlayingYearOfPlenty: false }
      case ACTIONS.PLAY_ROAD_BUILDING:
        return {
          ...state, 
          isRoadBuilding: true, 
          freeRoadsAvailable: 2 
        };
      case ACTIONS.PLAY_KNIGHT_CARD:
        return { ...state }
      case ACTIONS.SET_IS_MOVING_ROBBER:
        return { ...state, isMovingRobber: true }
      default:
        throw new Error("Unknown Reducer Action: " + action.type);
    }
  }, initialState);

  return <Provider value={{ state, dispatch }}>{children}</Provider>;
};

export { store, StateProvider };
