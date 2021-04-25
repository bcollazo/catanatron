import React, { createContext, useReducer } from "react";
import ACTIONS from "./actions";

const initialState = {
  gameState: null,
  // UI
  isBuildingRoad: false,
  isBuildingSettlement: false,
  isBuildingCity: false,
  isLeftDrawerOpen: false,
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
        };
      case ACTIONS.SET_IS_BUILDING_ROAD:
        return { ...state, isBuildingRoad: true };
      case ACTIONS.SET_IS_BUILDING_SETTLEMENT:
        return { ...state, isBuildingSettlement: true };
      case ACTIONS.SET_IS_BUILDING_CITY:
        return { ...state, isBuildingCity: true };
      default:
        throw new Error("Unknown Reducer Action: " + action.type);
    }
  }, initialState);

  return <Provider value={{ state, dispatch }}>{children}</Provider>;
};

export { store, StateProvider };
