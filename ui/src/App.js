import React from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";

import "./App.scss";
import GameScreen from "./pages/GameScreen";
import HomePage from "./pages/HomePage";

import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";
import { blue, green } from "@material-ui/core/colors";
import { StateProvider } from "./store";

const theme = createMuiTheme({
  palette: {
    primary: {
      main: blue[900],
    },
    secondary: {
      main: green[900],
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <StateProvider>
        <Router>
          <Switch>
            <Route path="/games/:gameId">
              <GameScreen />
            </Route>
            <Route path="/" exact={true}>
              <HomePage />
            </Route>
          </Switch>
        </Router>
      </StateProvider>
    </ThemeProvider>
  );
}

export default App;
