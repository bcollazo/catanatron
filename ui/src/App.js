import React from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";

import "./tailwind.output.css";
import "./App.scss";
import GameScreen from "./pages/GameScreen";
import HomePage from "./pages/HomePage";

import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";
import { blue, green } from "@material-ui/core/colors";

const theme = createMuiTheme({
  palette: {
    primary: {
      main: blue[600],
    },
    secondary: {
      main: green[600],
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <Switch>
          <Route path="/games/:gameId">
            <GameScreen />
          </Route>
          <Route path="/">
            <HomePage />
          </Route>
        </Switch>
      </Router>
    </ThemeProvider>
  );
}

export default App;
