import React from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import { SnackbarProvider } from "notistack";
import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";
import { blue, green } from "@material-ui/core/colors";
import Fade from "@material-ui/core/Fade";

import GameScreen from "./pages/GameScreen";
import HomePage from "./pages/HomePage";
import { StateProvider } from "./store";

import "./App.scss";

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
        <SnackbarProvider
          classes={{ containerRoot: ["snackbar-container"] }}
          maxSnack={1}
          TransitionComponent={Fade}
        >
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
        </SnackbarProvider>
      </StateProvider>
    </ThemeProvider>
  );
}

export default App;
