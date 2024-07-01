import React from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import { SnackbarProvider } from "notistack";
import { createTheme, ThemeProvider } from "@material-ui/core/styles";
import { blue, green } from "@material-ui/core/colors";
import Fade from "@material-ui/core/Fade";

import GameScreen from "./pages/GameScreen";
import HomePage from "./pages/HomePage";
import { StateProvider } from "./store";

import "./App.scss";

const theme = createTheme({
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
          autoHideDuration={1000}
          TransitionComponent={Fade}
          TransitionProps={{ timeout: 100 }}
        >
          <Router>
            <Switch>
              <Route path="/games/:gameId/states/:stateIndex">
                <GameScreen replayMode={true} />
              </Route>
              <Route path="/games/:gameId">
                <GameScreen replayMode={false} />
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
