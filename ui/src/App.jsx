import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { SnackbarProvider } from "notistack";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import { blue, green } from "@mui/material/colors";
import Fade from "@mui/material/Fade";

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
            <Routes>
              <Route path="/games/:gameId/states/:stateIndex">
                <GameScreen replayMode={true} />
              </Route>
              <Route path="/games/:gameId">
                <GameScreen replayMode={false} />
              </Route>
              <Route path="/" exact={true}>
                <HomePage />
              </Route>
            </Routes>
          </Router>
        </SnackbarProvider>
      </StateProvider>
    </ThemeProvider>
  );
}

export default App;
