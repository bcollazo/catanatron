import React from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";

import "./tailwind.output.css";
import "./App.scss";
import GamePage from "./pages/GamePage";
import GameScreen from "./pages/GameScreen";
import HomePage from "./pages/HomePage";

function App() {
  return (
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
  );
}

export default App;
