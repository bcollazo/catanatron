import React from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";

import "./tailwind.output.css";
import "./App.scss";
import GamePage from "./pages/GamePage";
import HomePage from "./pages/HomePage";

function App() {
  return (
    <Router>
      <Switch>
        <Route path="/games/:gameId">
          <GamePage />
        </Route>
        <Route path="/">
          <HomePage />
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
