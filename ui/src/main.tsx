import React from 'react';
import ReactDOM from "react-dom/client";
import './index.css';
import App from './App';
import * as serviceWorker from './serviceWorker';

const docRoot = document.getElementById("root");
if (!docRoot)
  throw new Error('FATAL: No document root element found! Please add an id="root" HTML element to your index.html');
const root = ReactDOM.createRoot(docRoot);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
