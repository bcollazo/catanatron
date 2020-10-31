import React from "react";
import styled from "styled-components";

const Panel = styled.div`
  height: 300px;
  padding: 20px;

  background: white;
  border: 10px solid #ddd;
  border-bottom: none;
  border-top-right-radius: 40px;
  border-top-left-radius: 40px;
`;

const Button = styled.div`
  background: #ddd;
  border-radius: 8px;
  padding: 10px 32px;
  margin: 8px;
  font-size: 1.2em;
  width: 300px;
  text-align: center;
  cursor: pointer;
  user-select: none;
`;

export function ControlPanel({ onClickNext, onClickAutomation }) {
  return (
    <Panel>
      <Button onClick={onClickNext}>Tick</Button>
      <Button onClick={onClickAutomation}>Toggle Automation</Button>
    </Panel>
  );
}
