import React from "react";
import styled from "styled-components";

const Panel = styled.div`
  position: absolute;
  bottom: 10px;
  left: 10px;
`;

const Button = styled.div`
  background: #fff;
  border-radius: 8px;
  padding: 10px 32px;
  margin: 8px;
  font-size: 1.2em;
  width: 240px;
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
