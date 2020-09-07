import React from "react";
import styled from "styled-components";

const Panel = styled.div`
  position: fixed;
  bottom: 0;
  width: 100%;
  height: 300px;
  padding: 20px;

  background: white;
  border: 10px solid #ddd;
  border-bottom: none;
  border-top-right-radius: 40px;
  border-top-left-radius: 40px;
`;

const CardsContainer = styled.div`
  display: flex;
  justify-content: center;
`;

const Card = styled.div`
  width: 100px;
  height: 160px;
  background: white;
  border-radius: 8px;
  border: 1px solid #ddd;
  display: flex;
  margin: 0px 8px;
`;

const InnerCard = styled.div`
  background: #ddd;
  margin: 8px;
  border-radius: 8px;
  width: 100%;
`;

const ActionContainer = styled.div`
  display: flex;
  justify-content: center;
  margin-top: 20px;
`;

const Button = styled.div`
  background: #ddd;
  border-radius: 8px;
  padding: 10px 32px;
  margin: 8px;
  font-size: 1.2em;
  width: 200px;
  text-align: center;
  cursor: pointer;
`;

export function ControlPanel() {
  return (
    <Panel>
      <CardsContainer>
        <Card>
          <InnerCard />
        </Card>
        <Card>
          <InnerCard />
        </Card>
        <Card>
          <InnerCard />
        </Card>
        <Card>
          <InnerCard />
        </Card>
      </CardsContainer>
      <ActionContainer>
        <Button>Trade</Button>
        <Button>Build</Button>
        <Button>End Turn</Button>
      </ActionContainer>
    </Panel>
  );
}
