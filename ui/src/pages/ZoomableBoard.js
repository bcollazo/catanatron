import React from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

export default function ZoomableBoard() {
  return (
    <TransformWrapper
      options={{
        limitToBounds: false,
      }}
    >
      {({
        zoomIn,
        zoomOut,
        resetTransform,
        positionX,
        positionY,
        scale,
        previousScale,
      }) => (
        <React.Fragment>
          <div className="board-container">
            {
              <TransformComponent>
                <div className="example-text">
                  <div className="block"></div>
                  <h1>Lorem ipsum</h1>
                  <p>
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed
                    do eiusmod tempor incididunt ut labore et dolore magna
                    aliqua. Ut enim ad minim veniam, quis nostrud exercitation
                    ullamco laboris nisi ut aliquip ex ea commodo consequat.
                    Duis aute irure dolor in reprehenderit in voluptate velit
                    esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
                    occaecat cupidatat non proident, sunt in culpa qui officia
                    deserunt mollit anim id est laborum.
                  </p>
                  <h1>SVG</h1>
                  {/* <Robber
                        center={[positionX + center[0], positionY + center[1]]}
                        w={w}
                        h={h}
                        size={size}
                        coordinate={state.robber_coordinate}
                      /> */}
                </div>
              </TransformComponent>
            }
          </div>
          <div className="tools">
            <div className="info">
              <h3>State</h3>
              <h5>
                <span className="badge badge-secondary">
                  Position x : {positionX}px
                </span>
                <span className="badge badge-secondary">
                  Position y : {positionY}px
                </span>
                <span className="badge badge-secondary">Scale : {scale}</span>
                <span className="badge badge-secondary">
                  Previous scale : {previousScale}
                </span>
              </h5>
            </div>
            <button
              className="btn-gradient cyan small"
              onClick={zoomIn}
              data-testid="zoom-in-button"
            >
              Zoom In
            </button>
            <button
              className="btn-gradient blue small"
              onClick={zoomOut}
              data-testid="zoom-out-button"
            >
              Zoom Out
            </button>
            <button
              className="btn-gradient purple small"
              onClick={resetTransform}
              data-testid="reset-button"
            >
              Reset
            </button>
          </div>
        </React.Fragment>
      )}
    </TransformWrapper>
  );
}
