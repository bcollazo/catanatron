import React, { useState, useRef, useEffect } from "react";
import { Button } from "@material-ui/core";

// import ChevronLeftIcon from "@material-ui/icons/ChevronLeft";
import AccountBalanceIcon from "@material-ui/icons/AccountBalance";
import BuildIcon from "@material-ui/icons/Build";
import NavigateNextIcon from "@material-ui/icons/NavigateNext";
import MenuItem from "@material-ui/core/MenuItem";
import ClickAwayListener from "@material-ui/core/ClickAwayListener";
import Grow from "@material-ui/core/Grow";
import Paper from "@material-ui/core/Paper";
import Popper from "@material-ui/core/Popper";
import MenuList from "@material-ui/core/MenuList";
import SimCardIcon from "@material-ui/icons/SimCard";
import "./ActionsToolbar.scss";

export default function ActionsToolbar({
  zoomIn,
  zoomOut,
  onTick,
  disabled,
  botsTurn,
  prompt,
}) {
  const isRoll = prompt === "ROLL";
  const playButtons = (
    <>
      <OptionsButton
        disabled={false}
        menuListId="use-menu-list"
        icon={<SimCardIcon />}
        items={["Monopoly", "Year of Plenty", "Road Building", "Knight"]}
      >
        Use
      </OptionsButton>
      <OptionsButton
        disabled={isRoll}
        menuListId="build-menu-list"
        icon={<BuildIcon />}
        items={["Development Card", "City", "Settlement", "Road"]}
      >
        Buy
      </OptionsButton>
      <Button
        disabled={isRoll}
        variant="contained"
        color="secondary"
        startIcon={<AccountBalanceIcon />}
      >
        Trade
      </Button>
      <Button
        variant="contained"
        color="primary"
        startIcon={<NavigateNextIcon />}
        onClick={onTick}
      >
        {isRoll ? "ROLL" : "END"}
      </Button>
    </>
  );

  return (
    <div className="actions-toolbar">
      {!botsTurn && playButtons}
      {botsTurn && (
        <Button
          disabled={disabled}
          className="confirm-btn"
          variant="contained"
          color="primary"
          onClick={onTick}
        >
          Ok
        </Button>
      )}
      {/* <Button
        className="open-drawer-btn"
        startIcon={<ChevronLeftIcon />}
        onClick={toggleDrawer(true)}
      >
        Open Info
      </Button> */}

      {/* <Button onClick={zoomIn}>Zoom In</Button>
      <Button onClick={zoomOut}>Zoom Out</Button> */}
    </div>
  );
}

function OptionsButton({ menuListId, icon, children, items, disabled }) {
  const [open, setOpen] = useState(false);
  const anchorRef = useRef(null);

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };
  const handleClose = (event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target)) {
      return;
    }

    setOpen(false);
  };
  function handleListKeyDown(event) {
    if (event.key === "Tab") {
      event.preventDefault();
      setOpen(false);
    }
  }
  // return focus to the button when we transitioned from !open -> open
  const prevOpen = useRef(open);
  useEffect(() => {
    if (prevOpen.current === true && open === false) {
      anchorRef.current.focus();
    }

    prevOpen.current = open;
  }, [open]);

  return (
    <React.Fragment>
      <Button
        disabled={disabled}
        ref={anchorRef}
        aria-controls={open ? menuListId : undefined}
        aria-haspopup="true"
        variant="contained"
        color="secondary"
        startIcon={icon}
        onClick={handleToggle}
      >
        {children}
      </Button>
      <Popper
        className="action-popover"
        open={open}
        anchorEl={anchorRef.current}
        role={undefined}
        transition
        disablePortal
      >
        {({ TransitionProps, placement }) => (
          <Grow
            {...TransitionProps}
            style={{
              transformOrigin:
                placement === "bottom" ? "center top" : "center bottom",
            }}
          >
            <Paper>
              <ClickAwayListener onClickAway={handleClose}>
                <MenuList
                  autoFocusItem={open}
                  id={menuListId}
                  onKeyDown={handleListKeyDown}
                >
                  {items.map((item) => (
                    <MenuItem key={item} onClick={handleClose}>
                      {item}
                    </MenuItem>
                  ))}
                </MenuList>
              </ClickAwayListener>
            </Paper>
          </Grow>
        )}
      </Popper>
    </React.Fragment>
  );
}
