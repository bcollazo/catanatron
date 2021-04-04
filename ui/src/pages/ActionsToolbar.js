import React, { useState, useRef, useEffect } from "react";
import { Button } from "@material-ui/core";

import MenuItem from "@material-ui/core/MenuItem";

import AccountBalanceIcon from "@material-ui/icons/AccountBalance";
import BuildIcon from "@material-ui/icons/Build";
import NavigateNextIcon from "@material-ui/icons/NavigateNext";

import ClickAwayListener from "@material-ui/core/ClickAwayListener";
import Grow from "@material-ui/core/Grow";
import Paper from "@material-ui/core/Paper";
import Popper from "@material-ui/core/Popper";
import MenuList from "@material-ui/core/MenuList";
import SimCardIcon from "@material-ui/icons/SimCard";
import "./ActionsToolbar.scss";

export default function ActionsToolbar() {
  return (
    <div className="actions-toolbar">
      <OptionsButton
        menuListId="use-menu-list"
        icon={<SimCardIcon />}
        items={["Monopoly", "Year of Plenty", "Road Building", "Knight"]}
      >
        Use
      </OptionsButton>
      <OptionsButton
        menuListId="build-menu-list"
        icon={<BuildIcon />}
        items={["Development Card", "City", "Settlement", "Road"]}
      >
        Buy
      </OptionsButton>
      <Button
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
      >
        End
      </Button>
    </div>
  );
}

function OptionsButton({ menuListId, icon, children, items }) {
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
