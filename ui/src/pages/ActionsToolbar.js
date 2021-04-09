import React, { useState, useRef, useEffect } from "react";
import { Button } from "@material-ui/core";

import ChevronLeftIcon from "@material-ui/icons/ChevronLeft";
import AccountBalanceIcon from "@material-ui/icons/AccountBalance";
import BuildIcon from "@material-ui/icons/Build";
import NavigateNextIcon from "@material-ui/icons/NavigateNext";
import Divider from "@material-ui/core/Divider";
import InboxIcon from "@material-ui/icons/MoveToInbox";
import List from "@material-ui/core/List";
import ListItem from "@material-ui/core/ListItem";
import ListItemIcon from "@material-ui/core/ListItemIcon";
import ListItemText from "@material-ui/core/ListItemText";
import MailIcon from "@material-ui/icons/Mail";
import MenuItem from "@material-ui/core/MenuItem";
import { makeStyles } from "@material-ui/core/styles";
import SwipeableDrawer from "@material-ui/core/SwipeableDrawer";
import ClickAwayListener from "@material-ui/core/ClickAwayListener";
import Grow from "@material-ui/core/Grow";
import Paper from "@material-ui/core/Paper";
import Popper from "@material-ui/core/Popper";
import MenuList from "@material-ui/core/MenuList";
import SimCardIcon from "@material-ui/icons/SimCard";
import "./ActionsToolbar.scss";

const useStyles = makeStyles({
  list: {
    width: 250,
  },
});

function DrawerContent({ toggleDrawer }) {
  const classes = useStyles();

  return (
    <div
      className={classes.list}
      role="presentation"
      onClick={toggleDrawer(false)}
      onKeyDown={toggleDrawer(false)}
    >
      <List>
        {["Inbox", "Starred", "Send email", "Drafts"].map((text, index) => (
          <ListItem button key={text}>
            <ListItemIcon>
              {index % 2 === 0 ? <InboxIcon /> : <MailIcon />}
            </ListItemIcon>
            <ListItemText primary={text} />
          </ListItem>
        ))}
      </List>
      <Divider />
      <List>
        {["All mail", "Trash", "Spam"].map((text, index) => (
          <ListItem button key={text}>
            <ListItemIcon>
              {index % 2 === 0 ? <InboxIcon /> : <MailIcon />}
            </ListItemIcon>
            <ListItemText primary={text} />
          </ListItem>
        ))}
      </List>
    </div>
  );
}

export default function ActionsToolbar({ zoomIn, zoomOut, onTick }) {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const iOS = process.browser && /iPad|iPhone|iPod/.test(navigator.userAgent);

  const toggleDrawer = (open) => (event) => {
    if (
      event &&
      event.type === "keydown" &&
      (event.key === "Tab" || event.key === "Shift")
    ) {
      return;
    }

    setIsDrawerOpen(open);
  };
  const playButtons = (
    <>
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
        onClick={onTick}
      >
        End
      </Button>
    </>
  );

  const botsTurn = true;

  return (
    <div className="actions-toolbar">
      {!botsTurn && playButtons}
      {botsTurn && (
        <Button className="confirm-btn" variant="contained" color="primary">
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
      <SwipeableDrawer
        anchor={"left"}
        open={isDrawerOpen}
        onClose={toggleDrawer(false)}
        onOpen={toggleDrawer(true)}
        disableBackdropTransition={!iOS}
        disableDiscovery={iOS}
      >
        <DrawerContent toggleDrawer={toggleDrawer} />
      </SwipeableDrawer>
      {/* <Button onClick={zoomIn}>Zoom In</Button>
      <Button onClick={zoomOut}>Zoom Out</Button> */}
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
