import React from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
} from "@mui/material";
import "./ResourceSelector.scss";
import type { ResourceCard } from "../utils/api.types";

type ResourceSelectorProps = {
  open: boolean;
  onClose: () => void;
  onSelect: (option: ResourceCard | ResourceCard[]) => void;
  options: ResourceCard[][]; // this is only used when mode == "yearOfPlenty"
  mode: "monopoly" | "yearOfPlenty";
};

const ResourceSelector = ({
  open,
  onClose,
  options,
  onSelect,
  mode,
}: ResourceSelectorProps) => {
  const sortedOptions = React.useMemo(() => {
    const resourceOrder: ResourceCard[] = [
      "WOOD",
      "BRICK",
      "SHEEP",
      "WHEAT",
      "ORE",
    ];

    if (mode === "monopoly") {
      return resourceOrder;
    }

    const hasDoubleOptions = options.some((option) => option.length === 2);
    const filteredOptions = hasDoubleOptions
      ? options.filter((option) => option.length === 2)
      : options;

    return filteredOptions.sort((a: ResourceCard[], b: ResourceCard[]) => {
      const aFirstResource = a[0];
      const bFirstResource = b[0];
      if (aFirstResource !== bFirstResource) {
        return (
          resourceOrder.indexOf(aFirstResource) -
          resourceOrder.indexOf(bFirstResource)
        );
      }
      if (a.length === 2 && b.length === 2) {
        return resourceOrder.indexOf(a[1]) - resourceOrder.indexOf(b[1]);
      }
      return a.length === 1 ? 1 : -1;
    });
  }, [options, mode]);

  const getResourceSpan = (resource: ResourceCard) => (
    <span className={`resource-name ${resource.toLowerCase()}`}>
      {resource}
    </span>
  );

  const isMonopolyOption = (
    option: ResourceCard | ResourceCard[]
  ): option is ResourceCard => mode === "monopoly" && !!option;
  const optionToResourceSpan = (option: ResourceCard | ResourceCard[]) => {
    if (isMonopolyOption(option)) {
      return getResourceSpan(option);
    } else if (option.length === 1) {
      return (
        <>
          {getResourceSpan(option[0])}
          <span className="plus">x1</span>
        </>
      );
    } else {
      return (
        <>
          {getResourceSpan(option[0])}
          <span className="plus">+</span>
          {getResourceSpan(option[1])}
        </>
      );
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      className="resource-selector"
      maxWidth="xs"
      fullWidth
    >
      <DialogTitle>
        {mode === "monopoly"
          ? "Select Resource to Monopolize"
          : "Select Resources for Year of Plenty"}
      </DialogTitle>
      <DialogContent>
        <div className="resource-grid">
          {sortedOptions.map((option, index) => (
            <Button
              key={index}
              variant="contained"
              className="resource-button"
              onClick={() => onSelect(option)}
            >
              <Typography variant="body2">
                {optionToResourceSpan(option)}
              </Typography>
            </Button>
          ))}
        </div>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} className="cancel-button">
          Cancel
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ResourceSelector;
