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

type SelectorOption = ResourceCard | ResourceCard[];

type ResourceSelectorProps = {
  open: boolean;
  onClose: () => void;
  onSelect: (option: SelectorOption) => void;
  options: SelectorOption[];
  mode: "discard" | "monopoly" | "yearOfPlenty";
};

const ResourceSelector = ({
  open,
  onClose,
  options,
  onSelect,
  mode,
}: ResourceSelectorProps) => {
  const resourceOrder: ResourceCard[] = [
    "WOOD",
    "BRICK",
    "SHEEP",
    "WHEAT",
    "ORE",
  ];
  const isSingleResourceOption = (
    option: SelectorOption
  ): option is ResourceCard => !Array.isArray(option);

  const sortedOptions = React.useMemo(() => {
    if (mode === "monopoly") {
      return resourceOrder;
    }
    if (mode === "discard") {
      return options
        .filter(isSingleResourceOption)
        .sort((a, b) => resourceOrder.indexOf(a) - resourceOrder.indexOf(b));
    }

    const yearOfPlentyOptions = options.filter(
      (option): option is ResourceCard[] => Array.isArray(option)
    );
    const hasDoubleOptions = yearOfPlentyOptions.some(
      (option) => option.length === 2
    );
    const filteredOptions = hasDoubleOptions
      ? yearOfPlentyOptions.filter((option) => option.length === 2)
      : yearOfPlentyOptions;

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

  const optionToResourceSpan = (option: SelectorOption) => {
    if (isSingleResourceOption(option)) {
      return getResourceSpan(option);
    }
    if (option.length === 1) {
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
        {mode === "discard"
          ? "Select Resource to Discard"
          : mode === "monopoly"
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
