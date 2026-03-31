import React from "react";
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Typography,
} from "@mui/material";
import type { ResourceCard } from "../utils/api.types";
import "./DiscardPlannerDialog.scss";

type DiscardResourceCounts = Partial<Record<ResourceCard, number>>;

type DiscardPlannerDialogProps = {
  open: boolean;
  onClose: () => void;
  onConfirm: (resources: ResourceCard[]) => void;
  remainingDiscardCount: number;
  discardResourceCounts: DiscardResourceCounts;
  submitting?: boolean;
};

const RESOURCE_ORDER: ResourceCard[] = [
  "WOOD",
  "BRICK",
  "SHEEP",
  "WHEAT",
  "ORE",
];

export default function DiscardPlannerDialog({
  open,
  onClose,
  onConfirm,
  remainingDiscardCount,
  discardResourceCounts,
  submitting = false,
}: DiscardPlannerDialogProps) {
  const [discardSelections, setDiscardSelections] = React.useState<ResourceCard[]>(
    [],
  );

  React.useEffect(() => {
    if (!open) {
      setDiscardSelections([]);
    }
  }, [open]);

  const selectedCount = discardSelections.length;
  const selectedResourceCount = (resource: ResourceCard) =>
    discardSelections.filter((selection) => selection === resource).length;
  const canSelectResource = (resource: ResourceCard) => {
    const availableCount = discardResourceCounts[resource] ?? 0;
    return (
      selectedCount < remainingDiscardCount &&
      selectedResourceCount(resource) < availableCount
    );
  };
  const addDiscardSelection = (resource: ResourceCard) => {
    if (!canSelectResource(resource) || submitting) {
      return;
    }
    setDiscardSelections((current) => [...current, resource]);
  };
  const removeDiscardSelection = (resource: ResourceCard) => {
    setDiscardSelections((current) => {
      const index = current.lastIndexOf(resource);
      if (index === -1) {
        return current;
      }
      return [...current.slice(0, index), ...current.slice(index + 1)];
    });
  };
  const groupedDiscardSelections = RESOURCE_ORDER.map((resource) => ({
    resource,
    count: selectedResourceCount(resource),
  })).filter(({ count }) => count > 0);

  return (
    <Dialog
      open={open}
      onClose={submitting ? undefined : onClose}
      className="discard-planner"
      maxWidth="xs"
      fullWidth
    >
      <DialogTitle>
        Select Your Discards
        <Typography variant="body2" className="discard-note">
          {selectedCount === 0
            ? `Choose ${remainingDiscardCount} card${remainingDiscardCount === 1 ? "" : "s"}.`
            : `${selectedCount} of ${remainingDiscardCount} selected.`}
        </Typography>
      </DialogTitle>
      <DialogContent>
        <div className="discard-summary">
          {groupedDiscardSelections.length > 0 && (
            <div className="selected-discard-list">
              {groupedDiscardSelections.map(({ resource, count }) => (
                <Button
                  key={resource}
                  variant="outlined"
                  className="selected-discard-chip"
                  onClick={() => removeDiscardSelection(resource)}
                  disabled={submitting}
                >
                  {resource} x{count}
                </Button>
              ))}
            </div>
          )}
        </div>
        <div className="resource-grid">
          {RESOURCE_ORDER.filter((resource) => discardResourceCounts[resource] != null).map(
            (resource) => (
              <Button
                key={resource}
                variant="contained"
                className="resource-button"
                onClick={() => addDiscardSelection(resource)}
                disabled={submitting || !canSelectResource(resource)}
              >
                <Typography variant="body2">
                  <span className={`resource-name ${resource.toLowerCase()}`}>
                    {resource}
                  </span>
                  <br />
                  <span className="resource-meta">
                    {selectedResourceCount(resource)}/
                    {discardResourceCounts[resource]} selected
                  </span>
                </Typography>
              </Button>
            ),
          )}
        </div>
      </DialogContent>
      <DialogActions>
        <Button
          onClick={() => setDiscardSelections([])}
          className="clear-button"
          disabled={submitting || selectedCount === 0}
        >
          Clear
        </Button>
        <Button onClick={onClose} className="cancel-button" disabled={submitting}>
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={() => onConfirm(discardSelections)}
          disabled={submitting || selectedCount !== remainingDiscardCount}
        >
          Confirm
        </Button>
      </DialogActions>
    </Dialog>
  );
}
