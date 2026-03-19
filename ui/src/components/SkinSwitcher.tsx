import Button from "@mui/material/Button";
import ButtonGroup from "@mui/material/ButtonGroup";
import { useSkin, SKIN_NAMES, type SkinName } from "../SkinContext";

import "./SkinSwitcher.scss";

export default function SkinSwitcher() {
  const { skin, setSkin } = useSkin();

  return (
    <div className="skin-switcher">
      <ButtonGroup size="small" variant="outlined">
        {SKIN_NAMES.map((name: SkinName) => (
          <Button
            key={name}
            onClick={() => setSkin(name)}
            variant={skin === name ? "contained" : "outlined"}
          >
            {name}
          </Button>
        ))}
      </ButtonGroup>
    </div>
  );
}
