import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

// Catanatron skin (default SVGs)
import catanatronBrick from "./assets/tile_brick.svg";
import catanatronDesert from "./assets/tile_desert.svg";
import catanatronWheat from "./assets/tile_wheat.svg";
import catanatronWood from "./assets/tile_wood.svg";
import catanatronOre from "./assets/tile_ore.svg";
import catanatronSheep from "./assets/tile_sheep.svg";
import catanatronMaritime from "./assets/tile_maritime.svg";

// Classic skin (PNGs with transparency)
import classicBrick from "./assets/classic/brick.png";
import classicDesert from "./assets/classic/desert.png";
import classicWheat from "./assets/classic/wheat.png";
import classicWood from "./assets/classic/wood.png";
import classicOre from "./assets/classic/ore.png";
import classicSheep from "./assets/classic/sheep.png";

// Classic resource symbols (PNG with transparency)
import classicBrickSymbol from "./assets/classic/brick_resource.png";
import classicWheatSymbol from "./assets/classic/wheat_resource.png";
import classicWoodSymbol from "./assets/classic/wood_resource.png";
import classicOreSymbol from "./assets/classic/ore_resource.png";
import classicSheepSymbol from "./assets/classic/sheep_resource.png";

// Classic trading port
import classicTradingPort from "./assets/classic/trading_port.png";

export type SkinName = "catanatron" | "classic";

export type SkinAssets = {
  BRICK: string;
  SHEEP: string;
  ORE: string;
  WOOD: string;
  WHEAT: string;
  DESERT: string;
  MARITIME: string;
  // Resource symbols for cards (only used by classic skin)
  BRICK_SYMBOL?: string;
  SHEEP_SYMBOL?: string;
  ORE_SYMBOL?: string;
  WOOD_SYMBOL?: string;
  WHEAT_SYMBOL?: string;
  // Trading port boat image (only used by classic skin)
  TRADING_PORT?: string;
};

const SKINS: Record<SkinName, SkinAssets> = {
  catanatron: {
    BRICK: catanatronBrick,
    SHEEP: catanatronSheep,
    ORE: catanatronOre,
    WOOD: catanatronWood,
    WHEAT: catanatronWheat,
    DESERT: catanatronDesert,
    MARITIME: catanatronMaritime,
  },
  classic: {
    BRICK: classicBrick,
    SHEEP: classicSheep,
    ORE: classicOre,
    WOOD: classicWood,
    WHEAT: classicWheat,
    DESERT: classicDesert,
    MARITIME: catanatronMaritime, // no classic maritime, reuse default
    BRICK_SYMBOL: classicBrickSymbol,
    SHEEP_SYMBOL: classicSheepSymbol,
    ORE_SYMBOL: classicOreSymbol,
    WOOD_SYMBOL: classicWoodSymbol,
    WHEAT_SYMBOL: classicWheatSymbol,
    TRADING_PORT: classicTradingPort,
  },
};

type SkinContextType = {
  skin: SkinName;
  assets: SkinAssets;
  setSkin: (skin: SkinName) => void;
};

const SkinContext = createContext<SkinContextType>({
  skin: "classic",
  assets: SKINS.classic,
  setSkin: () => {},
});

export function SkinProvider({ children }: { children: ReactNode }) {
  const [skin, setSkin] = useState<SkinName>("classic");

  // Sync skin class on <body> so elements outside the React tree
  // (e.g. MUI drawer portals) can be styled per-skin.
  useEffect(() => {
    const body = document.body;
    SKIN_NAMES.forEach((name) => body.classList.remove(`skin-${name}`));
    body.classList.add(`skin-${skin}`);
  }, [skin]);

  return (
    <SkinContext.Provider value={{ skin, assets: SKINS[skin], setSkin }}>
      {children}
    </SkinContext.Provider>
  );
}

export function useSkin() {
  return useContext(SkinContext);
}

export const SKIN_NAMES: SkinName[] = ["classic", "catanatron"];
