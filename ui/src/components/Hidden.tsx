
import useMediaQuery from '@mui/material/useMediaQuery';
import { Box } from '@mui/system';

const SIZES = ["xs", "sm", "md", "lg", "xl"] as const;

interface HiddenProps {
    breakpoint: {
        size: typeof SIZES[number];
        direction: "up" | "down"
    }
    children: React.ReactNode
}
/**
 * JS implementation of the deprecated Hidden component in MUI
 * See https://v5-0-6.mui.com/guides/migration-v4/#hidden
 */
const HiddenJS = ({ breakpoint, children }: HiddenProps) => {
    const { size, direction } = breakpoint;
    const hidden = useMediaQuery(theme => theme.breakpoints[direction](size));
    return hidden ? null : children;
}

/**
 * Returns the CSS display property based on the breakpoint size and direction.
 * See https://mui.com/system/getting-started/usage/#responsive-values
 */
export const getCSSBoundaries = ({
  size,
  direction,
}: HiddenProps["breakpoint"]) => {
  if (size === "xs" && direction === "up") return { xs: "none" };
  if (size === "xl" && direction === "down") return "none";

  if (direction === "up") {
    return { xs: "block", [size]: "none" };
  }

  const visibleFrom = SIZES[SIZES.indexOf(size) + 1];
  return SIZES.reduce<Record<(typeof SIZES)[number], "none" | "block">>(
    (display, breakpoint) => {
      display[breakpoint] =
        visibleFrom && SIZES.indexOf(breakpoint) >= SIZES.indexOf(visibleFrom)
          ? "block"
          : "none";
      return display;
    },
    { xs: "none", sm: "none", md: "none", lg: "none", xl: "none" }
  );
};

/**
 * CSS implementation of the deprecated Hidden component in MUI
 * See https://v5-0-6.mui.com/guides/migration-v4/#hidden
 */
const HiddenCSS = ({ breakpoint, children }: HiddenProps) => {
    return <Box sx={{ display: getCSSBoundaries(breakpoint) }}>{children}</Box>
}

/**
 * Compatibility for the deprecated Material UI Hidden component.
 * See https://v5-0-6.mui.com/guides/migration-v4/#hidden
 *
 */
const Hidden = ({
  implementation = "css",
  ...props
}: { implementation?: "js" | "css" } & HiddenProps) => {
  const Component = implementation === "js" ? HiddenJS : HiddenCSS;
  return <Component {...props} />;
};

export default Hidden;
