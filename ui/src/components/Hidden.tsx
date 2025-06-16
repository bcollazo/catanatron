
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
const getCSSBoundaries = ({ size, direction }: HiddenProps["breakpoint"]) => {
    if (size === "xs" && direction === "up") return { xs: "none" };
    if (size === "xl" && direction === "down") return "none";
    const displayObj = { [size]: "none" };
    if (direction === "up")
        return { ...displayObj, "xs": "block" }
    if (direction === "down") {
        const nextSize = SIZES[SIZES.indexOf(size) + 1]
        return { ...displayObj, [nextSize]: "block" }
    }

}

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
