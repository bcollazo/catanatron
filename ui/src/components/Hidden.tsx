
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
 *
 */
const HiddenJS = ({ breakpoint, children }: HiddenProps) => {
    const { size, direction } = breakpoint;
    const hidden = useMediaQuery(theme => theme.breakpoints[direction](size));
    return hidden ? null : children;
}

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

const HiddenCSS = ({ breakpoint, children }: HiddenProps) => {
    return <Box sx={{ display: getCSSBoundaries(breakpoint) }}>{children}</Box>
}

const Hidden = ({ implementation, ...props }: { implementation: "js" | "css" } & HiddenProps) => {
    const Component = implementation === "js" ? HiddenJS : HiddenCSS;
    return <Component {...props} />
}

export default Hidden;
