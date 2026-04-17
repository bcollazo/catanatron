import { useLayoutEffect, useRef, useState } from "react";

/** Tracks element width/height; updates when layout or drawer padding changes. */
export function useResizeObserverDimensions<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [dims, setDims] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") {
      return undefined;
    }

    const measure = () => {
      const r = el.getBoundingClientRect();
      setDims({ width: r.width, height: r.height });
    };

    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return [ref, dims] as const;
}
