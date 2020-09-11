import React, { useEffect, useRef } from "react";

export default function SidePanel({ state }) {
  const actionRef = useRef(null);

  useEffect(() => {
    if (actionRef.current !== null) {
      actionRef.current.scrollTo(0, actionRef.current.scrollHeight);
    }
  }, [state.actions]);

  const actions = state.actions.map(({ color, action_type }) => {
    const colorClass = `text-white text-${color.toLowerCase()}-700`;
    return (
      <div className={colorClass}>
        {color}: {action_type}
      </div>
    );
  });

  return (
    <div className="h-full lg:w-1/2 w-1/2 bg-gray-900 p-6">
      <div ref={actionRef} className="h-64 overflow-auto">
        {actions}
      </div>
    </div>
  );
}
