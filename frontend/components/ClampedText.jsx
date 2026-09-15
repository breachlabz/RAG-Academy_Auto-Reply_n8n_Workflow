"use client";

import { useEffect, useRef, useState } from "react";

// A 5-line preview with a "Show more" toggle, only shown when the text
// actually overflows the clamp. Overflow is measured once, right after the
// clamped layout is in the DOM -- not on every expand/collapse, since once
// expanded clientHeight grows to match scrollHeight and the check would
// always read "no overflow" and hide the button that's currently in use.
export default function ClampedText({ text }) {
  const ref = useRef(null);
  const [expanded, setExpanded] = useState(false);
  const [overflows, setOverflows] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setOverflows(el.scrollHeight > el.clientHeight + 2);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  return (
    <div>
      <div ref={ref} className={"clamp-text" + (expanded ? "" : " clamped")}>
        {text}
      </div>
      {overflows && (
        <button
          type="button"
          className="link expand-toggle"
          onClick={() => setExpanded((e) => !e)}
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
}
