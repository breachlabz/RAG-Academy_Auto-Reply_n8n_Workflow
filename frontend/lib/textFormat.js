// Plain-text selection transforms for the review edit box. No rich-text
// editor, no new dependency -- these operate on a <textarea>'s native
// selectionStart/selectionEnd and return the new full text plus where the
// selection should land afterward, so the caller can restore focus.
//
// Bold/Italic wrap the selection in markdown-style markers (**/*). The
// markers are what is stored and sent from here; the backend
// (mail/graph.py's reply_html) turns them into <strong>/<em> when the reply
// goes out, so keep the marker syntax in sync with the regexes there.

function applyToWholeText(text, transform) {
  const replaced = transform(text);
  return { text: replaced, selectionStart: 0, selectionEnd: replaced.length };
}

export function applyCase(text, selectionStart, selectionEnd, mode) {
  const transform = {
    upper: (s) => s.toUpperCase(),
    lower: (s) => s.toLowerCase(),
    title: (s) =>
      s.replace(/\w\S*/g, (w) => w[0].toUpperCase() + w.slice(1).toLowerCase()),
  }[mode];

  if (selectionEnd <= selectionStart) {
    return applyToWholeText(text, transform);
  }
  const before = text.slice(0, selectionStart);
  const selected = text.slice(selectionStart, selectionEnd);
  const after = text.slice(selectionEnd);
  const replaced = transform(selected);
  return {
    text: before + replaced + after,
    selectionStart,
    selectionEnd: selectionStart + replaced.length,
  };
}

export function wrapSelection(text, selectionStart, selectionEnd, marker) {
  const before = text.slice(0, selectionStart);
  const after = text.slice(selectionEnd);

  if (selectionEnd <= selectionStart) {
    // Nothing selected: insert an empty pair, cursor between them --
    // same as most markdown editors do.
    const wrapped = marker + marker;
    return {
      text: before + wrapped + after,
      selectionStart: selectionStart + marker.length,
      selectionEnd: selectionStart + marker.length,
    };
  }
  const selected = text.slice(selectionStart, selectionEnd);
  const wrapped = marker + selected + marker;
  return {
    text: before + wrapped + after,
    selectionStart: selectionStart + marker.length,
    selectionEnd: selectionStart + marker.length + selected.length,
  };
}

// Bullet ("- ") / numbered ("1. ") list toggle over every line the selection
// touches (the cursor's own line when nothing is selected). Blank lines are
// left alone and don't consume a number. If every non-blank line already has
// this kind of prefix it is removed instead; a line with the *other* kind is
// converted. The prefixes are plain text on purpose, so a list reads fine
// even in a plain-text email; mail/graph.py's reply_html renders the same
// syntax as <ul>/<ol> when the reply goes out as HTML -- keep the two in sync.
const LIST_PREFIX = /^(?:- |\d+\. )/;

export function toggleList(text, selectionStart, selectionEnd, kind) {
  const start = text.lastIndexOf("\n", selectionStart - 1) + 1;
  let end = text.indexOf("\n", selectionEnd);
  if (end === -1) end = text.length;

  const lines = text.slice(start, end).split("\n");
  const own = kind === "number" ? /^\d+\. / : /^- /;
  const content = lines.filter((l) => l.trim() !== "");
  const remove = content.length > 0 && content.every((l) => own.test(l));

  let n = 0;
  const changed = lines
    .map((line) => {
      if (line.trim() === "") return line;
      const bare = line.replace(LIST_PREFIX, "");
      if (remove) return bare;
      n += 1;
      return (kind === "number" ? `${n}. ` : "- ") + bare;
    })
    .join("\n");

  return {
    text: text.slice(0, start) + changed + text.slice(end),
    selectionStart: start,
    selectionEnd: start + changed.length,
  };
}
