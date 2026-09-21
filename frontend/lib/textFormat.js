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
