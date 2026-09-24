// One reviewer-supplied file, read entirely client-side. Kept in React state
// only for the duration of one edit -- never written to localStorage/
// IndexedDB, and the base64 travels to the backend in the one POST /send
// call, which itself never writes it to disk or the DB (see api.py's
// review_send and threads/store.py's attachment_name comment).
//
// Mirrors mail/graph.py's MAX_ATTACHMENT_BYTES -- Graph rejects a same-call
// fileAttachment above this size, so it's enforced here too, to fail before
// the upload rather than after.
export const MAX_ATTACHMENT_BYTES = 3 * 1024 * 1024;

export class AttachmentTooLargeError extends Error {}

// Resolves to { name, contentType, base64 } -- base64 has no "data:...;base64,"
// prefix, ready to send straight through as attachment_content_b64.
export function readFileAsAttachment(file) {
  if (file.size > MAX_ATTACHMENT_BYTES) {
    return Promise.reject(
      new AttachmentTooLargeError(
        `${file.name} is ${(file.size / (1024 * 1024)).toFixed(1)}MB -- the limit is 3MB.`
      )
    );
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error || new Error("could not read file"));
    reader.onload = () => {
      // reader.result is "data:<mime>;base64,<data>" for readAsDataURL.
      const base64 = String(reader.result).split(",", 2)[1] || "";
      resolve({ name: file.name, contentType: file.type, base64 });
    };
    reader.readAsDataURL(file);
  });
}
