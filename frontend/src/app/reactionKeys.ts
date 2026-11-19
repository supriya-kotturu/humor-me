// Key mapping, kept pure so it can be reviewed apart from React.
// Rubric and rules are frozen in docs/plans/live-loop.md ("Reaction labels").
// Whether a key may send (live room, open window) is decided once, in useShow's `react`.

export type ReactionKey = "W" | "K" | "Space" | "D";
export type Room = "sim" | "live";

export const REACTIONS: { key: ReactionKey; hint: string; meaning: string }[] = [
  { key: "W", hint: "W", meaning: "Most of the room, sustained" },
  { key: "K", hint: "K", meaning: "A clear laugh from a good part of the room, or applause" },
  { key: "Space", hint: "Space", meaning: "A few chuckles, or a groan" },
  { key: "D", hint: "D", meaning: "Nothing, it fell flat" },
];

// Physical key positions: assumes a QWERTY operator keyboard.
const KEY_BY_CODE: Record<string, ReactionKey> = { KeyW: "W", KeyK: "K", Space: "Space", KeyD: "D" };

const TYPING_TAGS = new Set(["INPUT", "SELECT", "TEXTAREA"]);

export type KeyInput = {
  code: string;
  repeat: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
  targetTag: string | null;
  targetEditable: boolean;
};

// The fields of a keydown event that the key rules read, in one place for every handler.
export function keyInput(event: KeyboardEvent): KeyInput {
  const target = event.target instanceof HTMLElement ? event.target : null;
  return {
    code: event.code,
    repeat: event.repeat,
    ctrlKey: event.ctrlKey,
    metaKey: event.metaKey,
    altKey: event.altKey,
    shiftKey: event.shiftKey,
    targetTag: target?.tagName ?? null,
    targetEditable: target?.isContentEditable ?? false,
  };
}

export function isTyping(tag: string | null, editable: boolean): boolean {
  return editable || (tag !== null && TYPING_TAGS.has(tag.toUpperCase()));
}

// Returns the reaction key, or null when the press is not a reaction (and is left to the browser).
export function reactionKeyFor(input: KeyInput): ReactionKey | null {
  const key = KEY_BY_CODE[input.code];
  if (!key) return null;
  if (input.repeat) return null;
  // Browser shortcuts (Ctrl+K, Ctrl+D, Cmd+W...) must neither count as a laugh nor be swallowed.
  if (input.ctrlKey || input.metaKey || input.altKey || input.shiftKey) return null;
  if (isTyping(input.targetTag, input.targetEditable)) return null;
  return key;
}

const RATING_BY_CODE: Record<string, number> = {
  Digit1: 1, Digit2: 2, Digit3: 3, Digit4: 4, Digit5: 5,
  Numpad1: 1, Numpad2: 2, Numpad3: 3, Numpad4: 4, Numpad5: 5,
};

// The end-of-show rating, 1 to 5. Same guards as reactions: no repeats, no shortcuts, never while typing.
export function ratingFor(input: KeyInput): number | null {
  const value = RATING_BY_CODE[input.code];
  if (!value || input.repeat) return null;
  if (input.ctrlKey || input.metaKey || input.altKey || input.shiftKey) return null;
  if (isTyping(input.targetTag, input.targetEditable)) return null;
  return value;
}

// P pauses or resumes the show. Same guards as reactions.
export function isPauseKey(input: KeyInput): boolean {
  if (input.code !== "KeyP" || input.repeat) return false;
  if (input.ctrlKey || input.metaKey || input.altKey || input.shiftKey) return false;
  return !isTyping(input.targetTag, input.targetEditable);
}
