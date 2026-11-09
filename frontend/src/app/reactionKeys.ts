// Key mapping and the "should this key send?" rule, kept pure so it can be reviewed apart from React.
// Rubric and rules are frozen in docs/plans/live-loop.md ("Reaction labels").

export type ReactionKey = "W" | "K" | "Space" | "D";
export type Room = "sim" | "live";

export const REACTIONS: { key: ReactionKey; hint: string; meaning: string }[] = [
  { key: "W", hint: "W", meaning: "Most of the room, sustained" },
  { key: "K", hint: "K", meaning: "A clear laugh from a good part of the room, or applause" },
  { key: "Space", hint: "Space", meaning: "A few chuckles, or a groan" },
  { key: "D", hint: "D", meaning: "Nothing, it fell flat" },
];

const KEY_BY_CODE: Record<string, ReactionKey> = { KeyW: "W", KeyK: "K", Space: "Space", KeyD: "D" };

const TYPING_TAGS = new Set(["INPUT", "SELECT", "TEXTAREA"]);

export type KeyInput = {
  code: string;
  repeat: boolean;
  targetTag: string | null;
  targetEditable: boolean;
};

export type ShowGate = { room: Room | null; windowOpen: boolean };

export function isTyping(tag: string | null, editable: boolean): boolean {
  return editable || (tag !== null && TYPING_TAGS.has(tag.toUpperCase()));
}

// Returns the key to send, or null when the press must be ignored (and left to the browser).
export function reactionKeyFor(input: KeyInput, gate: ShowGate): ReactionKey | null {
  const key = KEY_BY_CODE[input.code];
  if (!key) return null;
  if (input.repeat) return null;
  if (isTyping(input.targetTag, input.targetEditable)) return null;
  if (gate.room !== "live" || !gate.windowOpen) return null;
  return key;
}
