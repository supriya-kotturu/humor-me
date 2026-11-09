# frontend/

Next.js UI for the live show: a single operator screen.

- **Waveform avatar**: frequency bars driven by the TTS audio.
- **Before a show**: form for venue, energy, age, operator name and a Simulated/Live toggle. A Live show asks for confirmation first, because key presses are recorded as the room's reaction.
- **During a show**: the current line, a "React now" indicator that follows `reaction_window` messages, and four buttons (W, K, Space, D) with the rubric meaning from the plan's "Reaction labels".
- **Never shown**: the director's mode (the backend doesn't send it).
- **Planned**: end-of-episode rating and pairwise "which bit was better?" votes.

## Keys

Rules live in `src/app/reactionKeys.ts` (pure, one function). Keys use `event.code` (`KeyW`, `KeyK`, `Space`, `KeyD`), ignore repeats and typing in inputs, and only act in a Live show while a window is open. Space is `preventDefault`-ed so the page doesn't scroll.

- **Live:** a press sends `{type: "reaction", bit, joke, key}`; the button stays lit until the window closes and a later press moves the highlight.
- **Simulated:** buttons send nothing; the one matching `sim_reaction` lights briefly.
- **No double fire:** buttons take `tabIndex={-1}` and ignore mouse-down focus, the handler blurs whatever has focus, and a keyboard-generated click (`detail === 0`) is ignored.

## Setup and run

```sh
cd frontend
npm install
npm run dev        # http://localhost:3000
npm run build      # production build check
```

The backend must be running on `http://localhost:8000` for the websocket (CORS allows
`http://localhost:3000`).

## Stack

Next.js 15.5 (App Router, TypeScript, `src/` dir, ESLint, no Tailwind). Pinned to a patched
15.5.x release (CVE-2025-66478); upgrade within 15.5 with `npm install next@15.5 eslint-config-next@15.5`.

## Structure

| Path | Role |
|---|---|
| `src/app/page.tsx` | Setup form, live-show confirmation, reaction buttons, global key listener |
| `src/app/useShow.ts` | Websocket client and show state (window, highlight, notice, end status) |
| `src/app/reactionKeys.ts` | Key mapping and the "should this key send?" rule |
| `src/app/Waveform.tsx` | Canvas bars animated from audio amplitude |
