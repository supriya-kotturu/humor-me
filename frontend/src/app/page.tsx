"use client";

import { useEffect, useState } from "react";
import { useShow, type ShowSetup, type ShowState } from "./useShow";
import { REACTIONS, isPauseKey, keyInput, ratingFor, reactionKeyFor, type Room } from "./reactionKeys";
import Waveform from "./Waveform";
import styles from "./page.module.css";

const VENUES = ["bar", "company_event", "showcase"];
const ENERGIES = ["low", "normal", "high"];
const AGES = ["teens", "young_adults", "mixed"];

const STATUS: Record<ShowState, string> = {
  idle: "Ready when you are.",
  connecting: "Warming up the voice. The first show can take a little while.",
  speaking: "",
  done: "That's the set.",
  error: "Can't reach the backend on localhost:8000. Is it running?",
};

const label = (value: string) => value.replace(/_/g, " ");

export default function Home() {
  const {
    state, line, analyser, start, room, reactionWindow, pressed, flash, notice, endStatus, rating, rate, react,
    paused, togglePause,
  } = useShow();
  const [setup, setSetup] = useState<ShowSetup>({
    venue: "bar",
    energy: "normal",
    age: "mixed",
    operator: "",
    room: "sim",
    consent: false,
  });
  const [confirming, setConfirming] = useState(false);
  const busy = state === "connecting" || state === "speaking";
  // Display only; whether a press sends is decided in `react`. A simulated room opens windows too.
  const canReact = room === "live" && reactionWindow !== null;

  // Keys are read only during a show. No control can hold focus then (the form and dialog are gone and the
  // reaction buttons never take it), so a key press can't also click a button.
  useEffect(() => {
    if (!busy) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const input = keyInput(event);
      if (isPauseKey(input)) {
        event.preventDefault();
        togglePause();
        return;
      }
      const key = reactionKeyFor(input);
      if (!key) return;
      event.preventDefault(); // Space would scroll the page
      react(key);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, react, togglePause]);

  // The prompt owns the digit keys only while it is up; the form's fields are not on screen then.
  const ratingOpen = rating === "ask";
  useEffect(() => {
    if (!ratingOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const value = ratingFor(keyInput(event));
      if (value === null) return;
      event.preventDefault();
      rate(value);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [ratingOpen, rate]);

  const launch = () => {
    setConfirming(false);
    start(setup);
    set({ consent: false }); // consent is for one show; the next one asks again
  };

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!setup.operator.trim()) return;
    if (setup.room === "live") setConfirming(true);
    else launch();
  };

  const set = (patch: Partial<ShowSetup>) => setSetup((s) => ({ ...s, ...patch }));

  const select = (name: "venue" | "energy" | "age", text: string, options: string[]) => (
    <label className={styles.field}>
      <span>{text}</span>
      <select value={setup[name]} onChange={(e) => set({ [name]: e.target.value })}>
        {options.map((o) => (
          <option key={o} value={o}>
            {label(o)}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <main className={styles.stage}>
      <div className={styles.wave} aria-hidden="true">
        <Waveform analyser={analyser} />
      </div>
      <p className={styles.line} aria-live="polite">
        {state === "speaking" ? line : state === "done" && endStatus === "aborted" ? "The show ended early." : STATUS[state]}
      </p>
      {notice && (
        <p className={styles.notice} role="alert">
          {notice}
        </p>
      )}

      {busy ? (
        <section className={styles.panel} aria-label="Reactions">
          <div className={styles.controls}>
            <p className={canReact ? styles.windowOpen : styles.windowIdle} role="status">
              {paused ? "Paused" : room === "sim" ? "Simulated room" : canReact ? "React now" : "Waiting for the line to finish"}
            </p>
            {/* Never takes focus, like the reaction buttons, so Space can't press it. */}
            <button
              type="button"
              tabIndex={-1}
              className={styles.start}
              aria-pressed={paused}
              onMouseDown={(e) => e.preventDefault()}
              onClick={togglePause}
            >
              {paused ? "Resume (P)" : "Pause (P)"}
            </button>
          </div>
          <div className={styles.buttons}>
            {REACTIONS.map(({ key, hint, meaning }) => {
              const lit = room === "live" ? pressed === key : flash === key;
              return (
                <button
                  key={key}
                  type="button"
                  tabIndex={-1}
                  className={`${styles.reaction} ${lit ? styles.lit : ""}`}
                  aria-pressed={lit}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => react(key)}
                >
                  <span className={styles.hint}>{hint}</span>
                  <span className={styles.meaning}>{meaning}</span>
                </button>
              );
            })}
          </div>
        </section>
      ) : rating === "ask" || rating === "saving" ? (
        <section className={styles.panel} aria-label="Rating">
          <p className={styles.confirm}>How did the room rate the set?</p>
          <div className={styles.row}>
            {[1, 2, 3, 4, 5].map((n) => (
              <button
                key={n}
                type="button"
                className={styles.rate}
                disabled={rating === "saving"}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => rate(n)}
              >
                {n}
              </button>
            ))}
            <button type="button" className={styles.start} disabled={rating === "saving"} onClick={() => rate(null)}>
              Skip
            </button>
          </div>
        </section>
      ) : confirming ? (
        <section className={styles.panel} role="alertdialog" aria-label="Confirm live show">
          <p className={styles.confirm}>
            This is a live show. Every key you press (W, K, Space, D) is recorded as the room&apos;s reaction and becomes
            training data. Press only what the room actually did.
          </p>
          <div className={styles.row}>
            <button type="button" className={styles.start} onClick={launch} autoFocus>
              Confirm and start
            </button>
            <button type="button" className={styles.start} onClick={() => setConfirming(false)}>
              Back
            </button>
          </div>
        </section>
      ) : (
        <form className={styles.form} onSubmit={onSubmit}>
          {select("venue", "Venue", VENUES)}
          {select("energy", "Energy", ENERGIES)}
          {select("age", "Age", AGES)}
          <label className={styles.field}>
            <span>Operator</span>
            <input value={setup.operator} onChange={(e) => set({ operator: e.target.value })} required />
          </label>
          <fieldset className={styles.toggle}>
            <legend>Room</legend>
            {(["sim", "live"] as Room[]).map((r) => (
              <label key={r}>
                <input type="radio" name="room" checked={setup.room === r} onChange={() => set({ room: r })} />
                {r === "sim" ? "Simulated" : "Live"}
              </label>
            ))}
          </fieldset>
          <label className={styles.consent}>
            <input type="checkbox" checked={setup.consent} onChange={(e) => set({ consent: e.target.checked })} />
            The room was told this show is recorded and agreed
          </label>
          <button type="submit" className={styles.start} disabled={!setup.operator.trim()}>
            {state === "done" ? "Run it again" : "Start the show"}
          </button>
        </form>
      )}
    </main>
  );
}
