"use client";

import { useEffect, useState } from "react";
import { useShow, type ShowSetup, type ShowState } from "./useShow";
import { REACTIONS, reactionKeyFor, type Room } from "./reactionKeys";
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
  const { state, line, analyser, start, room, reactionWindow, pressed, flash, notice, endStatus, react } = useShow();
  const [setup, setSetup] = useState<ShowSetup>({
    venue: "bar",
    energy: "normal",
    age: "mixed",
    operator: "",
    room: "sim",
  });
  const [confirming, setConfirming] = useState(false);
  const busy = state === "connecting" || state === "speaking";
  const windowOpen = reactionWindow !== null;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target instanceof HTMLElement ? event.target : null;
      const key = reactionKeyFor(
        {
          code: event.code,
          repeat: event.repeat,
          targetTag: target?.tagName ?? null,
          targetEditable: target?.isContentEditable ?? false,
        },
        { room, windowOpen },
      );
      if (!key) return;
      // Space would also scroll the page or click whatever button still holds focus.
      event.preventDefault();
      (document.activeElement as HTMLElement | null)?.blur();
      react(key);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [room, windowOpen, react]);

  const launch = () => {
    // Start keeps focus after the click; a later Space would re-trigger it instead of reacting.
    (document.activeElement as HTMLElement | null)?.blur();
    setConfirming(false);
    start(setup);
  };

  const onSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!setup.operator.trim()) return;
    if (setup.room === "live") setConfirming(true);
    else launch();
  };

  const set = (patch: Partial<ShowSetup>) => setSetup((s) => ({ ...s, ...patch }));

  const select = (name: keyof ShowSetup, text: string, options: string[]) => (
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
      <div className={busy ? styles.wave : styles.waveIdle} aria-hidden="true">
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
          <p className={windowOpen ? styles.windowOpen : styles.windowIdle} role="status">
            {windowOpen ? "React now" : room === "sim" ? "Simulated room" : "Waiting for the line to finish"}
          </p>
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
                  // Keeping focus off the button stops Space from also clicking it.
                  onMouseDown={(e) => e.preventDefault()}
                  // detail is 0 for a keyboard-generated click; the global listener already handled that press.
                  onClick={(e) => e.detail > 0 && room === "live" && react(key)}
                >
                  <span className={styles.hint}>{hint}</span>
                  <span className={styles.meaning}>{meaning}</span>
                </button>
              );
            })}
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
            <button type="button" className={styles.secondary} onClick={() => setConfirming(false)}>
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
          <button type="submit" className={styles.start} disabled={!setup.operator.trim()}>
            {state === "done" ? "Run it again" : "Start the show"}
          </button>
        </form>
      )}
    </main>
  );
}
