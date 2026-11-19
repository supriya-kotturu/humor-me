"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactionKey, Room } from "./reactionKeys";
import { startRecorder } from "./recorder";

const WS_URL = "ws://localhost:8000/ws";
const FLASH_MS = 700;

export type ShowState = "idle" | "connecting" | "speaking" | "done" | "error";
export type EndStatus = "finished" | "aborted";
// ask: the prompt is up; saving: sent, waiting for the server; saved/skipped: done with this show.
export type RatingState = "ask" | "saving" | "saved" | "skipped";
export type ReactionWindow = { bit: number; joke: number };

export type ShowSetup = {
  venue: string;
  energy: string;
  age: string;
  operator: string;
  room: Room;
  consent: boolean;
};

export function useShow() {
  const [state, setState] = useState<ShowState>("idle");
  const [line, setLine] = useState("");
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [room, setRoom] = useState<Room | null>(null);
  const [reactionWindow, setReactionWindow] = useState<ReactionWindow | null>(null);
  const [pressed, setPressed] = useState<ReactionKey | null>(null);
  const [flash, setFlash] = useState<ReactionKey | null>(null);
  const [notice, setNotice] = useState("");
  const [endStatus, setEndStatus] = useState<EndStatus | null>(null);
  const [rating, setRating] = useState<RatingState | null>(null);
  const [paused, setPaused] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  // Mirrors of state so `react` reads the current window without waiting for a re-render.
  const windowRef = useRef<ReactionWindow | null>(null);
  const roomRef = useRef<Room | null>(null);
  const flashTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stopMicRef = useRef<(() => void) | null>(null);

  const micToken = useRef(0);

  const stopMic = useCallback(() => {
    micToken.current++; // a microphone still opening must stop as soon as it opens
    stopMicRef.current?.();
    stopMicRef.current = null;
  }, []);

  useEffect(
    () => () => {
      stopMicRef.current?.();
      const socket = socketRef.current;
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
      contextRef.current?.close();
      if (flashTimer.current) clearTimeout(flashTimer.current);
    },
    [],
  );

  const closeWindow = useCallback(() => {
    windowRef.current = null;
    setReactionWindow(null);
    setPressed(null);
  }, []);

  const start = useCallback(
    (setup: ShowSetup) => {
      // Browsers only allow audio that begins from a click, so the context is created here.
      const context = contextRef.current ?? new AudioContext();
      contextRef.current = context;
      void context.resume();
      let node = analyserRef.current;
      if (!node) {
        node = context.createAnalyser();
        node.fftSize = 2048;
        node.smoothingTimeConstant = 0.75;
        node.connect(context.destination);
        analyserRef.current = node;
      }
      setAnalyser(node);
      setLine("");
      setNotice("");
      setEndStatus(null);
      setRating(null);
      setFlash(null);
      setPaused(false);
      closeWindow();
      roomRef.current = setup.room;
      setRoom(setup.room);
      setState("connecting");

      // A late close from the previous show must not touch this one's window or state.
      stopMic();
      const old = socketRef.current;
      if (old) {
        old.onopen = old.onclose = old.onerror = old.onmessage = null;
        old.close();
      }
      const socket = new WebSocket(WS_URL);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;
      let finished = false;
      let failed = false;

      socket.onopen = () =>
        socket.send(
          JSON.stringify({
            type: "start",
            context: { venue: setup.venue, energy: setup.energy, age: setup.age },
            room: setup.room,
            operator: setup.operator.trim(),
            consent: setup.consent,
          }),
        );
      socket.onerror = () => {
        failed = true;
        setState("error");
      };
      socket.onclose = () => {
        stopMic();
        closeWindow();
        if (finished) {
          setRating((r) => (r === "ask" || r === "saving" ? "skipped" : r)); // nobody left to take the rating
          return;
        }
        if (failed) return;
        // The server logs a dropped show as aborted; the operator must see that too.
        setEndStatus("aborted");
        setNotice("Lost the connection to the backend.");
        setState("done");
      };
      socket.onmessage = async (event) => {
        if (typeof event.data === "string") {
          const msg = JSON.parse(event.data);
          if (msg.type === "line") {
            setLine(msg.text);
            setNotice("");
            setState("speaking");
          } else if (msg.type === "reaction_window") {
            if (msg.state === "open") {
              windowRef.current = { bit: msg.bit, joke: msg.joke };
              setReactionWindow(windowRef.current);
              setPressed(null);
            } else {
              closeWindow();
            }
          } else if (msg.type === "sim_reaction") {
            setFlash(msg.key);
            if (flashTimer.current) clearTimeout(flashTimer.current);
            flashTimer.current = setTimeout(() => setFlash(null), FLASH_MS);
          } else if (msg.type === "record") {
            // The server asked because the operator ticked consent; it waits for the first chunk before the show.
            const token = ++micToken.current;
            const send = (msg: object) => {
              if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(msg));
            };
            startRecorder({
              onChunk: (pcm) => {
                if (socket.readyState === WebSocket.OPEN) socket.send(pcm);
              },
              onStart: (start) => send({ type: "audio_start", ...start }),
              onHealth: (health) => send({ type: "audio_health", ...health }),
              onFail: (reason) =>
                send({ type: "audio_failed", reason: `Recording failed (${reason}), so the rest of this show is not recorded.` }),
            }).then(
              (stop) => {
                if (token === micToken.current && socket.readyState === WebSocket.OPEN) stopMicRef.current = stop;
                else stop(); // the server gave up or the show ended while the microphone was opening
              },
              (error) => {
                if (token !== micToken.current || socket.readyState !== WebSocket.OPEN) return;
                const why = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
                socket.send(JSON.stringify({ type: "audio_failed", reason: `Couldn't record (${why}), so this show is not recorded.` }));
              },
            );
          } else if (msg.type === "record_stop") {
            stopMic();
            if (msg.reason) setNotice(String(msg.reason));
          } else if (msg.type === "error") {
            setNotice(String(msg.message));
            setRating((r) => (r === "saving" ? "ask" : r)); // a refused rating can be tried again
          } else if (msg.type === "rating_saved") {
            setRating("saved");
            socket.close();
          } else if (msg.type === "show_end") {
            finished = true;
            setPaused(false);
            void context.resume();
            stopMic();
            closeWindow();
            setEndStatus(msg.status === "aborted" ? "aborted" : "finished");
            setState("done");
            // The socket stays open for the rating row, unless the show died before it had a log: then there is
            // nothing to rate, and the failure notice must stay on screen.
            if (msg.ratable === true) setRating("ask");
            else socket.close();
          }
          return;
        }
        const played = () => {
          if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "played" }));
        };
        try {
          const audio = await context.decodeAudioData(event.data);
          const source = context.createBufferSource();
          source.buffer = audio;
          source.connect(node);
          source.onended = played;
          source.start();
        } catch {
          // The server waits for `played` before the next line; without it the show hangs.
          setNotice("Couldn't play that line, so it was skipped.");
          played();
        }
      };
    },
    [closeWindow, stopMic],
  );

  // Pausing freezes playback: the line in progress stops mid-word, and the server waits for its `played` without a
  // time limit, so nothing new starts. A reaction window already open still closes on the server's clock. The room
  // recording keeps running, since it has its own capture path.
  const togglePause = useCallback(() => {
    const context = contextRef.current;
    if (!context) return;
    setPaused((was) => {
      void (was ? context.resume() : context.suspend());
      return !was;
    });
  }, []);

  // Only a Live show with an open window sends; a simulated room answers for itself.
  const react = useCallback((key: ReactionKey) => {
    const open = windowRef.current;
    const socket = socketRef.current;
    if (roomRef.current !== "live" || !open || socket?.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify({ type: "reaction", bit: open.bit, joke: open.joke, key }));
    setPressed(key);
    return true;
  }, []);

  const rate = useCallback((value: number | null) => {
    const socket = socketRef.current;
    if (value === null || socket?.readyState !== WebSocket.OPEN) {
      setRating("skipped");
      socket?.close();
      return;
    }
    setRating("saving");
    socket.send(JSON.stringify({ type: "rating", value }));
  }, []);

  return {
    state, line, analyser, start, room, reactionWindow, pressed, flash, notice, endStatus, rating, rate, react,
    paused, togglePause,
  };
}
