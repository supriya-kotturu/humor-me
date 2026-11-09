"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactionKey, Room } from "./reactionKeys";

const WS_URL = "ws://localhost:8000/ws";
const FLASH_MS = 700;

export type ShowState = "idle" | "connecting" | "speaking" | "done" | "error";
export type EndStatus = "finished" | "aborted";
export type ReactionWindow = { bit: number; joke: number };

export type ShowSetup = {
  venue: string;
  energy: string;
  age: string;
  operator: string;
  room: Room;
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
  const socketRef = useRef<WebSocket | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  // Mirrors of state so `react` reads the current window without waiting for a re-render.
  const windowRef = useRef<ReactionWindow | null>(null);
  const roomRef = useRef<Room | null>(null);
  const flashTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      socketRef.current?.close();
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
      const node = context.createAnalyser();
      node.fftSize = 2048;
      node.smoothingTimeConstant = 0.75;
      node.connect(context.destination);
      setAnalyser(node);
      setLine("");
      setNotice("");
      setEndStatus(null);
      setFlash(null);
      closeWindow();
      roomRef.current = setup.room;
      setRoom(setup.room);
      setState("connecting");

      socketRef.current?.close();
      const socket = new WebSocket(WS_URL);
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;
      let finished = false;

      socket.onopen = () =>
        socket.send(
          JSON.stringify({
            type: "start",
            context: { venue: setup.venue, energy: setup.energy, age: setup.age },
            room: setup.room,
            operator: setup.operator.trim(),
          }),
        );
      socket.onerror = () => setState("error");
      socket.onclose = () => {
        closeWindow();
        setState((s) => (finished || s === "error" ? s : "idle"));
      };
      socket.onmessage = async (event) => {
        if (typeof event.data === "string") {
          const msg = JSON.parse(event.data);
          if (msg.type === "line") {
            setLine(msg.text);
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
          } else if (msg.type === "error") {
            setNotice(String(msg.message));
          } else if (msg.type === "show_end") {
            finished = true;
            closeWindow();
            setEndStatus(msg.status === "aborted" ? "aborted" : "finished");
            setState("done");
            socket.close();
          }
          return;
        }
        const audio = await context.decodeAudioData(event.data);
        const source = context.createBufferSource();
        source.buffer = audio;
        source.connect(node);
        source.onended = () => {
          if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "played" }));
        };
        source.start();
      };
    },
    [closeWindow],
  );

  // Only a Live show with an open window sends; a simulated room answers for itself.
  const react = useCallback((key: ReactionKey) => {
    const open = windowRef.current;
    const socket = socketRef.current;
    if (roomRef.current !== "live" || !open || socket?.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify({ type: "reaction", bit: open.bit, joke: open.joke, key }));
    setPressed(key);
    return true;
  }, []);

  return { state, line, analyser, start, room, reactionWindow, pressed, flash, notice, endStatus, react };
}
