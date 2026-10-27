import { useEffect, useRef, useState } from "react";
import { BASE } from "./api.js";

const EVENT_TYPES = [
  "hello", "tick", "day_closed", "order_placed", "order_received",
  "retrain_started", "model_retrained", "retrain_failed",
];

/**
 * Subscribes to the backend's Server-Sent Events stream.
 * `onEvent(type, data)` is called for every event. EventSource reconnects on
 * its own; `connected` reflects the current state so the UI can show it.
 */
export function useLiveStream(onEvent) {
  const [connected, setConnected] = useState(false);
  const handler = useRef(onEvent);
  handler.current = onEvent;

  useEffect(() => {
    const es = new EventSource(`${BASE}/stream`);
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    for (const type of EVENT_TYPES) {
      es.addEventListener(type, (e) => {
        setConnected(true);
        try {
          handler.current(type, JSON.parse(e.data));
        } catch { /* ignore malformed event */ }
      });
    }
    return () => es.close();
  }, []);

  return connected;
}
