/**
 * WebSocket client for the chat stream.
 *
 * The server sends one JSON object per frame, discriminated by `type`. Before this, every frame
 * was a bare string and the client appended all of them to the current message -- so an error,
 * a section heading and a generated token were indistinguishable once they arrived, and the end
 * of a response had to be guessed from a silence timer.
 *
 * Unknown event types are ignored rather than rendered. That is what lets a newer server add
 * events without breaking a client that has not been rebuilt.
 */

export interface SourceItem {
  score: number;
  document: string | null;
  content_preview: string;
}

export interface SourcesData {
  documents: SourceItem[];
  grounded: boolean;
  /** empty_index | below_threshold | below_rerank_threshold -- null when grounded. */
  reason: string | null;
  message: string;
}

export type StreamEvent =
  | { type: 'sources'; data: SourcesData }
  | { type: 'answer_start'; data: { grounded: boolean } }
  | { type: 'token'; data: { text: string } }
  | { type: 'done'; data: { took_seconds: number | null; declined: boolean } }
  | { type: 'error'; data: { message: string } };

type EventHandler = (event: StreamEvent) => void;
type ErrorHandler = (error: string) => void;

const WS_BASE = (() => {
  const apiUrl = import.meta.env.VITE_API_URL ?? '';
  if (apiUrl) {
    return apiUrl.replace(/^http/, 'ws');
  }
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}`;
})();

const WS_URL = `${WS_BASE}/chat/stream`;

const KNOWN_TYPES = new Set(['sources', 'answer_start', 'token', 'done', 'error']);

export class ChatWebSocket {
  private ws: WebSocket | null = null;
  private readonly onEvent: EventHandler;
  private readonly onError: ErrorHandler;

  constructor(onEvent: EventHandler, onError: ErrorHandler) {
    this.onEvent = onEvent;
    this.onError = onError;
  }

  private handleFrame(raw: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      // A frame that is not JSON means the server is speaking the old string protocol, or
      // something in front of it is rewriting frames. Surfacing it as an error is better than
      // silently pasting it into the answer, which is exactly the behaviour being replaced.
      this.onError('Received a malformed frame from the server');
      return;
    }

    const event = parsed as StreamEvent;
    if (!event || typeof event.type !== 'string' || !KNOWN_TYPES.has(event.type)) {
      return; // forward compatibility: skip what we do not understand
    }
    this.onEvent(event);
  }

  private connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }
      if (this.ws?.readyState === WebSocket.CONNECTING) {
        // Wait for existing connection attempt
        const checkConnection = setInterval(() => {
          if (this.ws?.readyState === WebSocket.OPEN) {
            clearInterval(checkConnection);
            resolve();
          } else if (this.ws?.readyState === WebSocket.CLOSED) {
            clearInterval(checkConnection);
            reject(new Error('Connection failed'));
          }
        }, 50);
        return;
      }

      this.ws = new WebSocket(WS_URL);

      this.ws.onopen = () => {
        resolve();
      };

      this.ws.onmessage = (event) => {
        this.handleFrame(event.data as string);
      };

      this.ws.onerror = () => {
        this.onError('WebSocket connection error');
        reject(new Error('WebSocket connection error'));
      };

      this.ws.onclose = () => {
        this.ws = null;
        // No auto-reconnect
      };
    });
  }

  async sendMessage(text: string, rag: boolean): Promise<void> {
    try {
      await this.connect();
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ text, rag }));
      } else {
        this.onError('WebSocket is not connected');
      }
    } catch (error) {
      this.onError(error instanceof Error ? error.message : 'Failed to connect');
    }
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }

  reconnect(): void {
    this.disconnect();
    this.ws = null;
  }
}
