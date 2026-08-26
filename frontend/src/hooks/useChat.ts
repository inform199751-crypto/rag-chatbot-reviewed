import { useCallback, useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { ChatWebSocket, type SourcesData, type StreamEvent } from '../services/websocket';
import { getConversation, resetChatHistory, type StoredMessage } from '../services/api';

export interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  isStreaming?: boolean;
  /** Retrieved context, when the request used RAG. Rendered apart from the answer. */
  sources?: SourcesData;
  /** False when the answer came from the model alone rather than from the documents. */
  grounded?: boolean;
  /** The server chose not to answer because it had no supporting documents. */
  declined?: boolean;
  isError?: boolean;
}

/** Apply a change to the in-flight bot message, leaving everything else alone. */
function patchLastBot(messages: Message[], patch: (msg: Message) => Message): Message[] {
  const last = messages[messages.length - 1];
  if (last?.sender !== 'bot') return messages;
  return [...messages.slice(0, -1), patch(last)];
}

/**
 * Fold one stream event into the message list.
 *
 * Kept out of the component as a pure function so the protocol handling can be read -- and
 * tested -- without a React tree, and so the callback nesting inside the effect stays shallow.
 */
export function applyStreamEvent(messages: Message[], event: StreamEvent): Message[] {
  switch (event.type) {
    case 'sources':
      return patchLastBot(messages, (msg) => ({
        ...msg,
        sources: event.data,
        grounded: event.data.grounded,
      }));

    case 'answer_start':
      return patchLastBot(messages, (msg) => ({
        ...msg,
        grounded: event.data.grounded,
        isStreaming: true,
      }));

    case 'token':
      return patchLastBot(messages, (msg) => ({
        ...msg,
        text: msg.text + event.data.text,
        isStreaming: true,
      }));

    case 'error':
      return patchLastBot(messages, (msg) => ({
        ...msg,
        text: event.data.message,
        isError: true,
        isStreaming: false,
      }));

    case 'done':
      // The single signal that the response is over. This used to be inferred from 500 ms of
      // silence, which a slow model crosses between ordinary tokens -- the UI would close the
      // message and unlock the input while generation was still running.
      return patchLastBot(messages, (msg) => ({
        ...msg,
        isStreaming: false,
        declined: event.data.declined,
      }));

    default:
      return messages;
  }
}

/** Attach a transport error to the current bot message, or add one if there is none. */
export function applyTransportError(messages: Message[], error: string, nextId: number): Message[] {
  const last = messages[messages.length - 1];
  if (last?.sender === 'bot') {
    return [...messages.slice(0, -1), { ...last, text: error, isError: true, isStreaming: false }];
  }
  return [
    ...messages,
    { id: nextId, text: error, sender: 'bot', timestamp: new Date(), isError: true },
  ];
}


const CONVERSATION_KEY = 'rag-chatbot.conversation-id';

/**
 * The thread this browser is in, created on first use and kept across reloads.
 *
 * Generated client-side rather than handed out by the server so that a conversation exists
 * before its first message does -- there is nothing to ask the server for until something has
 * been said, and a round trip on page load to obtain an empty thread buys nothing.
 */
function loadConversationId(): string {
  const existing = localStorage.getItem(CONVERSATION_KEY);
  if (existing) return existing;
  const fresh = crypto.randomUUID();
  localStorage.setItem(CONVERSATION_KEY, fresh);
  return fresh;
}

/** Turn a stored row back into the shape the UI renders. */
function fromStored(row: StoredMessage, id: number): Message {
  return {
    id,
    text: row.content,
    sender: row.role === 'user' ? 'user' : 'bot',
    timestamp: new Date(row.created_at),
    // `?? undefined` rather than leaving null: the UI treats undefined as "retrieval never ran"
    // and false as "ran and found nothing", and those render differently.
    grounded: row.grounded ?? undefined,
    sources: row.sources.length
      ? { documents: row.sources, grounded: row.grounded ?? false, reason: null, message: '' }
      : undefined,
  };
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<ChatWebSocket | null>(null);
  const idRef = useRef(0);
  const [conversationId, setConversationId] = useState(loadConversationId);

  // Defined at hook level rather than inside the effect: nesting them one layer deeper puts the
  // `setMessages` updater five callbacks in, which is past what the linter allows and past what
  // is comfortable to read.
  const handleEvent = useCallback((event: StreamEvent) => {
    // flushSync keeps token order visible during a fast stream; without it React can batch
    // updates and the text appears in bursts.
    flushSync(() => {
      setMessages((prev) => applyStreamEvent(prev, event));
      if (event.type === 'done' || event.type === 'error') {
        setIsStreaming(false);
      }
    });
  }, []);

  const handleError = useCallback((error: string) => {
    flushSync(() => {
      setMessages((prev) => applyTransportError(prev, error, ++idRef.current));
      // A transport error never produces a `done`, so the lock has to be released here or the
      // input stays disabled for the rest of the session.
      setIsStreaming(false);
    });
  }, []);

  useEffect(() => {
    const ws = new ChatWebSocket(handleEvent, handleError, conversationId);
    wsRef.current = ws;
    return () => {
      ws.disconnect();
    };
  }, [handleEvent, handleError, conversationId]);

  // Restore the transcript. A failure here is left silent on purpose: the thread is a
  // convenience, and an error banner about history the user has not asked for yet would be the
  // first thing they see on an otherwise working page.
  useEffect(() => {
    let cancelled = false;
    getConversation(conversationId)
      .then((data) => {
        if (cancelled || data.messages.length === 0) return;
        idRef.current = data.messages.length;
        setMessages(data.messages.map((row, i) => fromStored(row, i + 1)));
      })
      .catch(() => {
        /* start empty */
      });
    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  const sendMessage = useCallback(
    (text: string, rag: boolean) => {
      if (!text.trim() || isStreaming) return;

      const userMsg: Message = {
        id: ++idRef.current,
        text,
        sender: 'user',
        timestamp: new Date(),
      };
      const botPlaceholder: Message = {
        id: ++idRef.current,
        text: '',
        sender: 'bot',
        timestamp: new Date(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, botPlaceholder]);
      setIsStreaming(true);
      wsRef.current?.sendMessage(text, rag);
    },
    [isStreaming],
  );

  /**
   * Start a new thread.
   *
   * The old one is kept rather than deleted -- persisting a transcript and then destroying it
   * on "new chat" would defeat the point. Deleting is a separate, explicit action.
   */
  const clearMessages = useCallback(() => {
    const fresh = crypto.randomUUID();
    localStorage.setItem(CONVERSATION_KEY, fresh);
    setMessages([]);
    idRef.current = 0;
    setIsStreaming(false);
    setConversationId(fresh);
    resetChatHistory();
  }, []);

  return { messages, isStreaming, sendMessage, clearMessages };
}
