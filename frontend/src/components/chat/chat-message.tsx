import { cn } from "@/lib/utils"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { AlertTriangle, Bot, FileText, Info, User } from "lucide-react"
import ReactMarkdown from "react-markdown"
import type { SourcesData } from "@/services/websocket"

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  isStreaming?: boolean
  /** Retrieved context for this answer, when the request used RAG. */
  sources?: SourcesData
  /** False when the answer came from the model alone rather than from the documents. */
  grounded?: boolean
  /** The server declined to answer because it had no supporting documents. */
  declined?: boolean
  isError?: boolean
}

interface ChatMessageProps {
  message: Message
}

/**
 * The retrieved chunks, or an explanation of why there were none.
 *
 * Rendered apart from the answer on purpose. These used to arrive as one markdown blob glued to
 * the front of the reply, which made it impossible to tell what the model was given from what
 * the model said.
 */
function SourcesBlock({ sources }: { sources: SourcesData }) {
  if (!sources.grounded) {
    return (
      <div className="mb-3 flex gap-2 rounded-lg border border-border/40 bg-secondary/40 px-3 py-2 text-xs text-muted-foreground">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span>{sources.message || "No relevant documents were found."}</span>
      </div>
    )
  }

  return (
    <details className="group mb-3 rounded-lg border border-border/40 bg-secondary/40">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground">
        <FileText className="h-3.5 w-3.5 shrink-0" />
        {sources.documents.length} source{sources.documents.length === 1 ? "" : "s"} used
      </summary>
      <div className="space-y-2 px-3 pb-3">
        {sources.documents.map((doc, i) => (
          <div key={i} className="rounded border border-border/30 bg-background/40 p-2 text-xs">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="truncate font-mono text-foreground/80">
                {doc.document?.split(/[\\/]/).pop() ?? "unknown"}
              </span>
              <span className="shrink-0 rounded bg-primary/15 px-1.5 py-0.5 text-primary">
                {doc.score.toFixed(3)}
              </span>
            </div>
            <p className="leading-relaxed text-muted-foreground">{doc.content_preview}</p>
          </div>
        ))}
      </div>
    </details>
  )
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user"
  // Only worth flagging once there is an answer to attribute. `grounded === undefined` means
  // the request never went through retrieval, so there is nothing to say about it.
  const showUngroundedBadge =
    !isUser && message.grounded === false && !message.declined && Boolean(message.content)

  return (
    <div
      className={cn(
        "flex gap-4 animate-fade-in-up",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      <Avatar className={cn(
        "h-10 w-10 shrink-0 border-2",
        isUser
          ? "border-primary/30 bg-primary/10"
          : "border-accent/30 bg-accent/10"
      )}>
        <AvatarFallback className={cn(
          "text-foreground",
          isUser ? "bg-message-user" : "bg-message-ai"
        )}>
          {isUser ? (
            <User className="h-5 w-5" />
          ) : (
            <Bot className="h-5 w-5 text-primary" />
          )}
        </AvatarFallback>
      </Avatar>

      <div
        className={cn(
          "max-w-[75%] rounded-2xl px-5 py-4",
          isUser
            ? "bg-message-user border border-border/50"
            : "bg-message-ai border border-border/30",
          message.isError && "border-destructive/40 bg-destructive/10"
        )}
      >
        {message.sources && <SourcesBlock sources={message.sources} />}

        {showUngroundedBadge && (
          <div className="mb-3 flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200/90">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>
              Answered from the model&apos;s own knowledge, not from your documents.
            </span>
          </div>
        )}

        {message.isError ? (
          <div className="flex gap-2 text-sm text-destructive-foreground/90">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{message.content}</span>
          </div>
        ) : message.isStreaming && !message.content ? (
          <TypingIndicator />
        ) : !message.content ? null : (
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown
              components={{
                p: ({ children }) => (
                  <p className="text-foreground/90 leading-relaxed mb-2 last:mb-0">
                    {children}
                  </p>
                ),
                code: ({ children, className }) => {
                  const isInline = !className
                  return isInline ? (
                    <code className="bg-secondary px-1.5 py-0.5 rounded text-primary font-mono text-sm">
                      {children}
                    </code>
                  ) : (
                    <code className="block bg-secondary p-4 rounded-lg overflow-x-auto font-mono text-sm text-foreground/90">
                      {children}
                    </code>
                  )
                },
                pre: ({ children }) => (
                  <pre className="bg-secondary rounded-lg overflow-hidden my-3">
                    {children}
                  </pre>
                ),
                ul: ({ children }) => (
                  <ul className="list-disc list-inside space-y-1 text-foreground/90 my-2">
                    {children}
                  </ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal list-inside space-y-1 text-foreground/90 my-2">
                    {children}
                  </ol>
                ),
                h1: ({ children }) => (
                  <h1 className="text-xl font-semibold text-foreground mb-3">{children}</h1>
                ),
                h2: ({ children }) => (
                  <h2 className="text-lg font-semibold text-foreground mb-2">{children}</h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-base font-semibold text-foreground mb-2">{children}</h3>
                ),
                a: ({ children, href }) => (
                  <a
                    href={href}
                    className="text-primary hover:underline"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {children}
                  </a>
                ),
                blockquote: ({ children }) => (
                  <blockquote className="border-l-2 border-primary/50 pl-4 italic text-muted-foreground my-3">
                    {children}
                  </blockquote>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
            {message.isStreaming && message.content && (
              <span className="inline-block w-2 h-5 bg-primary ml-1 animate-pulse" />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-2 h-2 bg-primary/60 rounded-full animate-typing-dot"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </div>
  )
}
