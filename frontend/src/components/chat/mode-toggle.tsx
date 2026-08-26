import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Database, Brain, Globe } from "lucide-react"

export interface ChatModes {
  rag: boolean
  reasoning: boolean
  webSearch: boolean
}

/** Which modes the server behind this UI implements. Undefined until it has been asked. */
export interface ModeAvailability {
  rag: boolean
  reasoning: boolean
  webSearch: boolean
}

interface ModeToggleProps {
  readonly modes: ChatModes
  readonly onModesChange: (modes: ChatModes) => void
  readonly available?: ModeAvailability
}

const modeConfig = [
  {
    key: "rag" as const,
    icon: Database,
    label: "RAG Mode",
    description: "Use uploaded documents for context",
    unavailableReason: "This deployment has no vector index configured.",
  },
  {
    key: "reasoning" as const,
    icon: Brain,
    label: "Reasoning",
    description: "The model shows its working, which the server strips before replying",
    // Not a per-request switch: whether a model emits a reasoning block is a property of the
    // model, so there is nothing here for the user to turn on.
    unavailableReason: "The configured model does not produce a reasoning block.",
  },
  {
    key: "webSearch" as const,
    icon: Globe,
    label: "Web Search",
    description: "Search the web for answers",
    unavailableReason: "Not implemented on this server.",
  },
]


/** Visual state for one toggle. Extracted so the JSX stays a single expression rather than a
 * ternary nested inside a ternary, which is unreadable once a third state exists. */
function stateClass(isAvailable: boolean, isActive: boolean): string {
  if (!isAvailable) return "text-muted-foreground/40 cursor-not-allowed line-through decoration-1"
  if (isActive) return "bg-primary/15 text-primary border border-primary/30 hover:bg-primary/20"
  return "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
}

export function ModeToggle({ modes, onModesChange, available }: ModeToggleProps) {
  const toggleMode = (key: keyof ChatModes) => {
    if (available && !available[key]) return
    onModesChange({ ...modes, [key]: !modes[key] })
  }

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex items-center gap-1">
        {modeConfig.map(({ key, icon: Icon, label, description, unavailableReason }) => {
          const isAvailable = available ? available[key] : true
          return (
            <Tooltip key={key}>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => toggleMode(key)}
                  disabled={!isAvailable}
                  aria-disabled={!isAvailable}
                  className={cn(
                    "h-8 px-2.5 gap-1.5 text-xs font-medium transition-all duration-200",
                    stateClass(isAvailable, modes[key])
                  )}
                >
                  <Icon className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">{label}</span>
                </Button>
              </TooltipTrigger>
              {/* A disabled control has to say why, or it reads as a bug rather than a boundary. */}
              <TooltipContent side="top" className="bg-popover border-border">
                <p className="font-medium">{label}</p>
                <p className="text-xs text-muted-foreground">
                  {isAvailable ? description : unavailableReason}
                </p>
              </TooltipContent>
            </Tooltip>
          )
        })}
      </div>
    </TooltipProvider>
  )
}
