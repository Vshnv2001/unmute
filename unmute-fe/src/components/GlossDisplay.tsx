import { Badge } from "@/components/ui/badge"

interface GlossDisplayProps {
  gloss: string[]
  activeToken?: string
  unmatched?: string[]
}

export function GlossDisplay({ gloss, activeToken, unmatched = [] }: GlossDisplayProps) {
  if (gloss.length === 0) {
    return (
      <div className="text-center text-text-muted text-sm italic">
        No gloss tokens found
      </div>
    )
  }

  return (
    <div className="flex flex-wrap gap-2">
      {gloss.map((token, index) => {
        const isMissing = unmatched.includes(token)
        const isActive = activeToken === token
        
        return (
          <Badge
            key={`${token}-${index}`}
            variant={isMissing ? "missing" : isActive ? "active" : "default"}
          >
            {token}
          </Badge>
        )
      })}
    </div>
  )
}
