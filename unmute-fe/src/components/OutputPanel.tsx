import { RefreshCw, Play } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { AvatarViewer } from "@/components/AvatarViewer"
import type { PlanItem } from "@/hooks/useTranslation"
import type { AvatarController } from "@/lib/avatar-controller"

interface OutputPanelProps {
  plan: PlanItem[]
  currentToken?: string
  currentGifUrl?: string
  isPlaying: boolean
  onReplay: () => void
  onAvatarReady: (controller: AvatarController) => void
}

export function OutputPanel({ 
  plan, 
  currentToken, 
  currentGifUrl, 
  isPlaying, 
  onReplay,
  onAvatarReady 
}: OutputPanelProps) {
  const hasContent = plan.length > 0

  return (
    <Card className="p-8 flex flex-col gap-5 min-h-[600px]">
      {hasContent ? (
        <div className="flex flex-col gap-5 flex-1 animate-fade-in-up">
          {/* Players Grid - Vertical Stack */}
          <div className="flex flex-col gap-5 flex-1">
            {/* GIF Player - Top */}
            <div className="flex-1 min-h-0 flex flex-col rounded-[20px] border border-border-soft bg-bg-cream overflow-hidden">
              <div className="px-4 py-3 border-b border-border-soft bg-bg-card flex items-center justify-between">
                <span className="text-xs font-semibold tracking-[0.08em] uppercase text-text-muted">
                  Reference
                </span>
                {currentToken && (
                  <Badge variant="accent" className="text-[0.65rem] px-2.5 py-1">
                    {currentToken}
                  </Badge>
                )}
              </div>
              <div className="flex-1 flex items-center justify-center bg-bg-input relative">
                {currentGifUrl ? (
                  <img 
                    src={currentGifUrl} 
                    alt="Sign Language" 
                    className="max-h-full max-w-full object-contain"
                    style={{ maxHeight: '400px', maxWidth: '400px' }}
                  />
                ) : (
                  <div className="flex flex-col items-center gap-2.5 text-text-light">
                    <Play className="w-11 h-11 opacity-40" />
                    <span className="text-sm">Ready when you are</span>
                  </div>
                )}
              </div>
            </div>

            {/* 3D Avatar - Bottom */}
            <div className="flex-1 min-h-[300px] flex flex-col rounded-[20px] border border-border-soft bg-bg-cream overflow-hidden">
              <div className="px-4 py-3 border-b border-border-soft bg-bg-card flex items-center justify-between">
                <span className="text-xs font-semibold tracking-[0.08em] uppercase text-text-muted">
                  Avatar
                </span>
                <Badge variant="accent" className="text-[0.65rem] px-2.5 py-1">
                  Live
                </Badge>
              </div>
              <AvatarViewer onReady={onAvatarReady} />
            </div>
          </div>

          {/* Controls */}
          <div className="flex justify-center gap-3 pt-5">
            <Button 
              variant="outline" 
              onClick={onReplay}
              disabled={isPlaying || plan.length === 0}
            >
              <RefreshCw className="w-[18px] h-[18px]" />
              Play again
            </Button>
          </div>
        </div>
      ) : (
        /* Empty State */
        <div className="flex-1 flex flex-col items-center justify-center text-text-light">
          <Play className="w-16 h-16 opacity-30 mb-4" />
          <p className="text-sm">Enter text or use voice input to see the translation</p>
        </div>
      )}
    </Card>
  )
}
