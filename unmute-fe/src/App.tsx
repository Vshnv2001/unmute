import { useState, useCallback, useEffect } from "react"
import { Header } from "@/components/Header"
import { InputPanel } from "@/components/InputPanel"
import { OutputPanel } from "@/components/OutputPanel"
import { VideoCall } from "@/components/VideoCall"
import { useTranslation, type TranslationResult } from "@/hooks/useTranslation"
import { useSignPlayback } from "@/hooks/useSignPlayback"
import type { AvatarController } from "@/lib/avatar-controller"
import { Button } from "@/components/ui/button"
import { Video, MessageSquare } from "lucide-react"

type AppMode = "translate" | "video"

function App() {
  const [mode, setMode] = useState<AppMode>("translate")
  const [avatar, setAvatar] = useState<AvatarController | null>(null)
  const { result, setResult, isLoading, translate } = useTranslation()
  const { isPlaying, currentToken, currentGifUrl, playSequence, stopPlayback } = useSignPlayback({ avatar })

  const handleTranslate = useCallback(async (text: string) => {
    stopPlayback()
    const translationResult = await translate(text)
    if (translationResult && translationResult.plan.length > 0) {
      playSequence(translationResult.plan)
    }
  }, [translate, playSequence, stopPlayback])

  const handleVoiceResult = useCallback((voiceResult: TranslationResult) => {
    stopPlayback()
    setResult(voiceResult)
    if (voiceResult.plan.length > 0) {
      playSequence(voiceResult.plan)
    }
  }, [setResult, playSequence, stopPlayback])

  const handleReplay = useCallback(() => {
    if (result && result.plan.length > 0) {
      playSequence(result.plan)
    }
  }, [result, playSequence])

  const handleAvatarReady = useCallback((controller: AvatarController) => {
    setAvatar(controller)
  }, [])

  // Auto-play when result changes
  useEffect(() => {
    if (result && result.plan.length > 0 && avatar && !isPlaying) {
      // Only auto-play if not already playing (avoid duplicate plays)
    }
  }, [result, avatar, isPlaying])

  return (
    <div className="min-h-screen flex flex-col items-center py-12 px-6 max-w-[1400px] mx-auto w-full">
      <Header />
      
      {/* Mode Switcher */}
      <div className="flex gap-2 mb-8 animate-fade-in" style={{ animationDelay: "0.1s" }}>
        <Button 
          variant={mode === "translate" ? "default" : "outline"}
          onClick={() => setMode("translate")}
          className={mode === "translate" 
            ? "bg-[var(--accent-terracotta)] hover:bg-[var(--accent-warm)] text-white" 
            : "border-[var(--border-warm)] text-[var(--text-secondary)] hover:bg-[var(--bg-cream)]"
          }
        >
          <MessageSquare className="w-4 h-4 mr-2" />
          Translate
        </Button>
        <Button 
          variant={mode === "video" ? "default" : "outline"}
          onClick={() => setMode("video")}
          className={mode === "video" 
            ? "bg-[var(--accent-terracotta)] hover:bg-[var(--accent-warm)] text-white" 
            : "border-[var(--border-warm)] text-[var(--text-secondary)] hover:bg-[var(--bg-cream)]"
          }
        >
          <Video className="w-4 h-4 mr-2" />
          Video Call
        </Button>
      </div>
      
      {mode === "translate" ? (
        <main className="w-full grid grid-cols-1 lg:grid-cols-[1fr_1.5fr] gap-8 animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
          <InputPanel
            onTranslate={handleTranslate}
            onVoiceResult={handleVoiceResult}
            isLoading={isLoading}
            result={result}
          />
          
          <OutputPanel
            plan={result?.plan || []}
            currentToken={currentToken}
            currentGifUrl={currentGifUrl}
            isPlaying={isPlaying}
            onReplay={handleReplay}
            onAvatarReady={handleAvatarReady}
          />
        </main>
      ) : (
        <main className="w-full max-w-4xl animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
          <VideoCall />
        </main>
      )}
    </div>
  )
}

export default App
