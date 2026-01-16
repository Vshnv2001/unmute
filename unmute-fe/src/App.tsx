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
      <div className="inline-flex gap-1.5 p-1.5 bg-[var(--bg-cream)] rounded-[20px] border border-[var(--border-soft)] mb-8 animate-fade-in" style={{ animationDelay: "0.1s" }}>
        <Button 
          variant="ghost"
          onClick={() => setMode("translate")}
          className={mode === "translate" 
            ? "bg-bg-card text-accent-terracotta shadow-soft font-medium px-6 py-3.5 rounded-[14px]" 
            : "text-text-secondary hover:text-text-primary hover:bg-white/50 px-6 py-3.5 rounded-[14px]"
          }
        >
          <MessageSquare className="w-5 h-5 mr-2" />
          Translate
        </Button>
        <Button 
          variant="ghost"
          onClick={() => setMode("video")}
          className={mode === "video" 
            ? "bg-bg-card text-accent-terracotta shadow-soft font-medium px-6 py-3.5 rounded-[14px]" 
            : "text-text-secondary hover:text-text-primary hover:bg-white/50 px-6 py-3.5 rounded-[14px]"
          }
        >
          <Video className="w-5 h-5 mr-2" />
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
        <main className="w-full max-w-6xl animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
          <VideoCall />
        </main>
      )}
    </div>
  )
}

export default App
