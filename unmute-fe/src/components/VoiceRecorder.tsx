import { useState } from "react"
import { Mic, Square, RotateCcw, Check } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useVoiceRecording } from "@/hooks/useVoiceRecording"
import { useTranslation, type TranslationResult } from "@/hooks/useTranslation"
import { cn } from "@/lib/utils"

interface VoiceRecorderProps {
  onResult: (result: TranslationResult) => void
}

export function VoiceRecorder({ onResult }: VoiceRecorderProps) {
  const [showReview, setShowReview] = useState(false)
  const [editedTranscription, setEditedTranscription] = useState("")
  const { translate, isLoading: isTranslating } = useTranslation()

  const {
    isRecording,
    isProcessing,
    transcription,
    toggleRecording,
    clearTranscription,
  } = useVoiceRecording({
    onResult: (result) => {
      // Full translation received directly
      onResult(result)
      setShowReview(false)
    },
    onTranscription: (text) => {
      // Only transcription received, show review
      setEditedTranscription(text)
      setShowReview(true)
    },
  })

  const handleConfirm = async () => {
    if (!editedTranscription.trim()) return
    
    const result = await translate(editedTranscription.trim())
    if (result) {
      onResult(result)
      setShowReview(false)
      clearTranscription()
    }
  }

  const handleRetry = () => {
    setShowReview(false)
    setEditedTranscription("")
    clearTranscription()
  }

  const getStatusText = () => {
    if (isRecording) return "Recording... Click to stop"
    if (isProcessing) return "Transcribing & Translating..."
    return "Tap to start speaking"
  }

  return (
    <div className="flex flex-col items-center py-10 animate-fade-in">
      {/* Mic Button */}
      <button
        onClick={toggleRecording}
        disabled={isProcessing}
        className={cn(
          "w-[120px] h-[120px] rounded-full bg-bg-cream border-2 border-border-warm cursor-pointer relative flex items-center justify-center transition-all duration-400",
          "before:content-[''] before:absolute before:inset-[-6px] before:rounded-full before:bg-gradient-accent before:opacity-0 before:transition-opacity before:duration-400 before:-z-10",
          "hover:border-accent-terracotta hover:bg-bg-card hover:shadow-hover hover:scale-105",
          "hover:before:opacity-15",
          isRecording && "border-[#E07A5F] bg-[#FEF3F0] animate-pulse-recording before:bg-gradient-to-br before:from-[#E07A5F] before:to-[#F2A285] before:opacity-25",
          isProcessing && "opacity-60 cursor-not-allowed"
        )}
      >
        {isRecording ? (
          <Square className="w-10 h-10 text-[#E07A5F] fill-current" />
        ) : (
          <Mic className={cn(
            "w-10 h-10 text-text-secondary transition-all duration-300",
            "group-hover:text-accent-terracotta"
          )} />
        )}
      </button>

      <p className="mt-5 text-sm text-text-secondary">
        {getStatusText()}
      </p>

      {/* Audio Wave Animation */}
      {isRecording && (
        <div className="flex items-center justify-center gap-[5px] h-9 mt-5">
          {[0, 1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="w-1 bg-gradient-accent rounded animate-wave"
              style={{
                animationDelay: `${i * 0.12}s`,
                height: [10, 18, 28, 18, 10][i],
              }}
            />
          ))}
        </div>
      )}

      {/* Transcription Review */}
      {showReview && transcription && (
        <div className="w-full mt-6 p-6 bg-bg-cream border border-border-soft rounded-[20px] animate-fade-in">
          <div className="text-xs font-semibold tracking-[0.1em] uppercase text-text-muted mb-3">
            What we heard
          </div>
          <Input
            value={editedTranscription}
            onChange={(e) => setEditedTranscription(e.target.value)}
            placeholder="Your words will appear here..."
          />
          <div className="flex gap-3 mt-4">
            <Button
              className="flex-1"
              onClick={handleConfirm}
              disabled={isTranslating || !editedTranscription.trim()}
            >
              {isTranslating ? (
                <>
                  <div className="w-[18px] h-[18px] border-2 border-white/25 border-t-white rounded-full animate-spin" />
                  Translating...
                </>
              ) : (
                <>
                  <Check className="w-[18px] h-[18px]" />
                  Looks good
                </>
              )}
            </Button>
            <Button variant="outline" onClick={handleRetry}>
              <RotateCcw className="w-[18px] h-[18px]" />
              Try again
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
