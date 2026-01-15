import { useState } from "react"
import { PenLine, Mic } from "lucide-react"
import { Card } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { VoiceRecorder } from "@/components/VoiceRecorder"
import { GlossDisplay } from "@/components/GlossDisplay"
import type { TranslationResult } from "@/hooks/useTranslation"

interface InputPanelProps {
  onTranslate: (text: string) => Promise<void>
  onVoiceResult: (result: TranslationResult) => void
  isLoading: boolean
  result: TranslationResult | null
}

export function InputPanel({ onTranslate, onVoiceResult, isLoading, result }: InputPanelProps) {
  const [mode, setMode] = useState<"text" | "voice">("text")
  const [inputText, setInputText] = useState("")

  const handleTranslate = async () => {
    if (!inputText.trim()) return
    await onTranslate(inputText.trim())
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleTranslate()
    }
  }

  return (
    <Card className="p-8 flex flex-col">
      {/* Mode Tabs */}
      <div className="flex justify-center mb-8">
        <Tabs value={mode} onValueChange={(v) => setMode(v as "text" | "voice")}>
          <TabsList>
            <TabsTrigger value="text">
              <PenLine className="w-[18px] h-[18px]" />
              <span>Text</span>
            </TabsTrigger>
            <TabsTrigger value="voice">
              <Mic className="w-[18px] h-[18px]" />
              <span>Voice</span>
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Text Input Section */}
      {mode === "text" && (
        <div className="animate-fade-in">
          <Textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="What would you like to say?"
            rows={4}
          />
          <Button 
            className="w-full mt-5" 
            onClick={handleTranslate}
            disabled={isLoading || !inputText.trim()}
          >
            {isLoading ? "Translating..." : "Translate to Sign Language"}
          </Button>
        </div>
      )}

      {/* Voice Input Section */}
      {mode === "voice" && (
        <VoiceRecorder onResult={onVoiceResult} />
      )}

      {/* Output Section - Gloss Display */}
      {result && (
        <div className="mt-8 animate-fade-in-up">
          <div className="h-px bg-gradient-to-r from-transparent via-border-soft to-transparent mb-6" />
          
          <div className="mb-6">
            <div className="text-xs font-semibold tracking-[0.1em] uppercase text-text-muted mb-3">
              Sign Sequence
            </div>
            <GlossDisplay 
              gloss={result.gloss || []} 
              unmatched={result.unmatched || []}
            />
          </div>

          {/* Status Notes */}
          {(result.notes || (result.unmatched && result.unmatched.length > 0)) && (
            <div className="text-center text-sm text-text-muted italic">
              {result.unmatched && result.unmatched.length > 0 && (
                <span>Unmatched: {result.unmatched.join(", ")}. </span>
              )}
              {result.notes}
            </div>
          )}
        </div>
      )}
    </Card>
  )
}
