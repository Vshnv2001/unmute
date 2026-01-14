import { useState, useCallback, useRef } from "react"
import type { TranslationResult } from "./useTranslation"

const TRANSCRIBE_API_URL = "http://127.0.0.1:8000/api/transcribe"

interface UseVoiceRecordingOptions {
  onResult?: (result: TranslationResult) => void
  onTranscription?: (transcription: string) => void
  onError?: (error: string) => void
}

export function useVoiceRecording(options: UseVoiceRecordingOptions = {}) {
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [transcription, setTranscription] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onloadend = () => {
        const base64 = (reader.result as string).split(",")[1]
        resolve(base64)
      }
      reader.onerror = reject
      reader.readAsDataURL(blob)
    })
  }

  const transcribeAudio = useCallback(async () => {
    try {
      setIsProcessing(true)
      setError(null)

      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" })
      const base64Audio = await blobToBase64(audioBlob)

      const res = await fetch(TRANSCRIBE_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_data: base64Audio,
          mime_type: "audio/webm",
          auto_translate: true,
        }),
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.detail || "Transcription failed")
      }

      const data = await res.json()

      // Check if response contains full translation data
      if (data.plan && Array.isArray(data.plan) && data.gloss && Array.isArray(data.gloss)) {
        // Full translation received
        options.onResult?.(data as TranslationResult)
      } else if (data.transcription && data.transcription.trim()) {
        // Transcription only
        setTranscription(data.transcription)
        options.onTranscription?.(data.transcription)
      } else {
        throw new Error("No speech detected. Please try again.")
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : "Transcription failed"
      setError(message)
      options.onError?.(message)
    } finally {
      setIsProcessing(false)
    }
  }, [options])

  const startRecording = useCallback(async () => {
    try {
      setError(null)
      setTranscription(null)

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })

      streamRef.current = stream
      audioChunksRef.current = []

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      })

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop())
        await transcribeAudio()
      }

      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start(100) // Collect data every 100ms
      setIsRecording(true)
    } catch (e) {
      const message = e instanceof Error ? e.message : "Could not access microphone"
      setError(message)
      options.onError?.(message)
    }
  }, [transcribeAudio, options])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }, [])

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }, [isRecording, startRecording, stopRecording])

  const clearTranscription = useCallback(() => {
    setTranscription(null)
    setError(null)
  }, [])

  return {
    isRecording,
    isProcessing,
    transcription,
    error,
    startRecording,
    stopRecording,
    toggleRecording,
    clearTranscription,
  }
}
