import { useState, useCallback } from "react"

const API_URL = "http://127.0.0.1:8000/api/translate"
const LANDMARKS_URL = "http://127.0.0.1:8000/api/sign"

export interface PlanItem {
  type: "sign" | "text"
  token: string
  sign_name?: string
  assets?: {
    gif?: string
  }
}

export interface TranslationResult {
  plan: PlanItem[]
  gloss: string[]
  unmatched?: string[]
  notes?: string
  transcription?: string
}

export interface LandmarksData {
  pose_frames?: PoseFrame[]
  hand_frames?: PoseFrame[]
  frames?: PoseFrame[]
}

export interface PoseFrame {
  pose?: number[][]
}

export function useTranslation() {
  const [result, setResult] = useState<TranslationResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const translate = useCallback(async (text: string): Promise<TranslationResult | null> => {
    setIsLoading(true)
    setError(null)

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })

      if (!res.ok) {
        throw new Error("Translation failed")
      }

      const data: TranslationResult = await res.json()
      setResult(data)
      return data
    } catch (e) {
      const message = e instanceof Error ? e.message : "Translation failed"
      setError(message)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const fetchLandmarks = useCallback(async (signName: string): Promise<LandmarksData | null> => {
    try {
      const resp = await fetch(`${LANDMARKS_URL}/${signName}/landmarks`)
      if (resp.ok) {
        return await resp.json()
      }
      return null
    } catch (e) {
      console.error("Error fetching landmarks:", e)
      return null
    }
  }, [])

  const clearResult = useCallback(() => {
    setResult(null)
    setError(null)
  }, [])

  return {
    result,
    setResult,
    isLoading,
    error,
    translate,
    fetchLandmarks,
    clearResult,
  }
}
