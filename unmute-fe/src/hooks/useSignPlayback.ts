import { useState, useCallback, useRef } from "react"
import type { PlanItem, LandmarksData, HandFrame } from "./useTranslation"
import type { AvatarController, PoseFrame } from "@/lib/avatar-controller"

const API_BASE = "http://127.0.0.1:8000"
const LANDMARKS_URL = `${API_BASE}/api/sign`

interface UseSignPlaybackOptions {
  avatar: AvatarController | null
}

export function useSignPlayback({ avatar }: UseSignPlaybackOptions) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentToken, setCurrentToken] = useState<string | undefined>()
  const [currentGifUrl, setCurrentGifUrl] = useState<string | undefined>()
  
  const playbackIdRef = useRef(0)

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

  const playSingleSign = useCallback(async (
    item: PlanItem, 
    playbackId: number
  ): Promise<boolean> => {
    if (playbackId !== playbackIdRef.current) {
      return false
    }

    if (!item.sign_name) {
      return false
    }

    console.log(`Playing sign: ${item.sign_name}`)
    setCurrentToken(item.token)

    // Show GIF
    if (item.assets?.gif) {
      const gifUrl = `${API_BASE}${item.assets.gif}?t=${Date.now()}`
      setCurrentGifUrl(gifUrl)
    }

    const SIGN_DURATION = 3000 // 3 seconds per sign
    const startTime = Date.now()

    try {
      const data = await fetchLandmarks(item.sign_name)
      console.log(`[Landmarks Response] ${item.sign_name}:`, data)
      
      // Get frames from various possible formats
      let frames: PoseFrame[] | null = null
      
      // Check for pose_frames (can be either full pose or hand data)
      if (data?.pose_frames && Array.isArray(data.pose_frames) && data.pose_frames.length > 0) {
        frames = data.pose_frames as PoseFrame[]
        // Check if it's hand data (has left_hand/right_hand) or pose data (has pose)
        const isHandData = frames[0] && ('left_hand' in frames[0] || 'right_hand' in frames[0])
        console.log(`Playing ${isHandData ? 'hand' : 'pose'} skeleton: ${item.token} (${frames.length} frames)`)
      } 
      // Check for hand_frames (legacy format with left/right)
      else if (data?.hand_frames && Array.isArray(data.hand_frames) && data.hand_frames.length > 0) {
        // Convert legacy hand_frames format to new format
        frames = data.hand_frames.map((f: HandFrame) => ({
          left_hand: f.left,
          right_hand: f.right
        })) as PoseFrame[]
        console.log(`Playing hand skeleton (converted): ${item.token} (${frames.length} frames)`)
      } 
      // Check for frames (generic format)
      else if (data?.frames && Array.isArray(data.frames) && data.frames.length > 0) {
        frames = data.frames as PoseFrame[]
        console.log(`Playing skeleton (legacy format): ${item.token} (${frames.length} frames)`)
      } else {
        console.log(`No frames found for ${item.token}`)
      }

      // Play skeleton animation (if available)
      if (frames && avatar) {
        try {
          // Calculate FPS to fit animation into 3 seconds
          const fps = Math.max(10, frames.length / (SIGN_DURATION / 1000))
          await avatar.playSequence(frames, fps)
        } catch (e) {
          console.error(`Error playing skeleton for ${item.token}:`, e)
        }
      }

      // Ensure we wait the full 3 seconds
      const elapsed = Date.now() - startTime
      const remaining = SIGN_DURATION - elapsed
      if (remaining > 0) {
        await new Promise(r => setTimeout(r, remaining))
      }

      console.log(`Finished: ${item.token} (${Date.now() - startTime}ms)`)
      setCurrentGifUrl(undefined)
      return true
    } catch (e) {
      console.error("Fetch error", e)
      // Still wait remaining time on error
      const elapsed = Date.now() - startTime
      const remaining = SIGN_DURATION - elapsed
      if (remaining > 0) {
        await new Promise(r => setTimeout(r, remaining))
      }
      setCurrentGifUrl(undefined)
      return false
    }
  }, [avatar, fetchLandmarks])

  const playSequence = useCallback(async (plan: PlanItem[]) => {
    if (isPlaying) return
    
    setIsPlaying(true)
    playbackIdRef.current++
    const currentPlaybackId = playbackIdRef.current

    console.log("Plan received:", plan.map(p => p.sign_name || p.token))

    // Filter to unique signs only
    const uniquePlan: PlanItem[] = []
    let lastSignName: string | null = null
    for (const item of plan) {
      if (item.type === "sign" && item.sign_name !== lastSignName) {
        uniquePlan.push(item)
        lastSignName = item.sign_name ?? null
      } else if (item.type !== "sign") {
        uniquePlan.push(item)
        lastSignName = null
      }
    }

    console.log("Unique plan:", uniquePlan.map(p => p.sign_name || p.token))

    // Detect single sign vs multiple signs
    const signs = uniquePlan.filter(item => item.type === "sign" && item.sign_name)
    const isSingleSign = signs.length === 1

    if (isSingleSign) {
      // Single sign mode: loop until new request
      const singleSign = signs[0]
      console.log(`Single sign detected: ${singleSign.sign_name}, starting loop`)

      while (currentPlaybackId === playbackIdRef.current) {
        if (currentPlaybackId !== playbackIdRef.current) break
        await playSingleSign(singleSign, currentPlaybackId)
        if (currentPlaybackId === playbackIdRef.current) {
          await new Promise(r => setTimeout(r, 300))
        }
      }

      console.log("Single sign loop ended (new request received)")
    } else {
      // Multiple signs: sequential playback
      for (let i = 0; i < uniquePlan.length; i++) {
        const item = uniquePlan[i]

        if (item.type === "sign" && item.sign_name) {
          console.log(`Starting sign ${i + 1}/${uniquePlan.length}: ${item.sign_name}`)
          await playSingleSign(item, currentPlaybackId)

          if (i < uniquePlan.length - 1) {
            await new Promise(r => setTimeout(r, 300))
          }
        } else {
          console.log(`Skipping non-sign: ${item.token}`)
          await new Promise(r => setTimeout(r, 500))
        }
      }
    }

    // Cleanup
    setCurrentToken(undefined)
    setCurrentGifUrl(undefined)
    setIsPlaying(false)
    console.log("Sequence complete")
  }, [isPlaying, playSingleSign])

  const stopPlayback = useCallback(() => {
    playbackIdRef.current++
    setCurrentToken(undefined)
    setCurrentGifUrl(undefined)
    setIsPlaying(false)
  }, [])

  return {
    isPlaying,
    currentToken,
    currentGifUrl,
    playSequence,
    stopPlayback,
  }
}
