import { useState, useCallback, useRef, useEffect } from "react"
import type { PlanItem, LandmarksData, HandFrame } from "./useTranslation"
import type { AvatarController, PoseFrame } from "@/lib/avatar-controller"

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
const LANDMARKS_URL = `${API_BASE}/api/sign`

interface UseSignPlaybackOptions {
  avatar: AvatarController | null
}

export function useSignPlayback({ avatar }: UseSignPlaybackOptions) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentToken, setCurrentToken] = useState<string | undefined>()
  const [currentGifUrl, setCurrentGifUrl] = useState<string | undefined>()
  
  const playbackIdRef = useRef(0)
  const avatarRef = useRef<AvatarController | null>(null)
  
  // Keep avatar ref in sync with prop
  useEffect(() => {
    avatarRef.current = avatar
    console.log(`[Playback] Avatar ref updated:`, { hasAvatar: !!avatar, avatarType: avatar?.constructor?.name })
  }, [avatar])

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
      // Check if URL is already absolute (starts with http:// or https://)
      const isAbsoluteUrl = item.assets.gif.startsWith('http://') || item.assets.gif.startsWith('https://')
      const gifUrl = isAbsoluteUrl 
        ? `${item.assets.gif}?t=${Date.now()}`
        : `${API_BASE}${item.assets.gif}?t=${Date.now()}`
      setCurrentGifUrl(gifUrl)
    }

    const SIGN_DURATION = 3000 // 3 seconds per sign
    const startTime = Date.now()

    try {
      const data = await fetchLandmarks(item.sign_name)
      console.log(`[Landmarks Response] ${item.sign_name}:`, {
        hasPoseFrames: !!data?.pose_frames,
        poseFramesLength: data?.pose_frames?.length,
        firstFrame: data?.pose_frames?.[0],
        L_orig: data?.L_orig,
        L_max: data?.L_max
      })
      
      // Get frames from various possible formats
      let frames: PoseFrame[] | null = null
      
      // Check for pose_frames (full-body pose data from backend)
      if (data?.pose_frames && Array.isArray(data.pose_frames) && data.pose_frames.length > 0) {
        frames = data.pose_frames as PoseFrame[]
        // Check if it's pose data (has pose property)
        const isPoseData = frames[0] && 'pose' in frames[0] && Array.isArray(frames[0].pose)
        if (isPoseData) {
          console.log(`Playing full-body pose skeleton: ${item.token} (${frames.length} frames)`)
          console.log(`[Debug] First frame structure:`, {
            hasPose: !!frames[0].pose,
            poseLength: frames[0].pose?.length,
            firstPoint: frames[0].pose?.[0],
            samplePoints: frames[0].pose?.slice(0, 3)
          })
          console.log(`[Debug] Avatar state check:`, {
            hasAvatar: !!avatar,
            avatarType: typeof avatar,
            avatarConstructor: avatar?.constructor?.name
          })
        } else {
          // Legacy hand data in pose_frames format
          const isHandData = frames[0] && ('left_hand' in frames[0] || 'right_hand' in frames[0])
          console.log(`Playing ${isHandData ? 'hand' : 'pose'} skeleton: ${item.token} (${frames.length} frames)`)
        }
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
      // Use ref to get latest avatar value (avoids stale closure issues)
      const currentAvatar = avatarRef.current
      console.log(`[Playback] ===== CHECKING CONDITIONS =====`)
      console.log(`[Playback] Checking conditions:`, {
        hasFrames: !!frames,
        framesLength: frames?.length,
        hasAvatar: !!currentAvatar,
        hasAvatarProp: !!avatar,
        avatarType: currentAvatar?.constructor?.name
      })
      
      // Wait for avatar if not ready yet
      if (frames && !currentAvatar) {
        console.log(`[Playback] Avatar not ready, waiting up to 2 seconds...`)
        const maxWait = 2000
        const checkInterval = 50
        const startWait = Date.now()
        
        while (!avatarRef.current && (Date.now() - startWait) < maxWait) {
          await new Promise(resolve => setTimeout(resolve, checkInterval))
        }
        
        if (avatarRef.current) {
          console.log(`[Playback] Avatar became ready after ${Date.now() - startWait}ms`)
        } else {
          console.warn(`[Playback] Avatar still not ready after ${maxWait}ms`)
        }
      }
      
      if (frames && avatarRef.current) {
        try {
          console.log(`[Playback] ===== STARTING ANIMATION =====`)
          console.log(`[Playback] Preparing to play ${frames.length} frames for ${item.token}`)
          // Calculate FPS to fit animation into 3 seconds
          const fps = Math.max(10, frames.length / (SIGN_DURATION / 1000))
          console.log(`[Playback] Calculated FPS: ${fps} for ${frames.length} frames`)
          console.log(`[Playback] Calling avatar.playSequence with ${frames.length} frames`)
          const result = await avatarRef.current.playSequence(frames, fps)
          console.log(`[Playback] Animation completed, result: ${result}`)
        } catch (e) {
          console.error(`[Playback] Error playing skeleton for ${item.token}:`, e)
          console.error(e)
        }
      } else {
        console.warn(`[Playback] Cannot play skeleton - missing requirements:`, {
          hasFrames: !!frames,
          framesLength: frames?.length,
          hasAvatar: !!avatarRef.current,
          framesType: frames?.constructor?.name,
          avatarType: avatarRef.current?.constructor?.name
        })
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
