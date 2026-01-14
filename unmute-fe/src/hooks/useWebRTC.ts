import { useState, useRef, useCallback, useEffect } from "react"

const WS_BASE = "ws://127.0.0.1:8000"

interface UseWebRTCOptions {
  roomId: string
  userId: string
}

export function useWebRTC({ roomId, userId }: UseWebRTCOptions) {
  const [isConnected, setIsConnected] = useState(false)
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null)
  const [localStream, setLocalStream] = useState<MediaStream | null>(null)
  const [peerCount, setPeerCount] = useState(0)
  const [error, setError] = useState<string | null>(null)
  
  const wsRef = useRef<WebSocket | null>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const localStreamRef = useRef<MediaStream | null>(null)

  const createPeerConnection = useCallback(() => {
    const config: RTCConfiguration = {
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" },
        { urls: "stun:stun1.l.google.com:19302" },
      ],
    }

    const pc = new RTCPeerConnection(config)

    pc.onicecandidate = (event) => {
      if (event.candidate && wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: "ice-candidate",
          candidate: event.candidate,
        }))
      }
    }

    pc.ontrack = (event) => {
      console.log("[WebRTC] Received remote track:", event.streams[0])
      setRemoteStream(event.streams[0])
    }

    pc.onconnectionstatechange = () => {
      console.log("[WebRTC] Connection state:", pc.connectionState)
      if (pc.connectionState === "connected") {
        setIsConnected(true)
      } else if (pc.connectionState === "disconnected" || pc.connectionState === "failed") {
        setIsConnected(false)
        setRemoteStream(null)
      }
    }

    pc.oniceconnectionstatechange = () => {
      console.log("[WebRTC] ICE connection state:", pc.iceConnectionState)
    }

    // Add local tracks if available
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => {
        console.log("[WebRTC] Adding local track:", track.kind)
        pc.addTrack(track, localStreamRef.current!)
      })
    }

    pcRef.current = pc
    return pc
  }, [])

  const startLocalStream = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      })
      localStreamRef.current = stream
      setLocalStream(stream)
      console.log("[WebRTC] Local stream started")
      return stream
    } catch (err) {
      console.error("[WebRTC] Error accessing media devices:", err)
      setError("Could not access camera/microphone. Please check permissions.")
      return null
    }
  }, [])

  const connect = useCallback(async () => {
    setError(null)
    
    // Get local media first
    const stream = await startLocalStream()
    if (!stream) return

    // Connect WebSocket
    const wsUrl = `${WS_BASE}/ws/room/${roomId}/${userId}`
    console.log("[WebRTC] Connecting to:", wsUrl)
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log("[WebRTC] WebSocket connected")
    }

    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data)
      console.log("[WebRTC] Received message:", data.type)

      switch (data.type) {
        case "room_info":
          setPeerCount(data.user_count)
          break

        case "user_joined":
          setPeerCount(data.user_count)
          // New user joined - we are the initiator, create offer
          console.log("[WebRTC] New user joined, creating offer for:", data.user_id)
          const pc = createPeerConnection()
          try {
            const offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            ws.send(JSON.stringify({
              type: "offer",
              offer: pc.localDescription,
              target_id: data.user_id,
            }))
          } catch (err) {
            console.error("[WebRTC] Error creating offer:", err)
          }
          break

        case "user_left":
          console.log("[WebRTC] User left:", data.user_id)
          setRemoteStream(null)
          setIsConnected(false)
          pcRef.current?.close()
          pcRef.current = null
          break

        case "offer":
          // Received offer - create answer
          console.log("[WebRTC] Received offer from:", data.sender_id)
          const answerPc = pcRef.current || createPeerConnection()
          try {
            await answerPc.setRemoteDescription(new RTCSessionDescription(data.offer))
            const answer = await answerPc.createAnswer()
            await answerPc.setLocalDescription(answer)
            ws.send(JSON.stringify({
              type: "answer",
              answer: answerPc.localDescription,
              target_id: data.sender_id,
            }))
          } catch (err) {
            console.error("[WebRTC] Error handling offer:", err)
          }
          break

        case "answer":
          // Received answer
          console.log("[WebRTC] Received answer from:", data.sender_id)
          if (pcRef.current) {
            try {
              await pcRef.current.setRemoteDescription(new RTCSessionDescription(data.answer))
            } catch (err) {
              console.error("[WebRTC] Error setting remote description:", err)
            }
          }
          break

        case "ice-candidate":
          // Received ICE candidate
          console.log("[WebRTC] Received ICE candidate from:", data.sender_id)
          if (pcRef.current && data.candidate) {
            try {
              await pcRef.current.addIceCandidate(new RTCIceCandidate(data.candidate))
            } catch (err) {
              console.error("[WebRTC] Error adding ICE candidate:", err)
            }
          }
          break
      }
    }

    ws.onerror = (err) => {
      console.error("[WebRTC] WebSocket error:", err)
      setError("Connection error. Please try again.")
    }

    ws.onclose = () => {
      console.log("[WebRTC] WebSocket closed")
    }
  }, [roomId, userId, createPeerConnection, startLocalStream])

  const disconnect = useCallback(() => {
    console.log("[WebRTC] Disconnecting...")
    
    // Close peer connection
    pcRef.current?.close()
    pcRef.current = null

    // Close WebSocket
    wsRef.current?.close()
    wsRef.current = null

    // Stop local stream
    localStreamRef.current?.getTracks().forEach((track) => {
      track.stop()
      console.log("[WebRTC] Stopped track:", track.kind)
    })
    localStreamRef.current = null

    setLocalStream(null)
    setRemoteStream(null)
    setIsConnected(false)
    setPeerCount(0)
  }, [])

  const toggleVideo = useCallback(() => {
    if (localStreamRef.current) {
      localStreamRef.current.getVideoTracks().forEach((track) => {
        track.enabled = !track.enabled
        console.log("[WebRTC] Video track enabled:", track.enabled)
      })
    }
  }, [])

  const toggleAudio = useCallback(() => {
    if (localStreamRef.current) {
      localStreamRef.current.getAudioTracks().forEach((track) => {
        track.enabled = !track.enabled
        console.log("[WebRTC] Audio track enabled:", track.enabled)
      })
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    isConnected,
    localStream,
    remoteStream,
    peerCount,
    error,
    connect,
    disconnect,
    toggleVideo,
    toggleAudio,
  }
}
