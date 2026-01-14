import { useRef, useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { useWebRTC } from "@/hooks/useWebRTC"
import { Video, VideoOff, Mic, MicOff, PhoneOff, Phone, Copy, Check } from "lucide-react"

export function VideoCall() {
  const [roomId, setRoomId] = useState("")
  const [userId] = useState(() => `user_${Math.random().toString(36).substr(2, 9)}`)
  const [inCall, setInCall] = useState(false)
  const [videoEnabled, setVideoEnabled] = useState(true)
  const [audioEnabled, setAudioEnabled] = useState(true)
  const [copied, setCopied] = useState(false)
  
  const localVideoRef = useRef<HTMLVideoElement>(null)
  const remoteVideoRef = useRef<HTMLVideoElement>(null)
  
  const { 
    isConnected, 
    localStream, 
    remoteStream, 
    peerCount, 
    error,
    connect, 
    disconnect,
    toggleVideo,
    toggleAudio,
  } = useWebRTC({ roomId, userId })

  // Attach local stream to video element
  useEffect(() => {
    if (localVideoRef.current && localStream) {
      localVideoRef.current.srcObject = localStream
    }
  }, [localStream])

  // Attach remote stream to video element
  useEffect(() => {
    if (remoteVideoRef.current && remoteStream) {
      remoteVideoRef.current.srcObject = remoteStream
    }
  }, [remoteStream])

  const handleJoinCall = async () => {
    if (!roomId.trim()) {
      alert("Please enter a room ID")
      return
    }
    setInCall(true)
    await connect()
  }

  const handleLeaveCall = () => {
    disconnect()
    setInCall(false)
    setVideoEnabled(true)
    setAudioEnabled(true)
  }

  const handleToggleVideo = () => {
    toggleVideo()
    setVideoEnabled(!videoEnabled)
  }

  const handleToggleAudio = () => {
    toggleAudio()
    setAudioEnabled(!audioEnabled)
  }

  const copyRoomId = async () => {
    await navigator.clipboard.writeText(roomId)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const generateRoomId = () => {
    const id = Math.random().toString(36).substr(2, 8)
    setRoomId(id)
  }

  if (!inCall) {
    return (
      <Card className="p-8 max-w-md mx-auto bg-[var(--bg-card)] border-[var(--border-soft)]">
        <h2 className="text-2xl font-semibold mb-2 text-center text-[var(--text-primary)]">
          Join Video Call
        </h2>
        <p className="text-sm text-[var(--text-muted)] text-center mb-6">
          Enter a room ID to join, or generate a new one to start a call
        </p>
        
        <div className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Enter Room ID"
              value={roomId}
              onChange={(e) => setRoomId(e.target.value)}
              className="text-center"
            />
            <Button 
              variant="outline" 
              onClick={generateRoomId}
              className="shrink-0"
            >
              Generate
            </Button>
          </div>
          
          <Button 
            onClick={handleJoinCall} 
            className="w-full bg-[var(--accent-terracotta)] hover:bg-[var(--accent-warm)] text-white"
            disabled={!roomId.trim()}
          >
            <Phone className="w-4 h-4 mr-2" />
            Join Room
          </Button>
          
          <p className="text-xs text-[var(--text-muted)] text-center">
            Share the room ID with others to have them join the same call
          </p>
        </div>
      </Card>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Status Bar */}
      <Card className="p-3 bg-[var(--bg-card)] border-[var(--border-soft)]">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <span className="text-sm text-[var(--text-secondary)]">
              Room: <span className="font-mono font-semibold">{roomId}</span>
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={copyRoomId}
              className="h-7 px-2"
            >
              {copied ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </Button>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm text-[var(--text-muted)]">
              {peerCount} user{peerCount !== 1 ? "s" : ""}
            </span>
            <span className={`flex items-center gap-1.5 text-sm ${isConnected ? "text-green-600" : "text-amber-500"}`}>
              <span className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-500" : "bg-amber-400"}`} />
              {isConnected ? "Connected" : "Waiting..."}
            </span>
          </div>
        </div>
      </Card>

      {/* Error message */}
      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Video Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Local Video */}
        <Card className="relative overflow-hidden aspect-video bg-[#1a1a1a] border-[var(--border-soft)]">
          <video
            ref={localVideoRef}
            autoPlay
            muted
            playsInline
            className={`w-full h-full object-cover ${!videoEnabled ? "hidden" : ""}`}
          />
          {!videoEnabled && (
            <div className="absolute inset-0 flex items-center justify-center bg-[#2a2a2a]">
              <div className="w-20 h-20 rounded-full bg-[var(--accent-terracotta)] flex items-center justify-center text-white text-2xl font-semibold">
                {userId.slice(-2).toUpperCase()}
              </div>
            </div>
          )}
          <div className="absolute bottom-2 left-2 flex items-center gap-2">
            <span className="bg-black/60 text-white px-2 py-1 rounded text-sm">
              You
            </span>
            {!audioEnabled && (
              <span className="bg-red-500/80 text-white p-1 rounded">
                <MicOff className="w-3 h-3" />
              </span>
            )}
          </div>
        </Card>

        {/* Remote Video */}
        <Card className="relative overflow-hidden aspect-video bg-[#1a1a1a] border-[var(--border-soft)]">
          {remoteStream ? (
            <video
              ref={remoteVideoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-[var(--text-muted)] gap-3">
              <div className="w-16 h-16 rounded-full border-2 border-dashed border-[var(--border-warm)] flex items-center justify-center">
                <Video className="w-6 h-6 opacity-50" />
              </div>
              <span className="text-sm">Waiting for peer to join...</span>
              <span className="text-xs opacity-60">Share the room ID above</span>
            </div>
          )}
          {remoteStream && (
            <span className="absolute bottom-2 left-2 bg-black/60 text-white px-2 py-1 rounded text-sm">
              Remote
            </span>
          )}
        </Card>
      </div>

      {/* Controls */}
      <Card className="p-4 bg-[var(--bg-card)] border-[var(--border-soft)]">
        <div className="flex justify-center gap-3">
          <Button 
            variant={videoEnabled ? "outline" : "secondary"}
            size="lg"
            onClick={handleToggleVideo}
            className={`rounded-full w-14 h-14 ${!videoEnabled ? "bg-red-100 hover:bg-red-200 text-red-600" : ""}`}
          >
            {videoEnabled ? <Video className="w-5 h-5" /> : <VideoOff className="w-5 h-5" />}
          </Button>
          
          <Button 
            variant={audioEnabled ? "outline" : "secondary"}
            size="lg"
            onClick={handleToggleAudio}
            className={`rounded-full w-14 h-14 ${!audioEnabled ? "bg-red-100 hover:bg-red-200 text-red-600" : ""}`}
          >
            {audioEnabled ? <Mic className="w-5 h-5" /> : <MicOff className="w-5 h-5" />}
          </Button>
          
          <Button 
            variant="destructive" 
            size="lg"
            onClick={handleLeaveCall}
            className="rounded-full w-14 h-14 bg-red-500 hover:bg-red-600"
          >
            <PhoneOff className="w-5 h-5" />
          </Button>
        </div>
      </Card>
    </div>
  )
}
