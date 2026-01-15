import { useRef, useEffect, useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { useWebRTC } from "@/hooks/useWebRTC"
import { useVoiceRecording } from "@/hooks/useVoiceRecording"
import { useSignPlayback } from "@/hooks/useSignPlayback"
import { AvatarViewer } from "@/components/AvatarViewer"
import { GlossDisplay } from "@/components/GlossDisplay"
import type { AvatarController } from "@/lib/avatar-controller"
import type { TranslationResult } from "@/hooks/useTranslation"
import { Video, VideoOff, Mic, MicOff, PhoneOff, Phone, Copy, Check, Languages, Play, Square, Loader2 } from "lucide-react"

export function VideoCall() {
  const [roomId, setRoomId] = useState("")
  const [userId] = useState(() => `user_${Math.random().toString(36).substr(2, 9)}`)
  const [inCall, setInCall] = useState(false)
  const [videoEnabled, setVideoEnabled] = useState(true)
  const [audioEnabled, setAudioEnabled] = useState(true)
  const [copied, setCopied] = useState(false)
  const [avatar, setAvatar] = useState<AvatarController | null>(null)
  const [translationResult, setTranslationResult] = useState<TranslationResult | null>(null)
  const [transcribedText, setTranscribedText] = useState<string>("")
  const [translationSource, setTranslationSource] = useState<"local" | "remote">("local")
  const [remoteSender, setRemoteSender] = useState<string>("")
  
  const localVideoRef = useRef<HTMLVideoElement>(null)
  const remoteVideoRef = useRef<HTMLVideoElement>(null)
  const sendTranslationRef = useRef<((translation: TranslationResult) => void) | null>(null)

  const { isPlaying, currentToken, currentGifUrl, playSequence, stopPlayback } = useSignPlayback({ avatar })

  // Handle receiving translation from remote peer
  const handleRemoteTranslation = useCallback((result: TranslationResult, senderName: string) => {
    console.log("[VideoCall] Received remote translation from:", senderName)
    stopPlayback()
    setTranslationResult(result)
    setTranslationSource("remote")
    setRemoteSender(senderName)
    if (result.transcription) {
      setTranscribedText(result.transcription)
    }
    if (result.plan.length > 0) {
      playSequence(result.plan)
    }
  }, [stopPlayback, playSequence])

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
    sendTranslation,
  } = useWebRTC({ roomId, userId, onRemoteTranslation: handleRemoteTranslation })

  // Keep ref updated for use in voice result handler
  useEffect(() => {
    sendTranslationRef.current = sendTranslation
  }, [sendTranslation])

  const handleVoiceResult = useCallback((result: TranslationResult) => {
    stopPlayback()
    setTranslationResult(result)
    setTranslationSource("local")
    setRemoteSender("")
    if (result.transcription) {
      setTranscribedText(result.transcription)
    }
    if (result.plan.length > 0) {
      playSequence(result.plan)
    }
    // Send translation to remote peer
    if (sendTranslationRef.current) {
      sendTranslationRef.current(result)
    }
  }, [stopPlayback, playSequence])

  const { 
    isRecording, 
    isProcessing, 
    error: voiceError,
    startRecording, 
    stopRecording 
  } = useVoiceRecording({
    onResult: handleVoiceResult,
    onTranscription: (text) => setTranscribedText(text),
  })

  const handleAvatarReady = useCallback((controller: AvatarController) => {
    setAvatar(controller)
  }, [])

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

  const handleTranscribeToggle = () => {
    if (isRecording) {
      stopRecording()
    } else {
      setTranscribedText("")
      setTranslationResult(null)
      stopPlayback()
      startRecording()
    }
  }

  const handleReplay = () => {
    if (translationResult && translationResult.plan.length > 0) {
      playSequence(translationResult.plan)
    }
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

      {/* Error messages */}
      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}
      {voiceError && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {voiceError}
        </div>
      )}

      {/* Main Content Grid - Video + Sign Language Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_400px] gap-4">
        {/* Left Side - Video Grid */}
        <div className="flex flex-col gap-4">
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

          {/* Video Controls */}
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

        {/* Right Side - Sign Language Panel */}
        <Card className="p-4 flex flex-col gap-4 bg-[var(--bg-card)] border-[var(--border-soft)]">
          {/* Panel Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Languages className="w-5 h-5 text-[var(--accent-terracotta)]" />
              <h3 className="font-semibold text-[var(--text-primary)]">Sign Language</h3>
            </div>
            <div className="flex items-center gap-2">
              {translationResult && (
                <Badge 
                  variant={translationSource === "remote" ? "default" : "accent"} 
                  className={`text-xs ${translationSource === "remote" ? "bg-blue-500" : ""}`}
                >
                  {translationSource === "remote" ? "Remote" : "Local"}
                </Badge>
              )}
              {isPlaying && (
                <Badge variant="accent" className="text-xs">
                  Playing
                </Badge>
              )}
            </div>
          </div>

          {/* Transcribe Button */}
          <Button
            onClick={handleTranscribeToggle}
            disabled={isProcessing}
            className={`w-full ${
              isRecording 
                ? "bg-red-500 hover:bg-red-600 text-white" 
                : "bg-[var(--accent-terracotta)] hover:bg-[var(--accent-warm)] text-white"
            }`}
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Processing...
              </>
            ) : isRecording ? (
              <>
                <Square className="w-4 h-4 mr-2" />
                Stop Recording
              </>
            ) : (
              <>
                <Mic className="w-4 h-4 mr-2" />
                Transcribe Speech
              </>
            )}
          </Button>

          {/* Recording Indicator */}
          {isRecording && (
            <div className="flex items-center justify-center gap-2 py-2">
              <span className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
              <span className="text-sm text-red-600 font-medium">Recording...</span>
            </div>
          )}

          {/* Transcribed Text */}
          {transcribedText && (
            <div className={`p-3 rounded-lg border ${
              translationSource === "remote" 
                ? "bg-blue-50 border-blue-200" 
                : "bg-[var(--bg-cream)] border-[var(--border-soft)]"
            }`}>
              <div className="flex items-center justify-between mb-1">
                <p className="text-xs text-[var(--text-muted)]">Transcribed:</p>
                <Badge 
                  variant={translationSource === "remote" ? "default" : "accent"} 
                  className={`text-[0.6rem] ${translationSource === "remote" ? "bg-blue-500" : ""}`}
                >
                  {translationSource === "remote" ? `From: ${remoteSender.slice(-4)}` : "You"}
                </Badge>
              </div>
              <p className="text-sm text-[var(--text-primary)]">{transcribedText}</p>
            </div>
          )}

          {/* Sign Sequence Display */}
          {translationResult && translationResult.gloss && translationResult.gloss.length > 0 && (
            <div className="p-3 rounded-lg border border-[var(--border-soft)] bg-[var(--bg-card)]">
              <div className="text-xs font-semibold tracking-[0.1em] uppercase text-[var(--text-muted)] mb-2">
                Sign Sequence
              </div>
              <GlossDisplay 
                gloss={translationResult.gloss} 
                activeToken={currentToken}
                unmatched={translationResult.unmatched || []}
              />
            </div>
          )}

          {/* GIF Display */}
          <div className="flex-1 min-h-[180px] flex flex-col rounded-lg border border-[var(--border-soft)] bg-[var(--bg-cream)] overflow-hidden">
            <div className="px-3 py-2 border-b border-[var(--border-soft)] bg-[var(--bg-card)] flex items-center justify-between">
              <span className="text-xs font-semibold tracking-wide uppercase text-[var(--text-muted)]">
                Reference GIF
              </span>
              {currentToken && (
                <Badge variant="accent" className="text-[0.6rem] px-2 py-0.5">
                  {currentToken}
                </Badge>
              )}
            </div>
            <div className="flex-1 flex items-center justify-center bg-[var(--bg-input)] p-2">
              {currentGifUrl ? (
                <img 
                  src={currentGifUrl} 
                  alt="Sign Language" 
                  className="max-h-full max-w-full object-contain rounded"
                />
              ) : (
                <div className="flex flex-col items-center gap-2 text-[var(--text-light)]">
                  <Play className="w-8 h-8 opacity-40" />
                  <span className="text-xs text-center">
                    {translationResult ? "Ready to play" : "Hit transcribe to see signs"}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Avatar Display */}
          <div className="flex-1 min-h-[200px] flex flex-col rounded-lg border border-[var(--border-soft)] bg-[var(--bg-cream)] overflow-hidden">
            <div className="px-3 py-2 border-b border-[var(--border-soft)] bg-[var(--bg-card)] flex items-center justify-between">
              <span className="text-xs font-semibold tracking-wide uppercase text-[var(--text-muted)]">
                Avatar
              </span>
              {isPlaying && (
                <Badge variant="accent" className="text-[0.6rem] px-2 py-0.5">
                  Live
                </Badge>
              )}
            </div>
            <AvatarViewer onReady={handleAvatarReady} />
          </div>

          {/* Replay Button */}
          {translationResult && translationResult.plan.length > 0 && (
            <Button
              variant="outline"
              onClick={handleReplay}
              disabled={isPlaying}
              className="w-full"
            >
              <Play className="w-4 h-4 mr-2" />
              Replay Signs
            </Button>
          )}
        </Card>
      </div>
    </div>
  )
}
