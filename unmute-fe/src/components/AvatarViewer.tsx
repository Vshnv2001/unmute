import { useEffect, useRef, useCallback } from "react"
import { AvatarController } from "@/lib/avatar-controller"

interface AvatarViewerProps {
  onReady?: (controller: AvatarController) => void
}

export function AvatarViewer({ onReady }: AvatarViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<AvatarController | null>(null)

  const handleResize = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.resize()
    }
  }, [])

  useEffect(() => {
    if (!containerRef.current) return

    // Create controller
    const controller = new AvatarController(containerRef.current)
    controllerRef.current = controller

    // Initial resize - do it multiple times to ensure container has dimensions
    controller.resize()
    setTimeout(() => controller.resize(), 100)
    setTimeout(() => controller.resize(), 500)

    // Notify parent
    onReady?.(controller)

    // Set up resize observer
    const resizeObserver = new ResizeObserver(() => {
      handleResize()
    })
    resizeObserver.observe(containerRef.current)

    // Cleanup
    return () => {
      resizeObserver.disconnect()
      controller.dispose()
      controllerRef.current = null
    }
  }, [onReady, handleResize])

  return (
    <div 
      ref={containerRef} 
      className="flex-1 min-h-0 bg-bg-input w-full"
      style={{ minHeight: '250px' }}
    />
  )
}
