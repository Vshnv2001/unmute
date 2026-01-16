import * as THREE from "three"

export interface PoseFrame {
  pose?: number[][]
  left_hand?: number[][]
  right_hand?: number[][]
}

interface BoneConnection {
  line: THREE.Line
  start: number
  end: number
}

// MediaPipe Hand landmark connections (21 landmarks per hand)
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm
  [5, 9], [9, 13], [13, 17],
]

export class AvatarController {
  private container: HTMLElement
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private renderer: THREE.WebGLRenderer
  private joints: THREE.Mesh[] = []
  private bones: BoneConnection[] = []
  private connections: number[][] = []
  // Hand-specific meshes
  private leftHandJoints: THREE.Mesh[] = []
  private rightHandJoints: THREE.Mesh[] = []
  private leftHandBones: BoneConnection[] = []
  private rightHandBones: BoneConnection[] = []
  private animationFrameId: number | null = null

  constructor(container: HTMLElement) {
    console.log('[Avatar] Initializing AvatarController')
    this.container = container
    this.scene = new THREE.Scene()
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100)
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })

    this.initScene()
    this.createRig()
    this.createHandRig()
    this.animate()
    console.log('[Avatar] AvatarController initialized, canvas appended')
  }

  private initScene(): void {
    // Scene background - warm beige to match UI
    this.scene.background = new THREE.Color(0xfbf9f7)

    // Camera setup
    this.camera.position.set(0, 0, 2)
    this.camera.lookAt(0, 0, 0)

    // Renderer
    this.renderer.setSize(1, 1) // Start small, resize later
    this.container.appendChild(this.renderer.domElement)

    // Lights
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
    directionalLight.position.set(2, 2, 5)
    this.scene.add(directionalLight)
    this.scene.add(new THREE.AmbientLight(0x404040))
  }

  resize(): void {
    const width = this.container.clientWidth
    const height = this.container.clientHeight
    console.log(`[Avatar] Resizing: ${width}x${height}`)
    if (width > 0 && height > 0) {
      this.camera.aspect = width / height
      this.camera.updateProjectionMatrix()
      this.renderer.setSize(width, height)
      console.log(`[Avatar] Canvas resized to: ${width}x${height}`)
    } else {
      console.warn(`[Avatar] Container has no dimensions: ${width}x${height}`)
    }
  }

  private createRig(): void {
    // MediaPipe Pose Topology (Complete 33 landmarks connections)
    this.connections = [
      // Face - Left eye
      [0, 1], [1, 2], [2, 3], [3, 7],
      // Face - Right eye
      [0, 4], [4, 5], [5, 6], [6, 8],
      // Mouth
      [9, 10],
      // Torso
      [11, 12],  // Shoulders
      [11, 23], [12, 24],  // Shoulders to hips
      [23, 24],  // Hips
      // Left arm
      [11, 13], [13, 15],  // Shoulder -> Elbow -> Wrist
      [15, 17], [15, 19], [15, 21],  // Wrist to pinky, index, thumb
      [17, 19],  // Pinky to index
      // Right arm
      [12, 14], [14, 16],  // Shoulder -> Elbow -> Wrist
      [16, 18], [16, 20], [16, 22],  // Wrist to pinky, index, thumb
      [18, 20],  // Pinky to index
      // Left leg
      [23, 25], [25, 27],  // Hip -> Knee -> Ankle
      [27, 29], [29, 31],  // Ankle -> Heel -> Foot index
      [27, 31],  // Ankle to foot index
      // Right leg
      [24, 26], [26, 28],  // Hip -> Knee -> Ankle
      [28, 30], [30, 32],  // Ankle -> Heel -> Foot index
      [28, 32],  // Ankle to foot index
    ]

    // Create joint spheres (33 joints for MediaPipe Pose)
    const geometry = new THREE.SphereGeometry(0.015, 8, 8)
    const matHead = new THREE.MeshLambertMaterial({ color: 0x4488ff })
    const matUpperBody = new THREE.MeshLambertMaterial({ color: 0x00ff88 })
    const matJoint = new THREE.MeshLambertMaterial({ color: 0xffff44 })

    for (let i = 0; i < 33; i++) {
      let mat: THREE.MeshLambertMaterial
      if (i <= 10) {
        mat = matHead  // Head landmarks (0-10)
      } else if (i <= 28) {
        mat = matUpperBody  // Upper body and visible lower body
      } else {
        mat = matJoint  // Feet landmarks (29-32)
      }

      const mesh = new THREE.Mesh(geometry, mat)
      mesh.visible = false
      this.scene.add(mesh)
      this.joints.push(mesh)
    }

    // Create bone connections
    const lineMat = new THREE.LineBasicMaterial({ color: 0x000000 })
    this.connections.forEach((conn) => {
      const lineGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 0, 0),
      ])
      const line = new THREE.Line(lineGeom, lineMat)
      line.visible = false
      this.scene.add(line)
      this.bones.push({ line, start: conn[0], end: conn[1] })
    })
  }

  private createHandRig(): void {
    // Create hand joints (21 per hand)
    const geometry = new THREE.SphereGeometry(0.012, 8, 8)
    const leftHandMat = new THREE.MeshLambertMaterial({ color: 0x4488ff })  // Blue for left
    const rightHandMat = new THREE.MeshLambertMaterial({ color: 0xff8844 }) // Orange for right
    const leftLineMat = new THREE.LineBasicMaterial({ color: 0x2266cc })
    const rightLineMat = new THREE.LineBasicMaterial({ color: 0xcc6622 })

    // Left hand joints
    for (let i = 0; i < 21; i++) {
      const mesh = new THREE.Mesh(geometry, leftHandMat)
      mesh.visible = false
      this.scene.add(mesh)
      this.leftHandJoints.push(mesh)
    }

    // Right hand joints
    for (let i = 0; i < 21; i++) {
      const mesh = new THREE.Mesh(geometry, rightHandMat)
      mesh.visible = false
      this.scene.add(mesh)
      this.rightHandJoints.push(mesh)
    }

    // Left hand bones
    HAND_CONNECTIONS.forEach((conn) => {
      const lineGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 0, 0),
      ])
      const line = new THREE.Line(lineGeom, leftLineMat)
      line.visible = false
      this.scene.add(line)
      this.leftHandBones.push({ line, start: conn[0], end: conn[1] })
    })

    // Right hand bones
    HAND_CONNECTIONS.forEach((conn) => {
      const lineGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 0, 0),
      ])
      const line = new THREE.Line(lineGeom, rightLineMat)
      line.visible = false
      this.scene.add(line)
      this.rightHandBones.push({ line, start: conn[0], end: conn[1] })
    })
  }

  private animate = (): void => {
    this.animationFrameId = requestAnimationFrame(this.animate)
    this.renderer.render(this.scene, this.camera)
  }

  private hideAllPose(): void {
    this.joints.forEach((j) => (j.visible = false))
    this.bones.forEach((b) => (b.line.visible = false))
  }

  private hideAllHands(): void {
    this.leftHandJoints.forEach((j) => (j.visible = false))
    this.rightHandJoints.forEach((j) => (j.visible = false))
    this.leftHandBones.forEach((b) => (b.line.visible = false))
    this.rightHandBones.forEach((b) => (b.line.visible = false))
  }

  private updateHand(
    points: number[][],
    joints: THREE.Mesh[],
    bones: BoneConnection[],
    offsetX: number
  ): boolean {
    if (!points || !Array.isArray(points) || points.length !== 21) {
      joints.forEach((j) => (j.visible = false))
      bones.forEach((b) => (b.line.visible = false))
      return false
    }

    // Check if hand is present (has non-zero points)
    const isPresent = points.some(
      (p) =>
        p &&
        Array.isArray(p) &&
        p.length >= 3 &&
        !isNaN(p[0]) &&
        (p[0] !== 0 || p[1] !== 0 || p[2] !== 0)
    )

    if (!isPresent) {
      joints.forEach((j) => (j.visible = false))
      bones.forEach((b) => (b.line.visible = false))
      return false
    }

    // Data is normalized to 0-1 range
    // Center it by subtracting 0.5, then scale for visibility
    const scale = 2.0  // Scale factor for visibility
    const zScale = 0.5 // Z depth scale

    // Update joints
    for (let i = 0; i < 21; i++) {
      const p = points[i]
      const j = joints[i]

      if (
        !p ||
        !Array.isArray(p) ||
        p.length < 3 ||
        (p[0] === 0 && p[1] === 0 && p[2] === 0)
      ) {
        j.visible = false
        continue
      }

      // Normalize from [0,1] to [-0.5, 0.5] by subtracting 0.5, then scale
      j.position.set(
        (p[0] - 0.5) * scale + offsetX,
        -(p[1] - 0.5) * scale,  // Flip Y axis for correct orientation
        -(p[2] - 0.5) * zScale
      )
      j.visible = true
    }

    // Update bones
    bones.forEach((b) => {
      const jStart = joints[b.start]
      const jEnd = joints[b.end]

      if (!jStart.visible || !jEnd.visible) {
        b.line.visible = false
        return
      }

      const pStart = jStart.position
      const pEnd = jEnd.position
      const positions = (b.line.geometry as THREE.BufferGeometry).attributes
        .position.array as Float32Array

      positions[0] = pStart.x
      positions[1] = pStart.y
      positions[2] = pStart.z
      positions[3] = pEnd.x
      positions[4] = pEnd.y
      positions[5] = pEnd.z

      ;(b.line.geometry as THREE.BufferGeometry).attributes.position.needsUpdate = true
      b.line.visible = true
    })

    return true
  }

  updateFrame(frame: PoseFrame | null): boolean {
    if (!frame) {
      this.hideAllPose()
      this.hideAllHands()
      return false
    }

    // Check if this is hand data
    if (frame.left_hand || frame.right_hand) {
      this.hideAllPose() // Hide pose when showing hands
      
      let hasValidData = false
      
      // Update left hand (offset to the left)
      if (frame.left_hand) {
        const valid = this.updateHand(frame.left_hand, this.leftHandJoints, this.leftHandBones, -0.5)
        if (valid) hasValidData = true
      } else {
        this.leftHandJoints.forEach((j) => (j.visible = false))
        this.leftHandBones.forEach((b) => (b.line.visible = false))
      }
      
      // Update right hand (offset to the right)
      if (frame.right_hand) {
        const valid = this.updateHand(frame.right_hand, this.rightHandJoints, this.rightHandBones, 0.5)
        if (valid) hasValidData = true
      } else {
        this.rightHandJoints.forEach((j) => (j.visible = false))
        this.rightHandBones.forEach((b) => (b.line.visible = false))
      }
      
      return hasValidData
    }

    // Handle pose data (full-body pose with raw coordinates)
    this.hideAllHands() // Hide hands when showing pose
    
    const points = frame.pose
    console.log('[Avatar] updateFrame - pose data:', {
      hasPose: !!points,
      isArray: Array.isArray(points),
      length: points?.length,
      firstPoint: points?.[0],
      samplePoints: points?.slice(0, 3)
    })
    
    if (!points || !Array.isArray(points) || points.length !== 33) {
      console.warn('[Avatar] Invalid pose data:', {
        hasPoints: !!points,
        isArray: Array.isArray(points),
        length: points?.length
      })
      this.hideAllPose()
      return false
    }

    // Check if pose is present
    const isPresent = points.some(
      (p) =>
        p &&
        Array.isArray(p) &&
        p.length >= 3 &&
        !isNaN(p[0]) &&
        (p[0] !== 0 || p[1] !== 0 || p[2] !== 0)
    )

    if (!isPresent) {
      console.warn('[Avatar] No valid pose points found (all zeros or invalid)')
      this.hideAllPose()
      return false
    }
    
    console.log('[Avatar] Updating pose with', points.filter(p => p && Array.isArray(p) && p.length >= 3).length, 'valid points')

    // Use raw coordinates without normalization - just scale for visibility
    // MediaPipe coordinates are in normalized image space (0-1), so we scale them
    const scale = 1.5  // Scale factor for visibility (reduced from 2.0 to 1.5 = 25% smaller)
    const zScale = 0.75  // Z depth scale (reduced proportionally)

    // First, hide all joints
    this.joints.forEach((j) => (j.visible = false))

    // Update joints using raw coordinates
    for (let i = 0; i < 33; i++) {
      const p = points[i]
      const j = this.joints[i]

      if (
        !p ||
        !Array.isArray(p) ||
        p.length < 3 ||
        (p[0] === 0 && p[1] === 0 && p[2] === 0)
      ) {
        continue
      }

      // Use raw coordinates directly (no centering/normalization)
      // MediaPipe coordinates: x, y in [0, 1], z is depth
      // Center by subtracting 0.5 for x,y to center the pose, but use raw z
      const x = (p[0] - 0.5) * scale
      const y = -(p[1] - 0.5) * scale  // Flip Y for correct orientation
      const z = -p[2] * zScale
      
      j.position.set(x, y, z)
      j.visible = true
      
      // Debug first few joints
      if (i < 3) {
        console.log(`[Avatar] Joint ${i}:`, { x, y, z, raw: p })
      }
    }

    // Update bones
    this.bones.forEach((b) => {
      const jStart = this.joints[b.start]
      const jEnd = this.joints[b.end]

      if (!jStart.visible || !jEnd.visible) {
        b.line.visible = false
        return
      }

      const pStart = jStart.position
      const pEnd = jEnd.position
      const positions = (b.line.geometry as THREE.BufferGeometry).attributes
        .position.array as Float32Array

      positions[0] = pStart.x
      positions[1] = pStart.y
      positions[2] = pStart.z
      positions[3] = pEnd.x
      positions[4] = pEnd.y
      positions[5] = pEnd.z

      ;(b.line.geometry as THREE.BufferGeometry).attributes.position.needsUpdate = true
      b.line.visible = true
    })

    return true
  }

  private checkFrameHasData(frame: PoseFrame): boolean {
    // Check for hand data
    if (frame.left_hand || frame.right_hand) {
      const checkHand = (points: number[][] | undefined) => {
        return points &&
          Array.isArray(points) &&
          points.length === 21 &&
          points.some(
            (p) =>
              p &&
              Array.isArray(p) &&
              p.length >= 3 &&
              !isNaN(p[0]) &&
              (p[0] !== 0 || p[1] !== 0 || p[2] !== 0)
          )
      }
      return !!(checkHand(frame.left_hand) || checkHand(frame.right_hand))
    }
    
    // Check for pose data
    const points = frame?.pose
    return !!(
      points &&
      Array.isArray(points) &&
      points.length === 33 &&
      points.some(
        (p) =>
          p &&
          Array.isArray(p) &&
          p.length >= 3 &&
          !isNaN(p[0]) &&
          (p[0] !== 0 || p[1] !== 0 || p[2] !== 0)
      )
    )
  }

  async playSequence(frames: PoseFrame[], fps = 30): Promise<boolean> {
    console.log('[Avatar] playSequence called:', {
      framesLength: frames?.length,
      fps,
      firstFrame: frames?.[0]
    })
    
    if (!frames || frames.length === 0) {
      console.warn('[Avatar] No frames provided to playSequence')
      return false
    }

    const interval = 1000 / fps
    let hasValidFrames = false
    let consecutiveBlankFrames = 0
    const maxConsecutiveBlanks = 10
    const earlyCheckFrames = 5

    // Early check for blank sequence
    let earlyBlankCount = 0
    for (let i = 0; i < Math.min(earlyCheckFrames, frames.length); i++) {
      const frame = frames[i]
      if (!this.checkFrameHasData(frame)) {
        earlyBlankCount++
      }
    }

    if (earlyBlankCount === Math.min(earlyCheckFrames, frames.length)) {
      console.warn("[Avatar] Skeleton sequence appears to be entirely blank (detected early)")
      this.updateFrame(frames[0])
      return false
    }
    
    console.log(`[Avatar] Starting animation with ${frames.length} frames at ${fps} fps`)

    // Play through frames
    for (const frame of frames) {
      const isValid = this.updateFrame(frame)

      if (isValid) {
        hasValidFrames = true
        consecutiveBlankFrames = 0
      } else {
        consecutiveBlankFrames++
        if (hasValidFrames && consecutiveBlankFrames >= maxConsecutiveBlanks) {
          console.log("Skeleton rendering went blank, stopping sequence early")
          break
        }
      }

      await new Promise((r) => setTimeout(r, interval))
    }

    return hasValidFrames
  }

  dispose(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId)
    }

    // Clean up Three.js resources - Pose
    this.joints.forEach((j) => {
      j.geometry.dispose()
      ;(j.material as THREE.Material).dispose()
    })

    this.bones.forEach((b) => {
      b.line.geometry.dispose()
      ;(b.line.material as THREE.Material).dispose()
    })

    // Clean up Three.js resources - Hands
    this.leftHandJoints.forEach((j) => {
      j.geometry.dispose()
      ;(j.material as THREE.Material).dispose()
    })

    this.rightHandJoints.forEach((j) => {
      j.geometry.dispose()
      ;(j.material as THREE.Material).dispose()
    })

    this.leftHandBones.forEach((b) => {
      b.line.geometry.dispose()
      ;(b.line.material as THREE.Material).dispose()
    })

    this.rightHandBones.forEach((b) => {
      b.line.geometry.dispose()
      ;(b.line.material as THREE.Material).dispose()
    })

    this.renderer.dispose()

    // Remove canvas from DOM
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement)
    }
  }
}
