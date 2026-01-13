import * as THREE from 'three';

export class AvatarController {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.joints = []; // 33 pose joints (spheres)
        this.bones = [];  // Pose bone connections (lines)
        this.initScene();
        this.createRig();
        this.animate();
    }

    initScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xFBF9F7); // Warm beige to match UI

        // Camera - use default aspect, will be fixed on resize
        this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
        this.camera.position.set(0, 0, 2);  // Zoomed out, looking at center
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(1, 1); // Start small, resize later
        this.container.appendChild(this.renderer.domElement);

        // Light
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(2, 2, 5);
        this.scene.add(light);
        this.scene.add(new THREE.AmbientLight(0x404040));
    }

    // Resize canvas when container becomes visible
    resize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        if (width > 0 && height > 0) {
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
        }
    }

    createRig() {
        // MediaPipe Pose Topology (Connections)
        // Based on MediaPipe Pose landmark structure (33 landmarks)
        this.connections = [
            // Face
            [0, 1], [1, 2], [2, 3], [3, 7],  // Left eye
            [0, 4], [4, 5], [5, 6], [6, 8],  // Right eye
            [9, 10],  // Mouth
            // Upper body
            [11, 12],  // Shoulders
            [11, 13], [13, 15],  // Left arm
            [12, 14], [14, 16],  // Right arm
            [11, 23], [12, 24],  // Shoulder to hip
            [23, 24],  // Hips
            // Lower body
            [23, 25], [25, 27],  // Left leg
            [24, 26], [26, 28],  // Right leg
        ];

        // Create Joints (Spheres) - 33 joints for MediaPipe Pose
        const geometry = new THREE.SphereGeometry(0.015, 8, 8);
        
        // Color scheme: head (blue), upper body (green), lower body (red), joints (yellow)
        const matHead = new THREE.MeshLambertMaterial({ color: 0x4488ff });
        const matUpperBody = new THREE.MeshLambertMaterial({ color: 0x00ff88 });
        const matLowerBody = new THREE.MeshLambertMaterial({ color: 0xff6644 });
        const matJoint = new THREE.MeshLambertMaterial({ color: 0xffff44 });
        
        // Line material for bones
        const lineMat = new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2 });

        // Create 33 joint spheres
        for (let i = 0; i < 33; i++) {
            let mat;
            if (i <= 10) {
                mat = matHead;  // Head landmarks (0-10)
            } else if (i <= 28) {
                mat = matUpperBody;  // Upper body and visible lower body
            } else {
                mat = matJoint;  // Feet landmarks (29-32)
            }
            
            const mesh = new THREE.Mesh(geometry, mat);
            mesh.visible = false;
            this.scene.add(mesh);
            this.joints.push(mesh);
        }

        // Create bone connections
        this.connections.forEach(conn => {
            const lineGeom = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0)
            ]);
            const line = new THREE.Line(lineGeom, lineMat);
            line.visible = false;
            this.scene.add(line);
            this.bones.push({ line, start: conn[0], end: conn[1] });
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }

    updateFrame(frame) {
        if (!frame) {
            // Hide all joints and bones if no frame
            this.joints.forEach(j => j.visible = false);
            this.bones.forEach(b => b.line.visible = false);
            return false; // Return false to indicate blank frame
        }

        // frame = { pose: [[x,y,z]...] } - 33 landmarks
        const points = frame.pose;
        
        if (!points || !Array.isArray(points) || points.length !== 33) {
            // Hide all joints and bones if invalid data
            this.joints.forEach(j => j.visible = false);
            this.bones.forEach(b => b.line.visible = false);
            return false; // Return false to indicate blank frame
        }

        // Check if pose is present - look for ANY non-zero joint
        const isPresent = points.some(p =>
            p && Array.isArray(p) && p.length >= 3 && 
            !isNaN(p[0]) && (p[0] !== 0 || p[1] !== 0 || p[2] !== 0)
        );

        if (!isPresent) {
            this.joints.forEach(j => j.visible = false);
            this.bones.forEach(b => b.line.visible = false);
            return false; // Return false to indicate blank frame
        }

        // MediaPipe Pose coordinates are normalized (x, y in [0, 1], z is relative depth)
        // We need to scale and center them appropriately
        // Scale factor to fit in the scene nicely
        const scale = 1.5;  // Scale up from normalized [0,1] range
        const zScale = 0.5;   // Scale depth (z is relative, usually smaller)

        // First, hide all joints
        this.joints.forEach(j => j.visible = false);

        // Update Joints
        for (let i = 0; i < 33; i++) {
            const p = points[i];
            const j = this.joints[i];

            // Skip if this joint is zero or invalid
            if (!p || !Array.isArray(p) || p.length < 3 ||
                (p[0] === 0 && p[1] === 0 && p[2] === 0)) {
                continue;
            }

            // Transform coordinates
            // MediaPipe uses: x, y in [0, 1] (image coordinates), z is relative depth
            // We flip Y because MediaPipe origin is top-left, but 3D is bottom-left
            j.position.set(
                (p[0] - 0.5) * scale,      // X: center and scale
                (-(p[1] - 0.5)) * scale,   // Y: flip, center and scale (MediaPipe origin is top-left)
                -p[2] * zScale              // Z: depth (negative to move away from camera)
            );
            j.visible = true;
        }

        // Update Bones
        this.bones.forEach(b => {
            const jStart = this.joints[b.start];
            const jEnd = this.joints[b.end];

            // Only show bone if both joints are visible
            if (!jStart.visible || !jEnd.visible) {
                b.line.visible = false;
                return;
            }

            const pStart = jStart.position;
            const pEnd = jEnd.position;
            const positions = b.line.geometry.attributes.position.array;

            positions[0] = pStart.x; positions[1] = pStart.y; positions[2] = pStart.z;
            positions[3] = pEnd.x; positions[4] = pEnd.y; positions[5] = pEnd.z;

            b.line.geometry.attributes.position.needsUpdate = true;
            b.line.visible = true;
        });
        
        return true; // Return true to indicate valid frame with visible joints
    }


    async playSequence(frames, fps = 30) {
        if (!frames || frames.length === 0) {
            return false; // No frames = blank
        }
        
        const interval = 1000 / fps;
        let hasValidFrames = false;
        let consecutiveBlankFrames = 0;
        const maxConsecutiveBlanks = 10; // If 10+ consecutive blank frames, consider it blank
        const earlyCheckFrames = 5; // Check first 5 frames to detect if sequence is entirely blank
        
        // First, quickly check if the sequence starts with blank frames
        let earlyBlankCount = 0;
        for (let i = 0; i < Math.min(earlyCheckFrames, frames.length); i++) {
            const frame = frames[i];
            const points = frame?.pose;
            const isPresent = points && Array.isArray(points) && points.length === 33 &&
                points.some(p =>
                    p && Array.isArray(p) && p.length >= 3 && 
                    !isNaN(p[0]) && (p[0] !== 0 || p[1] !== 0 || p[2] !== 0)
                );
            if (!isPresent) {
                earlyBlankCount++;
            }
        }
        
        // If all early frames are blank, the sequence is likely entirely blank
        if (earlyBlankCount === Math.min(earlyCheckFrames, frames.length)) {
            console.log('Skeleton sequence appears to be entirely blank (detected early)');
            // Still update the first frame to hide everything
            this.updateFrame(frames[0]);
            return false;
        }
        
        // Play through frames
        for (const frame of frames) {
            const isValid = this.updateFrame(frame);
            
            if (isValid) {
                hasValidFrames = true;
                consecutiveBlankFrames = 0;
            } else {
                consecutiveBlankFrames++;
                // If we hit too many consecutive blanks after having valid frames, stop early
                if (hasValidFrames && consecutiveBlankFrames >= maxConsecutiveBlanks) {
                    console.log('Skeleton rendering went blank, stopping sequence early');
                    break;
                }
            }
            
            await new Promise(r => setTimeout(r, interval));
        }
        
        // Return whether we had any valid frames
        return hasValidFrames;
    }
}
