import * as THREE from 'three';

export class AvatarController {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.joints = { left: [], right: [] }; // Spheres
        this.bones = { left: [], right: [] };  // Lines
        this.initScene();
        this.createRig();
        this.animate();
    }

    initScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a1a); // Dark gray

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

        // Grid (Floor)
        const gridHelper = new THREE.GridHelper(2, 10, 0x444444, 0x222222);
        gridHelper.position.y = -0.2;
        this.scene.add(gridHelper);
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
        // Hand Topology (Connections)
        // 0:Wrist, 1-4:Thumb, 5-8:Index, 9-12:Middle, 13-16:Ring, 17-20:Pinky
        this.connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],       // Index
            [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
            [0, 13], [13, 14], [14, 15], [15, 16],// Ring
            [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
        ];

        // Create Joints (Spheres)
        const geometry = new THREE.SphereGeometry(0.012, 8, 8);
        const matLeft = new THREE.MeshLambertMaterial({ color: 0x00ff88 });
        const matRight = new THREE.MeshLambertMaterial({ color: 0xff6644 });
        const lineMatLeft = new THREE.LineBasicMaterial({ color: 0x00ff88 });
        const lineMatRight = new THREE.LineBasicMaterial({ color: 0xff6644 });

        ['left', 'right'].forEach(side => {
            const mat = side === 'left' ? matLeft : matRight;
            const lineMat = side === 'left' ? lineMatLeft : lineMatRight;

            // 21 Joint Spheres
            for (let i = 0; i < 21; i++) {
                const mesh = new THREE.Mesh(geometry, mat);
                mesh.visible = false;
                this.scene.add(mesh);
                this.joints[side].push(mesh);
            }

            // 20 Bone Lines
            this.connections.forEach(conn => {
                const lineGeom = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0)
                ]);
                const line = new THREE.Line(lineGeom, lineMat);
                line.visible = false;
                this.scene.add(line);
                this.bones[side].push({ line, start: conn[0], end: conn[1] });
            });
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }

    updateFrame(frame) {
        if (!frame) return;

        // frame = { left: [[x,y,z]...], right: [[x,y,z]...] }
        ['left', 'right'].forEach(side => {
            const points = frame[side]; // Array of 21 [x,y,z]
            const joints = this.joints[side];
            const bones = this.bones[side];

            // Check if hand is present - look for ANY non-zero joint, not just wrist
            const isPresent = points && points.some(p =>
                p && !isNaN(p[0]) && (p[0] !== 0 || p[1] !== 0 || p[2] !== 0)
            );

            if (!isPresent) {
                joints.forEach(j => j.visible = false);
                bones.forEach(b => b.line.visible = false);
                return;
            }

            // Calculate bounding box to normalize hand size
            const validPoints = points.filter(p => p[0] !== 0 || p[1] !== 0 || p[2] !== 0);
            if (validPoints.length === 0) {
                joints.forEach(j => j.visible = false);
                bones.forEach(b => b.line.visible = false);
                return;
            }

            // Find min/max to get hand size
            let minX = Infinity, maxX = -Infinity;
            let minY = Infinity, maxY = -Infinity;
            for (const p of validPoints) {
                minX = Math.min(minX, p[0]);
                maxX = Math.max(maxX, p[0]);
                minY = Math.min(minY, p[1]);
                maxY = Math.max(maxY, p[1]);
            }

            // Calculate center and size
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const handWidth = maxX - minX;
            const handHeight = maxY - minY;
            const handSize = Math.max(handWidth, handHeight);

            // Target size for normalized hand (in world units)
            const targetSize = 0.3;
            const normalizeScale = handSize > 0.001 ? targetSize / handSize : 1;

            // Update Joints
            for (let i = 0; i < 21; i++) {
                const p = points[i];
                const j = joints[i];

                // Skip if this joint is zero
                if (p[0] === 0 && p[1] === 0 && p[2] === 0) {
                    j.visible = false;
                    continue;
                }

                // Normalize: center the hand, then scale to standard size
                const xOff = side === 'left' ? -0.35 : 0.35;  // Separation between hands
                const yOff = -0.1;  // Push hands lower

                j.position.set(
                    ((p[0] - centerX) * normalizeScale) + xOff,
                    (-(p[1] - centerY) * normalizeScale) + yOff,
                    -p[2] * normalizeScale * 0.3
                );
                j.visible = true;
            }

            // Update Bones
            bones.forEach(b => {
                const jStart = joints[b.start];
                const jEnd = joints[b.end];

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
        });
    }


    async playSequence(frames, fps = 30) {
        const interval = 1000 / fps;

        // Filter to only frames with actual hand data
        const validFrames = frames.filter(frame => {
            // Check if either hand has non-zero data
            const hasLeft = frame.left && frame.left.some(p => p[0] !== 0 || p[1] !== 0 || p[2] !== 0);
            const hasRight = frame.right && frame.right.some(p => p[0] !== 0 || p[1] !== 0 || p[2] !== 0);
            return hasLeft || hasRight;
        });

        console.log(`Playing ${validFrames.length} valid frames (out of ${frames.length} total)`);

        for (const frame of validFrames) {
            this.updateFrame(frame);
            await new Promise(r => setTimeout(r, interval));
        }

        // Hide hands after sequence
        ['left', 'right'].forEach(side => {
            this.joints[side].forEach(j => j.visible = false);
            this.bones[side].forEach(b => b.line.visible = false);
        });
    }
}
