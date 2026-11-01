// All resources belong to one session; a closed session cannot finish loading later.
const modal = document.getElementById('viewer-modal');
const container = document.getElementById('viewer-canvas');
const statusMessage = document.getElementById('viewer-status');
const retryButton = document.getElementById('retry-viewer-btn');
const resetButton = document.getElementById('reset-viewer-btn');
const closeButton = document.getElementById('close-viewer-btn');
let session = null;
let libraries = null;

function loadLibraries() {
    if (!libraries) {
        libraries = Promise.all([
            import('three'),
            import('./vendor/three/OrbitControls.js'),
            import('./vendor/three/OBJLoader.js'),
        ]).catch(error => {
            libraries = null;
            throw error;
        });
    }
    return libraries;
}

function disposeObject(object) {
    const materials = new Set();
    object.traverse(child => {
        child.geometry?.dispose();
        for (const material of [child.material].flat().filter(Boolean)) {
            materials.add(material);
        }
    });
    materials.forEach(material => material.dispose());
}

function release(current) {
    current.request.abort();
    current.observer?.disconnect();
    current.controls?.dispose();
    current.renderer?.setAnimationLoop(null);
    if (current.scene) disposeObject(current.scene);
    current.renderer?.dispose();
    current.renderer?.forceContextLoss();
    container.replaceChildren();
}

export function closeViewer() {
    if (!session) return;
    const current = session;
    session = null;
    release(current);
    modal.classList.add('hidden');
    document.body.classList.remove('viewer-open');
    current.trigger?.focus();
}

function prepareModel(THREE, OBJLoader, text, scene) {
    const object = new OBJLoader().parse(text);
    scene.add(object); // Own parsed resources even if validation fails below.
    // HMR exports camera coordinates (Y down). Match 4D-Humans' display rotation.
    object.rotation.x = Math.PI;
    let meshCount = 0;
    object.traverse(child => {
        if (!child.isMesh) return;
        meshCount++;
        const positions = child.geometry.attributes.position;
        if (!positions || !positions.count || !positions.array.every(Number.isFinite)) {
            throw new Error('The model contains invalid geometry.');
        }
        for (const material of [child.material].flat()) material?.dispose();
        child.material = new THREE.MeshPhongMaterial({
            color: 0xa5c8e1, shininess: 30, side: THREE.DoubleSide,
        });
        child.geometry.computeVertexNormals();
    });
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const radius = size.length() / 2;
    if (!meshCount || box.isEmpty() || !Number.isFinite(radius) || radius <= 0) {
        throw new Error('The model has no usable mesh.');
    }
    object.position.sub(box.getCenter(new THREE.Vector3()));
    return { size, radius };
}

function startRendering(THREE, OrbitControls, current, { size, radius }) {
    const scene = current.scene;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    current.renderer = renderer;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.append(renderer.domElement);
    renderer.domElement.addEventListener('webglcontextlost', event => {
        event.preventDefault();
        if (session !== current) return;
        release(current);
        statusMessage.textContent = 'The 3D display was interrupted. Try loading the model again.';
        statusMessage.hidden = false;
        retryButton.hidden = false;
        resetButton.disabled = true;
    });
    const camera = new THREE.PerspectiveCamera(45, 1, radius / 100, radius * 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    current.controls = controls;
    controls.enableDamping = true;
    controls.minDistance = radius * 1.1;
    controls.maxDistance = radius * 20;
    scene.add(new THREE.AmbientLight(0xffffff, 1.4));
    const light = new THREE.DirectionalLight(0xffffff, 2);
    light.position.set(3, 5, 5);
    scene.add(light);
    const backlight = new THREE.DirectionalLight(0xffffff, 1);
    backlight.position.set(-3, 2, -4);
    scene.add(backlight);

    function fitCamera() {
        const verticalHalfFov = THREE.MathUtils.degToRad(camera.fov) / 2;
        const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * camera.aspect);
        const distance = Math.max(
            size.y / (2 * Math.tan(verticalHalfFov)),
            size.x / (2 * Math.tan(horizontalHalfFov)),
        ) * 1.25 + size.z / 2;
        camera.far = Math.max(radius * 100, distance * 2);
        camera.updateProjectionMatrix();
        camera.position.set(0, 0, distance);
        controls.maxDistance = Math.max(radius * 20, distance * 2);
        controls.target.set(0, 0, 0);
        controls.update();
    }
    function resize() {
        const width = container.clientWidth;
        const height = container.clientHeight;
        if (!width || !height) return;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
        fitCamera();
    }
    current.observer = new ResizeObserver(resize);
    current.observer.observe(container);
    resize();
    resetButton.onclick = fitCamera;
    resetButton.disabled = false;
    statusMessage.hidden = true;
    renderer.setAnimationLoop(() => {
        controls.update();
        renderer.render(scene, camera);
    });
}

export async function openViewer(filename) {
    const trigger = session?.trigger || document.activeElement;
    if (session) release(session);
    const current = { request: new AbortController(), trigger };
    session = current;
    modal.classList.remove('hidden');
    document.body.classList.add('viewer-open');
    statusMessage.textContent = 'Loading 3D model…';
    statusMessage.hidden = false;
    retryButton.hidden = true;
    resetButton.disabled = true;
    closeButton.focus();
    retryButton.onclick = () => openViewer(filename);

    try {
        const [THREE, { OrbitControls }, { OBJLoader }] = await loadLibraries();
        if (session !== current) return;
        const response = await fetch(`/files/obj/${encodeURIComponent(filename)}`, {
            signal: current.request.signal,
        });
        if (!response.ok) {
            throw new Error(`Model could not be loaded (HTTP ${response.status}).`);
        }
        const text = await response.text();
        if (session !== current) return;

        const scene = new THREE.Scene();
        current.scene = scene;
        scene.background = new THREE.Color(0x0a0e1a);
        const bounds = prepareModel(THREE, OBJLoader, text, scene);
        startRendering(THREE, OrbitControls, current, bounds);
    } catch (error) {
        if (session !== current) return;
        release(current);
        statusMessage.textContent = `Unable to display the model. ${error.message}`;
        statusMessage.hidden = false;
        retryButton.hidden = false;
    }
}

closeButton.addEventListener('click', closeViewer);
modal.addEventListener('click', event => {
    if (event.target === modal) closeViewer();
});
modal.addEventListener('keydown', event => {
    if (event.key === 'Escape') closeViewer();
    if (event.key !== 'Tab') return;
    const buttons = [...modal.querySelectorAll('button')]
        .filter(button => !button.hidden && !button.disabled);
    const first = buttons[0];
    const last = buttons.at(-1);
    if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
    }
});
