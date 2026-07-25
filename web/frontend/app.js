import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const API_BASE = '';

// State
let selectedFiles = [];
let currentJobId = null;
let currentScene = null;
let currentRenderer = null;
let animationId = null;
let controls = null;

// DOM Elements
const sections = {
    upload: document.getElementById('upload-section'),
    processing: document.getElementById('processing-section'),
    results: document.getElementById('results-section')
};

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const thumbnailGrid = document.getElementById('thumbnail-grid');
const analyzeBtn = document.getElementById('analyze-btn');
const statusText = document.getElementById('status-text');
const resultsContainer = document.getElementById('results-container');
const downloadCsvBtn = document.getElementById('download-csv-btn');
const newAnalysisBtn = document.getElementById('new-analysis-btn');
const viewerModal = document.getElementById('viewer-modal');
const closeViewerBtn = document.getElementById('close-viewer-btn');
const canvasContainer = document.getElementById('3d-canvas-container');

// Event Listeners
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFiles(e.dataTransfer.files);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFiles(e.target.files);
    }
});

analyzeBtn.addEventListener('click', () => {
    if (selectedFiles.length > 0) {
        uploadImages(selectedFiles);
    }
});

newAnalysisBtn.addEventListener('click', resetApp);
downloadCsvBtn.addEventListener('click', () => downloadCSV(currentJobId));
closeViewerBtn.addEventListener('click', closeViewer);

function switchSection(sectionId) {
    Object.values(sections).forEach(sec => sec.classList.remove('active'));
    sections[sectionId].classList.add('active');
}

function handleFiles(files) {
    const validFiles = Array.from(files).filter(f => f.type.match('image.*'));
    if (validFiles.length > 0) {
        // Append to existing
        selectedFiles = [...selectedFiles, ...validFiles];
        
        thumbnailGrid.innerHTML = '';
        selectedFiles.forEach(file => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'thumbnail';
                thumbnailGrid.appendChild(img);
            };
            reader.readAsDataURL(file);
        });

        previewContainer.classList.remove('hidden');
    }
}

async function uploadImages(files) {
    switchSection('processing');
    statusText.textContent = "Uploading...";
    
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const data = await response.json();
        currentJobId = data.job_id;
        
        pollStatus(currentJobId);
    } catch (err) {
        alert("Error: " + err.message);
        resetApp();
    }
}

async function pollStatus(jobId) {
    try {
        const response = await fetch(`${API_BASE}/status/${jobId}`);
        const data = await response.json();
        
        statusText.textContent = data.message || "Processing...";

        if (data.status === 'done') {
            fetchResults(jobId);
        } else if (data.status === 'error') {
            throw new Error(data.message || "Processing failed");
        } else {
            setTimeout(() => pollStatus(jobId), 2000);
        }
    } catch (err) {
        alert("Error: " + err.message);
        resetApp();
    }
}

async function fetchResults(jobId) {
    try {
        statusText.textContent = "Loading results...";
        const response = await fetch(`${API_BASE}/results/${jobId}`);
        const data = await response.json();
        
        renderResults(data);
        switchSection('results');
    } catch (err) {
        alert("Error fetching results: " + err.message);
        resetApp();
    }
}

function getHealthStatusInfo(ratio) {
    if (ratio < 0.40) return { label: 'Underweight', class: 'under' };
    if (ratio < 0.50) return { label: 'Healthy', class: 'healthy' };
    if (ratio < 0.60) return { label: 'Increased Risk', class: 'increased' };
    return { label: 'High Risk', class: 'high' };
}

let cachedResults = null;

function renderResults(data) {
    resultsContainer.innerHTML = '';
    
    const results = data.results || [];
    cachedResults = results;

    if (results.length === 0) {
        resultsContainer.innerHTML = '<p class="muted">No results found.</p>';
        return;
    }

    results.forEach(imgResult => {
        const group = document.createElement('div');
        group.className = 'image-result-group';
        
        let html = '';
        
        if (imgResult.result_image) {
            const encodedImg = encodeURIComponent(imgResult.result_image);
            const imgUrl = `${API_BASE}/files/result-img/${encodedImg}`;
            html += `<img src="${imgUrl}" class="annotated-img" alt="Analyzed Image">`;
        }
        
        html += `<h3 style="margin-bottom: 1rem; color: var(--text-muted); font-size: 0.9rem;">${imgResult.image_name}</h3>`;
        html += `<div class="person-cards">`;

        if (!imgResult.persons || imgResult.persons.length === 0) {
            html += `<p class="muted">No people detected in this image.</p>`;
        } else {
            imgResult.persons.forEach(person => {
                const health = getHealthStatusInfo(person.ratio);
                const confPercent = Math.round(person.confidence * 100);
                
                html += `
                    <div class="person-card">
                        <div class="card-header">
                            <span class="muted">Person ${person.person_id}</span>
                            <span class="badge ${health.class}">${health.label}</span>
                        </div>
                        <div style="text-align: center; margin: 1.5rem 0;">
                            <div class="muted">WHR (Waist-to-Height Ratio)</div>
                            <div class="whr-value" style="color: var(--status-${health.class})">${person.ratio.toFixed(3)}</div>
                        </div>
                        <div class="measurements">
                            <div class="measure">
                                <div class="label">Waist</div>
                                <div class="val">${person.waist_cm.toFixed(1)} cm</div>
                            </div>
                            <div class="measure">
                                <div class="label">Height</div>
                                <div class="val">${person.height_cm.toFixed(1)} cm</div>
                            </div>
                        </div>
                        <div class="confidence">
                            <div style="display:flex; justify-content:space-between; font-size:0.85rem">
                                <span class="muted">AI Confidence</span>
                                <span>${confPercent}%</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confPercent}%"></div>
                            </div>
                        </div>
                        ${person.flags && person.flags.length ? `<div style="margin-bottom: 1rem; color: var(--status-increased); font-size: 0.85rem;">⚠️ ${person.flags.join(', ')}</div>` : ''}
                        <button class="btn viewer-btn" data-obj="${person.obj_file}">View 3D Model</button>
                    </div>
                `;
            });
        }
        
        html += `</div>`;
        group.innerHTML = html;
        resultsContainer.appendChild(group);
    });

    document.querySelectorAll('.viewer-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const objFile = e.target.getAttribute('data-obj');
            if (objFile) launch3DViewer(objFile);
        });
    });
}

function launch3DViewer(objFilename) {
    viewerModal.classList.remove('hidden');
    
    // Setup Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e1a);
    currentScene = scene;

    // Setup Camera
    const camera = new THREE.PerspectiveCamera(45, canvasContainer.clientWidth / canvasContainer.clientHeight, 0.1, 100);
    camera.position.set(0, 0, 3);

    // Setup Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    canvasContainer.innerHTML = '';
    canvasContainer.appendChild(renderer.domElement);
    currentRenderer = renderer;

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 2.0;
    
    controls.addEventListener('start', () => {
        controls.autoRotate = false;
    });

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 5, 5);
    scene.add(dirLight);

    // Grid (Subtle)
    const grid = new THREE.GridHelper(10, 10, 0x4a9eff, 0x4a9eff);
    grid.material.opacity = 0.1;
    grid.material.transparent = true;
    grid.position.y = -1;
    scene.add(grid);

    // Load OBJ
    const loader = new OBJLoader();
    const objUrl = `${API_BASE}/files/obj/${encodeURIComponent(objFilename)}`;
    
    loader.load(
        objUrl,
        (object) => {
            const material = new THREE.MeshPhongMaterial({
                color: 0xA5C8E1,
                shininess: 30,
                side: THREE.DoubleSide
            });
            
            object.traverse((child) => {
                if (child.isMesh) {
                    child.material = material;
                }
            });

            // Center mesh
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            object.position.sub(center);

            // Scale if needed (assuming roughly meters, scale to fit view)
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            if(maxDim > 0) {
                const scale = 2 / maxDim; // Fit within 2 units
                object.scale.setScalar(scale);
            }

            scene.add(object);
        },
        (xhr) => {
            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
        },
        (error) => {
            console.error('Error loading OBJ:', error);
            alert("Failed to load 3D model.");
        }
    );

    // Animation Loop
    function animate() {
        animationId = requestAnimationFrame(animate);
        if (controls) controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // Resize handler
    window.addEventListener('resize', onWindowResize);
}

function onWindowResize() {
    if (!currentRenderer || !currentScene) return;
    const camera = currentScene.children.find(c => c.isPerspectiveCamera);
    if (camera) {
        camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
        camera.updateProjectionMatrix();
        currentRenderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
    }
}

function closeViewer() {
    viewerModal.classList.add('hidden');
    if (animationId) cancelAnimationFrame(animationId);
    if (currentRenderer) {
        currentRenderer.dispose();
        canvasContainer.innerHTML = '';
    }
    window.removeEventListener('resize', onWindowResize);
    controls = null;
    currentScene = null;
    currentRenderer = null;
}

function downloadCSV(jobId) {
    if (!jobId || !cachedResults) return;
    
    let csv = 'Image,Person_ID,Waist_cm,Height_cm,Waist_Height_Ratio,Status,Confidence\n';
    cachedResults.forEach(img => {
        (img.persons || []).forEach(p => {
            csv += `${img.image_name},${p.person_id},${p.waist_cm.toFixed(2)},${p.height_cm.toFixed(2)},${p.ratio.toFixed(4)},${p.status},${p.confidence.toFixed(2)}\n`;
        });
    });
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'bodyscan_results.csv';
    a.click();
    URL.revokeObjectURL(url);
}

function resetApp() {
    selectedFiles = [];
    currentJobId = null;
    fileInput.value = '';
    thumbnailGrid.innerHTML = '';
    previewContainer.classList.add('hidden');
    switchSection('upload');
}
