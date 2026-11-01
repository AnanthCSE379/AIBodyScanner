import { openViewer, closeViewer } from './viewer.js?v=8';

const MAX_FILES = 10;
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const POLL_INTERVAL_MS = 2000;
const STATUS_COLORS = {
    Underweight: 'under',
    Healthy: 'healthy',
    'Increased Risk': 'increased',
    'High Risk': 'high',
};

const byId = id => document.getElementById(id);
const fileInput = byId('file-input');
const dropZone = byId('drop-zone');
const analyze = byId('analyze-btn');
const statusText = byId('status-text');
const errorBox = byId('app-error');
const csvButton = byId('download-csv-btn');
const resultsContainer = byId('results-container');
let selectedFiles = [];
let previewUrls = [];
let cachedResults = [];
let currentJobId = null;
let activeRequest = null;
let pollTimer = null;

function element(tag, className = '', text = '') {
    const node = document.createElement(tag);
    node.className = className;
    node.textContent = text;
    return node;
}

function switchSection(name) {
    for (const section of document.querySelectorAll('.app-container > section')) {
        section.classList.toggle('active', section.id === `${name}-section`);
    }
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = !message;
}

// Each upload/poll chain owns a controller. Ignore responses from older chains.
function stopRequests() {
    clearTimeout(pollTimer);
    activeRequest?.abort();
    activeRequest = null;
}

async function getJSON(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        const message = typeof data.detail === 'string'
            ? data.detail
            : `Request failed (HTTP ${response.status}).`;
        throw new Error(message);
    }
    return data;
}

function renderPreviews() {
    previewUrls.forEach(url => URL.revokeObjectURL(url));
    previewUrls = [];
    const grid = byId('thumbnail-grid');
    grid.replaceChildren();
    selectedFiles.forEach((file, index) => {
        const item = element('div', 'preview-item');
        const image = element('img', 'thumbnail');
        image.src = URL.createObjectURL(file);
        image.alt = file.name;
        previewUrls.push(image.src);
        const remove = element('button', 'remove-file', 'Remove');
        remove.setAttribute('aria-label', `Remove ${file.name}`);
        remove.onclick = () => {
            selectedFiles.splice(index, 1);
            renderPreviews();
        };
        item.append(image, element('span', 'preview-name', file.name), remove);
        grid.append(item);
    });
    byId('preview-container').classList.toggle('hidden', !selectedFiles.length);
    analyze.disabled = !selectedFiles.length;
}

function addFiles(incoming) {
    showError('');
    const errors = [];
    for (const file of incoming) {
        const supportedExtension = /\.(jpe?g|png)$/i.test(file.name);
        const supportedType = !file.type || ['image/jpeg', 'image/png'].includes(file.type);
        if (!supportedExtension || !supportedType) {
            errors.push(`${file.name}: select a JPEG or PNG.`);
        } else if (file.size > MAX_FILE_BYTES) {
            errors.push(`${file.name}: the limit is 20 MB per image.`);
        } else if (selectedFiles.length >= MAX_FILES) {
            errors.push('Select at most 10 images per analysis.');
            break;
        } else {
            selectedFiles.push(file);
        }
    }
    fileInput.value = '';
    renderPreviews();
    if (errors.length) showError(errors.join(' '));
}

async function poll(id, controller) {
    try {
        const data = await getJSON(`/status/${id}`, { signal: controller.signal });
        if (activeRequest !== controller) return;
        statusText.textContent = data.message || 'Processing…';
        if (data.status === 'error') throw new Error(data.message || 'Analysis failed.');
        if (data.status === 'done') {
            const response = await getJSON(`/results/${id}`, { signal: controller.signal });
            if (activeRequest !== controller) return;
            cachedResults = response.results || [];
            renderResults();
            switchSection('results');
            activeRequest = null;
        } else {
            pollTimer = setTimeout(() => poll(id, controller), POLL_INTERVAL_MS);
        }
    } catch (error) {
        if (activeRequest !== controller || error.name === 'AbortError') return;
        stopRequests();
        statusText.textContent = 'Analysis interrupted';
        showError(error.message);
        byId('retry-status-btn').hidden = false;
    }
}

async function upload() {
    if (!selectedFiles.length || activeRequest) return;
    stopRequests();
    showError('');
    cachedResults = [];
    csvButton.disabled = true;
    currentJobId = null;
    const controller = new AbortController();
    activeRequest = controller;
    switchSection('processing');
    byId('retry-status-btn').hidden = true;
    statusText.textContent = 'Uploading…';
    const body = new FormData();
    selectedFiles.forEach(file => body.append('files', file));
    body.append('original_names', JSON.stringify(selectedFiles.map(file => file.name)));
    try {
        const data = await getJSON('/upload', { method: 'POST', body, signal: controller.signal });
        if (activeRequest !== controller) return;
        currentJobId = data.job_id;
        await poll(currentJobId, controller);
    } catch (error) {
        if (activeRequest !== controller || error.name === 'AbortError') return;
        stopRequests();
        switchSection('upload');
        showError(error.message);
    }
}

function validMeasurement(person) {
    const measurements = [person.ratio, person.waist_cm, person.height_cm];
    return person.status !== 'Invalid'
        && measurements.every(value => Number.isFinite(value) && value > 0);
}

function format(value, digits = 1) {
    return Number.isFinite(value) ? value.toFixed(digits) : 'Unavailable';
}

function formatConfidence(value) {
    if (!Number.isFinite(value)) return 'Unavailable';
    const percentage = Math.round(Math.min(1, Math.max(0, value)) * 100);
    return `${percentage}%`;
}

function personCard(person) {
    const valid = validMeasurement(person);
    const card = element('div', 'person-card');
    const header = element('div', 'card-header');
    const statusClass = valid ? STATUS_COLORS[person.status] || '' : 'invalid';
    const statusLabel = valid ? person.status : 'Invalid';
    header.append(
        element('span', 'muted', `Person ${person.person_id}`),
        element('span', `badge ${statusClass}`, statusLabel),
    );
    const ratio = element('div', 'ratio-summary');
    ratio.append(
        element('div', 'muted', 'Waist-to-height ratio'),
        element('div', 'whr-value', valid ? format(person.ratio, 3) : 'Unavailable'),
    );
    const measurements = element('div', 'measurements');
    for (const [label, value] of [['Waist', person.waist_cm], ['Height', person.height_cm]]) {
        const measure = element('div', 'measure');
        measure.append(
            element('div', 'label', label),
            element('div', 'val', valid ? `${format(value)} cm` : 'Unavailable'),
        );
        measurements.append(measure);
    }
    const confidence = formatConfidence(person.confidence);
    const viewer = element('button', 'btn viewer-btn', 'View 3D Model');
    viewer.disabled = !person.obj_file;
    viewer.onclick = () => openViewer(person.obj_file);
    card.append(
        header, ratio, measurements,
        element('p', 'confidence muted', `AI confidence: ${confidence}`),
    );
    if (person.flags?.length) {
        card.append(element('p', 'measurement-flags', person.flags.join(', ')));
    }
    card.append(viewer);
    return card;
}

function renderResults() {
    resultsContainer.replaceChildren();
    csvButton.disabled = !cachedResults.some(image => image.persons?.length);
    if (!cachedResults.length) {
        resultsContainer.append(element('p', 'muted',
            'No results found. Try a clear, full-body image.'));
    }
    for (const result of cachedResults) {
        const group = element('div', 'image-result-group');
        if (result.result_image) {
            const image = element('img', 'annotated-img');
            image.src = `/files/result-img/${encodeURIComponent(result.result_image)}`;
            image.alt = `Analysis of ${result.image_name}`;
            image.onerror = () => {
                image.replaceWith(element('p', 'muted', 'The annotated image could not be loaded.'));
            };
            group.append(image);
        }
        group.append(element('h3', 'result-name muted', result.image_name));
        const cards = element('div', 'person-cards');
        for (const person of result.persons || []) cards.append(personCard(person));
        if (!result.persons?.length) {
            cards.append(element('p', 'muted',
                'No people detected. Try a clear, full-body image.'));
        }
        group.append(cards);
        resultsContainer.append(group);
    }
}

function csvCell(value) {
    let text = String(value ?? '');
    // CSV quoting alone does not stop spreadsheets from evaluating formulas.
    if (/^[\s]*[=+\-@]/.test(text) || /^[\t\r\n]/.test(text)) text = `'${text}`;
    return `"${text.replaceAll('"', '""')}"`;
}

function downloadCSV() {
    const rows = [[
        'Image', 'Person_ID', 'Waist_cm', 'Height_cm',
        'Waist_Height_Ratio', 'Status', 'Confidence',
    ]];
    for (const image of cachedResults) {
        for (const person of image.persons || []) {
            const valid = validMeasurement(person);
            rows.push([
                image.image_name,
                person.person_id,
                valid ? format(person.waist_cm, 2) : '',
                valid ? format(person.height_cm, 2) : '',
                valid ? format(person.ratio, 4) : '',
                valid ? person.status : 'Invalid',
                Number.isFinite(person.confidence) ? format(person.confidence, 2) : '',
            ]);
        }
    }
    const csv = rows.map(row => row.map(csvCell).join(',')).join('\r\n');
    const blob = new Blob(['\ufeff', csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = element('a');
    link.href = url;
    link.download = 'bodyscan_results.csv';
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function resetApp() {
    stopRequests();
    closeViewer();
    selectedFiles = [];
    cachedResults = [];
    currentJobId = null;
    csvButton.disabled = true;
    resultsContainer.replaceChildren();
    renderPreviews();
    showError('');
    switchSection('upload');
}

dropZone.addEventListener('dragover', event => {
    event.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', event => {
    event.preventDefault();
    dropZone.classList.remove('dragover');
    addFiles(event.dataTransfer.files);
});
byId('browse-btn').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => addFiles(fileInput.files));
analyze.addEventListener('click', upload);
byId('new-analysis-btn').addEventListener('click', resetApp);
byId('back-btn').addEventListener('click', () => {
    stopRequests();
    switchSection('upload');
    showError('');
});
byId('retry-status-btn').addEventListener('click', () => {
    if (!currentJobId) return;
    stopRequests();
    showError('');
    byId('retry-status-btn').hidden = true;
    activeRequest = new AbortController();
    poll(currentJobId, activeRequest);
});
csvButton.addEventListener('click', downloadCSV);
window.addEventListener('pagehide', () => {
    stopRequests();
    closeViewer();
    previewUrls.forEach(url => URL.revokeObjectURL(url));
});
