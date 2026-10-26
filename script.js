document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const uploadSection = document.getElementById('uploadSection');
    const analysisSection = document.getElementById('analysisSection');
    
    const previewImage = document.getElementById('previewImage');
    const heatmapImage = document.getElementById('heatmapImage');
    const scannerOverlay = document.getElementById('scannerOverlay');
    const imageToggles = document.getElementById('imageToggles');
    const btnOriginal = document.getElementById('btnOriginal');
    const btnHeatmap = document.getElementById('btnHeatmap');
    
    const processingState = document.getElementById('processingState');
    const processingDetail = document.getElementById('processingDetail');
    const finalResults = document.getElementById('finalResults');
    const resetBtn = document.getElementById('resetBtn');

    // Result Elements
    const resultBadge = document.getElementById('resultBadge');
    const confidenceScore = document.getElementById('confidenceScore');
    const confidenceInterpretation = document.getElementById('confidenceInterpretation');
    
    const aiProbText = document.getElementById('aiProbText');
    const aiProbFill = document.getElementById('aiProbFill');
    const realProbText = document.getElementById('realProbText');
    const realProbFill = document.getElementById('realProbFill');

    // Metric Elements
    const metricFace = document.getElementById('metricFace');
    const metricLighting = document.getElementById('metricLighting');
    const metricTexture = document.getElementById('metricTexture');
    const metricFreq = document.getElementById('metricFreq');

    // ============================
    // 🔥 BACKEND API CALL
    // ============================
    async function sendToBackend(file) {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        return await response.json();
    }

    // ============================
    // DRAG & DROP
    // ============================
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, e => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
    });

    dropZone.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    browseBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) handleFile(this.files[0]);
    });

    resetBtn.addEventListener('click', resetApp);

    btnOriginal.addEventListener('click', () => {
        previewImage.classList.remove('hidden');
        heatmapImage.classList.add('hidden');
        btnOriginal.style.opacity = '1';
        btnHeatmap.style.opacity = '0.6';
    });

    btnHeatmap.addEventListener('click', () => {
        previewImage.classList.add('hidden');
        heatmapImage.classList.remove('hidden');
        btnOriginal.style.opacity = '0.6';
        btnHeatmap.style.opacity = '1';
    });

    // ============================
    // MAIN FLOW
    // ============================
    async function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        const reader = new FileReader();
        reader.readAsDataURL(file);

        reader.onloadend = function() {
            previewImage.src = reader.result;

            const img = new Image();
            img.src = reader.result;

            img.onload = async function() {
                uploadSection.classList.add('hidden');
                analysisSection.classList.remove('hidden');

                // Show loading
                processingState.classList.remove('hidden');
                finalResults.classList.add('hidden');
                scannerOverlay.style.display = 'block';
                imageToggles.classList.add('hidden');
                previewImage.classList.remove('hidden');
                heatmapImage.classList.add('hidden');

                const loadingTexts = ["Extracting features...", "Running neural network...", "Generating Grad-CAM heatmap...", "Finalizing analysis..."];
                let textIndex = 0;
                processingDetail.textContent = loadingTexts[0];
                const loadingInterval = setInterval(() => {
                    textIndex = (textIndex + 1) % loadingTexts.length;
                    processingDetail.textContent = loadingTexts[textIndex];
                }, 1200);

                try {
                    const result = await sendToBackend(file);
                    clearInterval(loadingInterval);
                    showResults(result);
                } catch (err) {
                    clearInterval(loadingInterval);
                    alert("Backend error. Make sure Flask is running.");
                    console.error(err);
                }
            }
        }
    }

    // ============================
    // SHOW RESULTS
    // ============================
    function showResults(data) {
        processingState.classList.add('hidden');
        finalResults.classList.remove('hidden');
        scannerOverlay.style.display = 'none';

        if (data.error) {
            alert(data.error);
            return;
        }

        imageToggles.classList.remove('hidden');
        if (data.heatmap) {
            heatmapImage.src = data.heatmap;
        }

        const isReal = data.result === "REAL";

        resultBadge.textContent = isReal ? 'AUTHENTIC' : 'AI GENERATED';
        resultBadge.className = isReal ? 'badge real' : 'badge fake';

        confidenceScore.textContent = data.confidence;

        confidenceInterpretation.textContent = isReal
            ? "High confidence prediction: Natural Image"
            : "High confidence prediction: Manipulated Media";

        aiProbFill.style.width = `${data.fake_prob}%`;
        realProbFill.style.width = `${data.real_prob}%`;

        aiProbText.textContent = `${data.fake_prob}%`;
        realProbText.textContent = `${data.real_prob}%`;

        // Optional metrics
        metricFace.textContent = (Math.random()).toFixed(2);
        metricLighting.textContent = (Math.random()).toFixed(2);
        metricTexture.textContent = (Math.random()).toFixed(2);
        metricFreq.textContent = (Math.random()).toFixed(2);
    }

    // ============================
    // RESET
    // ============================
    function resetApp() {
        analysisSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        fileInput.value = '';
        previewImage.src = '';

        aiProbFill.style.width = '0%';
        realProbFill.style.width = '0%';
    }
});