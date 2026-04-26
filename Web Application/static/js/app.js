const form = document.getElementById("prediction-form");
const input = document.getElementById("image-input");
const dropzone = document.getElementById("dropzone");
const previewCard = document.getElementById("preview-card");
const previewImage = document.getElementById("preview-image");
const analyzeButton = document.getElementById("analyze-button");
const statusCard = document.getElementById("status-card");
const statusPill = document.getElementById("status-pill");
const resultTitle = document.getElementById("result-title");
const resultDescription = document.getElementById("result-description");
const detectedStage = document.getElementById("detected-stage");
const detectedConfidence = document.getElementById("detected-confidence");
const probabilityList = document.getElementById("probability-list");
const probabilityNote = document.getElementById("probability-note");

function setIdleState() {
  statusCard.className = "status-card idle";
  statusPill.textContent = "Awaiting Upload";
  resultTitle.textContent = "No prediction yet";
  resultDescription.textContent =
    "Upload a retinal image to view the model prediction, DR stage, and confidence distribution across all five classes.";
  detectedStage.textContent = "--";
  detectedConfidence.textContent = "--";
  probabilityNote.textContent = "Probabilities will appear after analysis.";
  probabilityList.innerHTML = "";
  highlightStage(null);
}

function highlightStage(activeSlug) {
  document.querySelectorAll(".stage-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.stage === activeSlug);
  });
}

function renderProbabilities(probabilities) {
  probabilityList.innerHTML = "";

  probabilities.forEach((item) => {
    const wrapper = document.createElement("article");
    wrapper.className = "probability-item";
    wrapper.innerHTML = `
      <div class="probability-row">
        <div>
          <span>${item.severity}</span>
          <strong>${item.label}</strong>
        </div>
        <strong>${item.percentage.toFixed(2)}%</strong>
      </div>
      <div class="progress-track">
        <div class="progress-bar" style="width:${item.percentage}%"></div>
      </div>
    `;
    probabilityList.appendChild(wrapper);
  });
}

function setLoadingState() {
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";
  statusCard.className = "status-card idle";
  statusPill.textContent = "Running Model";
  resultTitle.textContent = "Processing uploaded retinal image";
  resultDescription.textContent =
    "The Hybrid CNN-ViT model is evaluating the retina for diabetic retinopathy severity.";
  detectedStage.textContent = "--";
  detectedConfidence.textContent = "--";
  probabilityNote.textContent = "Model inference in progress.";
  probabilityList.innerHTML = "";
  highlightStage(null);
}

function setErrorState(message) {
  analyzeButton.disabled = false;
  analyzeButton.textContent = "Analyze Retinal Image";
  statusCard.className = "status-card danger";
  statusPill.textContent = "Prediction Error";
  resultTitle.textContent = "The image could not be analyzed";
  resultDescription.textContent = message;
  detectedStage.textContent = "--";
  detectedConfidence.textContent = "--";
  probabilityNote.textContent = "Please try another retinal image.";
  probabilityList.innerHTML = "";
  highlightStage(null);
}

function setSuccessState(payload) {
  const prediction = payload.prediction;
  const headline = prediction.has_dr
    ? `${prediction.label} detected`
    : "No diabetic retinopathy detected";

  statusCard.className = prediction.has_dr
    ? "status-card danger"
    : "status-card success";
  statusPill.textContent = prediction.has_dr ? "DR Detected" : "No DR";
  resultTitle.textContent = headline;
  resultDescription.textContent = prediction.description;
  detectedStage.textContent = prediction.severity;
  detectedConfidence.textContent = `${prediction.confidence_percentage.toFixed(2)}%`;
  probabilityNote.textContent =
    "Confidence scores from the five-class diabetic retinopathy classifier.";
  renderProbabilities(payload.probabilities);
  highlightStage(prediction.slug);

  analyzeButton.disabled = false;
  analyzeButton.textContent = "Analyze Retinal Image";
}

function showPreview(file) {
  const previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewCard.classList.add("visible");
}

function handleSelectedFiles(files) {
  if (!files || files.length === 0) {
    return;
  }

  const file = files[0];
  if (!file.type.startsWith("image/")) {
    setErrorState("Please upload a valid image file.");
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  showPreview(file);
  setIdleState();
}

input.addEventListener("change", (event) => {
  handleSelectedFiles(event.target.files);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  handleSelectedFiles(event.dataTransfer.files);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!input.files || input.files.length === 0) {
    setErrorState("Select a retinal image before starting the analysis.");
    return;
  }

  const formData = new FormData();
  formData.append("image", input.files[0]);

  setLoadingState();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed.");
    }

    setSuccessState(payload);
  } catch (error) {
    setErrorState(error.message);
  }
});

setIdleState();
