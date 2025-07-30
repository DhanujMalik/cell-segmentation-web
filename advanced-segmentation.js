// Import required libraries for image processing and ML
// Note: In a real implementation, you'd include these via CDN or npm
// <script src="https://cdn.jsdelivr.net/npm/opencv.js@4.8.0/opencv.js"></script>
// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
// Import JSZip for creating ZIP files
// <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.7.1/jszip.min.js"></script>

class AdvancedCellSegmentationTool {
  constructor() {
    // Initialize state variables similar to Python version
    this.currentStep = 0
    this.currentImage = null
    this.originalImage = null
    this.referenceImage = null
    this.processedImage = null
    this.zoomLevel = 1.0
    this.brushSize = 5
    this.currentLabel = 1
    this.currentTool = "brush"
    this.cropMode = false
    this.cropCoords = null
    this.isDrawing = false
    this.uploadedFiles = []

    // Video processing
    this.videoFrames = []
    this.currentFrameIndex = 0
    this.selectedFrameIndex = 0
    this.videoElement = null
    this.videoLoaded = false
    this.actualFPS = 30

    // Machine learning
    this.classifier = null
    this.trainingData = []
    this.trainingLabels = []
    this.featureParams = {}

    // Label system
    this.labelMask = null
    this.labelColors = {
      1: [255, 0, 0], // Red - Cell
      2: [0, 0, 0], // Black - Background
      3: [0, 255, 0], // Green - Nucleus
      4: [0, 0, 255], // Blue - Membrane
    }
    this.labelNames = {
      1: "Cell",
      2: "Background",
      3: "Nucleus",
      4: "Membrane",
    }
    this.nextLabelId = 5

    // Image enhancement parameters
    this.brightness = 1.0
    this.contrast = 1.0
    this.sharpness = 1.0
    this.denoise = false

    // Feature extraction parameters
    this.sigmaValues = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0]
    this.selectedFeatures = new Set()

    // Add these properties to the constructor
    this.videoProcessingActive = false
    this.processedFrameCount = 0
    this.totalFramesToProcess = 0
    this.outputFolderName = ""

    this.initializeElements()
    this.bindEvents()
    this.updateUI()
  }

  initializeElements() {
    // Get DOM elements
    this.fileInput = document.getElementById("file-input")
    this.mainCanvas = document.getElementById("main-canvas")
    this.labelCanvas = document.getElementById("label-canvas")
    this.canvasContainer = document.getElementById("canvas-container")
    this.uploadPlaceholder = document.getElementById("upload-placeholder")
    this.cropOverlay = document.getElementById("crop-overlay")

    // Get canvas contexts
    this.mainCtx = this.mainCanvas.getContext("2d")
    this.labelCtx = this.labelCanvas.getContext("2d")

    // Status elements
    this.statusElements = {
      zoom: document.getElementById("zoom-status"),
      tool: document.getElementById("tool-status"),
      brush: document.getElementById("brush-status"),
      label: document.getElementById("label-status"),
      file: document.getElementById("file-status"),
      progress: document.getElementById("progress-container"),
      progressFill: document.getElementById("progress-fill"),
      progressText: document.getElementById("progress-text"),
    }

    // Control elements
    this.controls = {
      brushSize: document.getElementById("brush-size"),
      brushSizeValue: document.getElementById("brush-size-value"),
      currentLabel: document.getElementById("current-label"),
      enableCrop: document.getElementById("enable-crop"),
      cropControls: document.getElementById("crop-controls"),
    }
  }

  bindEvents() {
    // File input events
    document.getElementById("browse-btn").addEventListener("click", () => {
      this.fileInput.click()
    })

    document.getElementById("reset-btn").addEventListener("click", () => {
      this.resetInput()
    })

    this.fileInput.addEventListener("change", (e) => {
      this.handleFileUpload(e)
    })

    // Tool events
    document.getElementById("brush-tool").addEventListener("click", () => {
      this.setTool("brush")
    })

    document.getElementById("erase-tool").addEventListener("click", () => {
      this.setTool("erase")
    })

    document.getElementById("crop-tool").addEventListener("click", () => {
      this.setTool("crop")
    })

    // Canvas events for interactive labeling
    this.labelCanvas.addEventListener("mousedown", (e) => {
      this.handleCanvasMouseDown(e)
    })

    this.labelCanvas.addEventListener("mousemove", (e) => {
      this.handleCanvasMouseMove(e)
    })

    this.labelCanvas.addEventListener("mouseup", (e) => {
      this.handleCanvasMouseUp(e)
    })

    // Brush size control
    this.controls.brushSize.addEventListener("input", (e) => {
      this.brushSize = Number.parseInt(e.target.value)
      this.controls.brushSizeValue.textContent = this.brushSize
      this.updateStatusBar()
    })

    // Label selection
    this.controls.currentLabel.addEventListener("change", (e) => {
      this.currentLabel = Number.parseInt(e.target.value)
      this.updateStatusBar()
    })

    // Processing buttons
    document.getElementById("clear-labels").addEventListener("click", () => {
      this.clearLabels()
    })

    document.getElementById("train-classifier").addEventListener("click", () => {
      this.trainClassifier()
    })

    document.getElementById("apply-features").addEventListener("click", () => {
      this.applyFeatures()
    })

    document.getElementById("run-segmentation").addEventListener("click", () => {
      this.runSegmentation()
    })

    // Feature selection checkboxes
    document.querySelectorAll('input[name="features"]').forEach((checkbox) => {
      checkbox.addEventListener("change", (e) => {
        if (e.target.checked) {
          this.selectedFeatures.add(e.target.value)
        } else {
          this.selectedFeatures.delete(e.target.value)
        }
      })
    })

    // Zoom controls
    document.getElementById("zoom-in").addEventListener("click", () => {
      this.adjustZoom(1.25)
    })

    document.getElementById("zoom-out").addEventListener("click", () => {
      this.adjustZoom(0.8)
    })

    document.getElementById("reset-zoom").addEventListener("click", () => {
      this.resetZoom()
    })

    // Video processing events
    document.getElementById("process-all-frames").addEventListener("click", () => {
      this.processAllVideoFrames()
    })

    document.getElementById("stop-processing").addEventListener("click", () => {
      this.stopVideoProcessing()
    })

    // Update UI state when classifier is trained
    const originalTrainClassifier = this.trainClassifier.bind(this)
    this.trainClassifier = async function () {
      await originalTrainClassifier()
      // Enable video processing button if video frames are loaded
      if (this.videoFrames && this.videoFrames.length > 0) {
        document.getElementById("process-all-frames").disabled = false
      }
    }

    // Update UI state when video frames are loaded
    const originalLoadVideoFrames = this.loadVideoFrames.bind(this)
    this.loadVideoFrames = async function (...args) {
      await originalLoadVideoFrames(...args)
      // Enable processing button if classifier is trained
      if (this.classifier && this.classifier.trained) {
        document.getElementById("process-all-frames").disabled = false
      }
    }
  }

  async handleFileUpload(event) {
    const files = Array.from(event.target.files)
    this.uploadedFiles = files
    const inputType = document.querySelector('input[name="inputType"]:checked').value

    if (files.length > 0) {
      this.statusElements.file.textContent = `${files.length} file(s) selected`

      if (inputType === "video") {
        await this.setupVideoElement(files[0])
      } else if (inputType === "folder") {
        await this.loadImage(files[0]) // Load first image as example
        this.setCurrentStep(1)
      } else {
        // Single image
        await this.loadImage(files[0])
        this.setCurrentStep(1)
      }
    }
  }

  async loadImage(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const img = new Image()
        img.onload = () => {
          this.originalImage = img
          this.currentImage = img
          this.drawImageOnCanvas()
          this.initializeLabelMask()
          this.uploadPlaceholder.style.display = "none"
          resolve()
        }
        img.onerror = reject
        img.src = e.target.result
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  drawImageOnCanvas() {
    if (!this.currentImage) return

    const width = this.currentImage.width * this.zoomLevel
    const height = this.currentImage.height * this.zoomLevel

    this.mainCanvas.width = width
    this.mainCanvas.height = height

    this.mainCtx.clearRect(0, 0, width, height)
    this.mainCtx.drawImage(this.currentImage, 0, 0, width, height)
  }

  initializeLabelMask() {
    if (!this.currentImage) return

    const width = this.currentImage.width
    const height = this.currentImage.height

    // Create label mask as ImageData
    this.labelMask = new ImageData(width, height)

    // Initialize label canvas
    this.labelCanvas.width = width * this.zoomLevel
    this.labelCanvas.height = height * this.zoomLevel
    this.labelCtx.clearRect(0, 0, this.labelCanvas.width, this.labelCanvas.height)
  }

  handleCanvasMouseDown(event) {
    const rect = this.labelCanvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    if (this.currentTool === "crop") {
      this.startCropSelection(x, y)
    } else {
      this.isDrawing = true
      this.paintLabel(x, y)
    }
  }

  handleCanvasMouseMove(event) {
    const rect = this.labelCanvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    if (this.currentTool === "crop" && this.cropMode) {
      this.updateCropSelection(x, y)
    } else if (this.isDrawing) {
      this.paintLabel(x, y)
    }
  }

  handleCanvasMouseUp(event) {
    if (this.currentTool === "crop" && this.cropMode) {
      this.finishCropSelection()
    } else {
      this.isDrawing = false
    }
  }

  paintLabel(canvasX, canvasY) {
    if (!this.currentImage || !this.labelMask) return

    // Convert canvas coordinates to image coordinates
    const imgX = Math.floor(canvasX / this.zoomLevel)
    const imgY = Math.floor(canvasY / this.zoomLevel)

    // Bounds checking
    if (imgX < 0 || imgX >= this.currentImage.width || imgY < 0 || imgY >= this.currentImage.height) return

    const halfSize = Math.floor(this.brushSize / 2)
    const labelValue = this.currentTool === "erase" ? 0 : this.currentLabel

    // Paint in circular brush pattern
    for (let dy = -halfSize; dy <= halfSize; dy++) {
      for (let dx = -halfSize; dx <= halfSize; dx++) {
        const x = imgX + dx
        const y = imgY + dy

        if (x >= 0 && x < this.currentImage.width && y >= 0 && y < this.currentImage.height) {
          const distance = Math.sqrt(dx * dx + dy * dy)
          if (distance <= halfSize) {
            const index = (y * this.currentImage.width + x) * 4

            // Store label in alpha channel of ImageData
            this.labelMask.data[index + 3] = labelValue
          }
        }
      }
    }

    this.drawLabelOverlay()
  }

  drawLabelOverlay() {
    if (!this.labelMask) return

    // Create overlay canvas
    const overlayCanvas = document.createElement("canvas")
    overlayCanvas.width = this.currentImage.width
    overlayCanvas.height = this.currentImage.height
    const overlayCtx = overlayCanvas.getContext("2d")

    const overlayData = overlayCtx.createImageData(this.currentImage.width, this.currentImage.height)

    // Draw labels with transparency
    for (let i = 0; i < this.labelMask.data.length; i += 4) {
      const label = this.labelMask.data[i + 3]
      if (label > 0 && this.labelColors[label]) {
        const color = this.labelColors[label]
        overlayData.data[i] = color[0] // R
        overlayData.data[i + 1] = color[1] // G
        overlayData.data[i + 2] = color[2] // B
        overlayData.data[i + 3] = 128 // A (50% transparency)
      }
    }

    overlayCtx.putImageData(overlayData, 0, 0)

    // Draw overlay on label canvas
    this.labelCtx.clearRect(0, 0, this.labelCanvas.width, this.labelCanvas.height)
    this.labelCtx.drawImage(overlayCanvas, 0, 0, this.labelCanvas.width, this.labelCanvas.height)
  }

  clearLabels() {
    if (this.labelMask) {
      // Clear all label data
      for (let i = 3; i < this.labelMask.data.length; i += 4) {
        this.labelMask.data[i] = 0
      }
      this.drawLabelOverlay()
    }
  }

  async trainClassifier() {
    if (!this.currentImage || !this.labelMask) {
      alert("Please load an image and add some labels first")
      return
    }

    // Check if we have labeled pixels
    let hasLabels = false
    for (let i = 3; i < this.labelMask.data.length; i += 4) {
      if (this.labelMask.data[i] > 0) {
        hasLabels = true
        break
      }
    }

    if (!hasLabels) {
      alert("Please label some pixels first")
      return
    }

    try {
      this.showProgress("Training classifier...")

      // Extract features from the current image
      const features = await this.extractFeatures(this.currentImage)

      // Prepare training data
      const trainingData = []
      const trainingLabels = []

      for (let y = 0; y < this.currentImage.height; y++) {
        for (let x = 0; x < this.currentImage.width; x++) {
          const index = (y * this.currentImage.width + x) * 4
          const label = this.labelMask.data[index + 3]

          if (label > 0) {
            const pixelFeatures = this.getPixelFeatures(features, x, y)
            trainingData.push(pixelFeatures)
            trainingLabels.push(label)
          }
        }
      }

      if (trainingData.length === 0) {
        throw new Error("No labeled pixels found")
      }

      // Train a simple classifier (in a real implementation, you'd use a proper ML library)
      this.classifier = {
        trainingData: trainingData,
        trainingLabels: trainingLabels,
        trained: true,
      }

      this.hideProgress()
      alert("Classifier trained successfully!")
      this.setCurrentStep(4)
    } catch (error) {
      this.hideProgress()
      console.error("Training failed:", error)
      alert(`Training failed: ${error.message}`)
    }
  }

  async extractFeatures(image) {
    // Convert image to grayscale for feature extraction
    const canvas = document.createElement("canvas")
    canvas.width = image.width
    canvas.height = image.height
    const ctx = canvas.getContext("2d")

    ctx.drawImage(image, 0, 0)
    const imageData = ctx.getImageData(0, 0, image.width, image.height)

    // Convert to grayscale
    const grayData = new Float32Array(image.width * image.height)
    for (let i = 0; i < imageData.data.length; i += 4) {
      const gray = 0.299 * imageData.data[i] + 0.587 * imageData.data[i + 1] + 0.114 * imageData.data[i + 2]
      grayData[i / 4] = gray / 255.0
    }

    const features = {}

    // Raw intensity
    if (this.selectedFeatures.has("raw")) {
      features.raw = grayData
    }

    // Gaussian smoothing
    if (this.selectedFeatures.has("gaussian")) {
      features.gaussian = this.applyGaussianFilter(grayData, image.width, image.height, 1.0)
    }

    // Laplacian
    if (this.selectedFeatures.has("laplacian")) {
      features.laplacian = this.applyLaplacianFilter(grayData, image.width, image.height)
    }

    // Gradient magnitude
    if (this.selectedFeatures.has("gradient")) {
      features.gradient = this.applyGradientFilter(grayData, image.width, image.height)
    }

    // Sobel edge detection
    if (this.selectedFeatures.has("sobel")) {
      features.sobel = this.applySobelFilter(grayData, image.width, image.height)
    }

    return features
  }

  applyGaussianFilter(data, width, height, sigma) {
    // Simple Gaussian blur implementation
    const result = new Float32Array(data.length)
    const kernelSize = Math.ceil(sigma * 3) * 2 + 1
    const kernel = this.createGaussianKernel(kernelSize, sigma)

    // Apply separable Gaussian filter
    const temp = new Float32Array(data.length)

    // Horizontal pass
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0
        let weightSum = 0

        for (let i = 0; i < kernelSize; i++) {
          const xi = x + i - Math.floor(kernelSize / 2)
          if (xi >= 0 && xi < width) {
            sum += data[y * width + xi] * kernel[i]
            weightSum += kernel[i]
          }
        }

        temp[y * width + x] = sum / weightSum
      }
    }

    // Vertical pass
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0
        let weightSum = 0

        for (let i = 0; i < kernelSize; i++) {
          const yi = y + i - Math.floor(kernelSize / 2)
          if (yi >= 0 && yi < height) {
            sum += temp[yi * width + x] * kernel[i]
            weightSum += kernel[i]
          }
        }

        result[y * width + x] = sum / weightSum
      }
    }

    return result
  }

  createGaussianKernel(size, sigma) {
    const kernel = new Float32Array(size)
    const center = Math.floor(size / 2)
    let sum = 0

    for (let i = 0; i < size; i++) {
      const x = i - center
      kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma))
      sum += kernel[i]
    }

    // Normalize
    for (let i = 0; i < size; i++) {
      kernel[i] /= sum
    }

    return kernel
  }

  applyLaplacianFilter(data, width, height) {
    const result = new Float32Array(data.length)
    const kernel = [0, -1, 0, -1, 4, -1, 0, -1, 0]

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let sum = 0
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const idx = (y + ky) * width + (x + kx)
            sum += data[idx] * kernel[(ky + 1) * 3 + (kx + 1)]
          }
        }
        result[y * width + x] = Math.abs(sum)
      }
    }

    return result
  }

  applyGradientFilter(data, width, height) {
    const result = new Float32Array(data.length)

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const gx = data[y * width + (x + 1)] - data[y * width + (x - 1)]
        const gy = data[(y + 1) * width + x] - data[(y - 1) * width + x]
        result[y * width + x] = Math.sqrt(gx * gx + gy * gy)
      }
    }

    return result
  }

  applySobelFilter(data, width, height) {
    const result = new Float32Array(data.length)
    const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1]

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let gx = 0,
          gy = 0

        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const idx = (y + ky) * width + (x + kx)
            const kernelIdx = (ky + 1) * 3 + (kx + 1)
            gx += data[idx] * sobelX[kernelIdx]
            gy += data[idx] * sobelY[kernelIdx]
          }
        }

        result[y * width + x] = Math.sqrt(gx * gx + gy * gy)
      }
    }

    return result
  }

  getPixelFeatures(features, x, y) {
    const pixelFeatures = []
    const width = this.currentImage.width
    const index = y * width + x

    for (const [featureName, featureData] of Object.entries(features)) {
      if (featureData && index < featureData.length) {
        pixelFeatures.push(featureData[index])
      }
    }

    return pixelFeatures
  }

  async runSegmentation() {
    if (!this.classifier || !this.classifier.trained) {
      alert("Please train the classifier first")
      return
    }

    try {
      this.showProgress("Running segmentation...")

      // Extract features from current image
      const features = await this.extractFeatures(this.currentImage)

      // Segment the image
      const segmentedData = await this.segmentImage(features)

      // Display result
      this.displaySegmentationResult(segmentedData)

      this.hideProgress()
      alert("Segmentation completed!")
    } catch (error) {
      this.hideProgress()
      console.error("Segmentation failed:", error)
      alert(`Segmentation failed: ${error.message}`)
    }
  }

  async segmentImage(features) {
    const width = this.currentImage.width
    const height = this.currentImage.height
    const segmented = new Uint8Array(width * height)

    // Simple nearest neighbor classification
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelFeatures = this.getPixelFeatures(features, x, y)
        const label = this.classifyPixel(pixelFeatures)
        segmented[y * width + x] = label
      }
    }

    return segmented
  }

  classifyPixel(pixelFeatures) {
    if (!this.classifier || !this.classifier.trained) {
      return 2 // Default to background
    }

    // Simple nearest neighbor classification
    let minDistance = Number.POSITIVE_INFINITY
    let bestLabel = 2

    for (let i = 0; i < this.classifier.trainingData.length; i++) {
      const trainFeatures = this.classifier.trainingData[i]
      let distance = 0

      for (let j = 0; j < Math.min(pixelFeatures.length, trainFeatures.length); j++) {
        const diff = pixelFeatures[j] - trainFeatures[j]
        distance += diff * diff
      }

      if (distance < minDistance) {
        minDistance = distance
        bestLabel = this.classifier.trainingLabels[i]
      }
    }

    return bestLabel
  }

  displaySegmentationResult(segmentedData) {
    const canvas = document.createElement("canvas")
    canvas.width = this.currentImage.width
    canvas.height = this.currentImage.height
    const ctx = canvas.getContext("2d")

    const imageData = ctx.createImageData(canvas.width, canvas.height)

    // Convert segmentation to RGB
    for (let i = 0; i < segmentedData.length; i++) {
      const label = segmentedData[i]
      const color = this.labelColors[label] || [128, 128, 128]

      const pixelIndex = i * 4
      imageData.data[pixelIndex] = color[0] // R
      imageData.data[pixelIndex + 1] = color[1] // G
      imageData.data[pixelIndex + 2] = color[2] // B
      imageData.data[pixelIndex + 3] = 255 // A
    }

    ctx.putImageData(imageData, 0, 0)

    // Display on main canvas
    this.mainCtx.clearRect(0, 0, this.mainCanvas.width, this.mainCanvas.height)
    this.mainCtx.drawImage(canvas, 0, 0, this.mainCanvas.width, this.mainCanvas.height)
  }

  applyFeatures() {
    if (!this.currentImage) {
      alert("Please load an image first")
      return
    }

    if (this.selectedFeatures.size === 0) {
      alert("Please select at least one feature")
      return
    }

    this.showProgress("Applying features...")

    setTimeout(async () => {
      try {
        const features = await this.extractFeatures(this.currentImage)

        // Display first selected feature as preview
        const firstFeature = this.selectedFeatures.values().next().value
        if (features[firstFeature]) {
          this.displayFeaturePreview(features[firstFeature])
        }

        this.hideProgress()
        this.setCurrentStep(3)
      } catch (error) {
        this.hideProgress()
        console.error("Feature application failed:", error)
        alert(`Feature application failed: ${error.message}`)
      }
    }, 100)
  }

  displayFeaturePreview(featureData) {
    const canvas = document.createElement("canvas")
    canvas.width = this.currentImage.width
    canvas.height = this.currentImage.height
    const ctx = canvas.getContext("2d")

    const imageData = ctx.createImageData(canvas.width, canvas.height)

    // Normalize feature data to 0-255 range
    let min = Number.POSITIVE_INFINITY,
      max = Number.NEGATIVE_INFINITY
    for (let i = 0; i < featureData.length; i++) {
      min = Math.min(min, featureData[i])
      max = Math.max(max, featureData[i])
    }

    const range = max - min
    for (let i = 0; i < featureData.length; i++) {
      const normalized = range > 0 ? ((featureData[i] - min) / range) * 255 : 0
      const pixelIndex = i * 4

      imageData.data[pixelIndex] = normalized // R
      imageData.data[pixelIndex + 1] = normalized // G
      imageData.data[pixelIndex + 2] = normalized // B
      imageData.data[pixelIndex + 3] = 255 // A
    }

    ctx.putImageData(imageData, 0, 0)

    // Display on main canvas
    this.mainCtx.clearRect(0, 0, this.mainCanvas.width, this.mainCanvas.height)
    this.mainCtx.drawImage(canvas, 0, 0, this.mainCanvas.width, this.mainCanvas.height)
  }

  setTool(tool) {
    this.currentTool = tool

    // Update tool button states
    document.querySelectorAll(".btn-tool").forEach((btn) => {
      btn.classList.remove("active")
    })

    document.getElementById(tool + "-tool").classList.add("active")
    this.updateStatusBar()
  }

  adjustZoom(factor) {
    this.zoomLevel *= factor
    this.zoomLevel = Math.max(0.1, Math.min(5, this.zoomLevel))

    if (this.currentImage) {
      this.drawImageOnCanvas()
      this.drawLabelOverlay()
    }

    this.updateStatusBar()
  }

  resetZoom() {
    this.zoomLevel = 1.0

    if (this.currentImage) {
      this.drawImageOnCanvas()
      this.drawLabelOverlay()
    }

    this.updateStatusBar()
  }

  resetInput() {
    this.uploadedFiles = []
    this.currentImage = null
    this.originalImage = null
    this.labelMask = null
    this.classifier = null
    this.statusElements.file.textContent = "No files selected"
    this.fileInput.value = ""

    if (this.mainCanvas) {
      this.mainCtx.clearRect(0, 0, this.mainCanvas.width, this.mainCanvas.height)
    }

    if (this.labelCanvas) {
      this.labelCtx.clearRect(0, 0, this.labelCanvas.width, this.labelCanvas.height)
    }

    this.uploadPlaceholder.style.display = "flex"
    this.setCurrentStep(0)
  }

  setCurrentStep(step) {
    this.currentStep = step
    this.updateUI()
  }

  updateUI() {
    const cards = ["input-card", "preprocess-card", "labeling-card", "features-card", "processing-card"]

    cards.forEach((cardId, index) => {
      const card = document.getElementById(cardId)
      if (index <= this.currentStep) {
        card.classList.remove("disabled")
      } else {
        card.classList.add("disabled")
      }
    })

    this.updateStatusBar()
  }

  updateStatusBar() {
    if (this.statusElements.zoom) {
      this.statusElements.zoom.textContent = `Zoom: ${Math.round(this.zoomLevel * 100)}%`
    }
    if (this.statusElements.tool) {
      this.statusElements.tool.textContent = `Tool: ${this.currentTool}`
    }
    if (this.statusElements.brush) {
      this.statusElements.brush.textContent = `Brush Size: ${this.brushSize}`
    }
    if (this.statusElements.label) {
      const labelName = this.labelNames[this.currentLabel] || "Unknown"
      this.statusElements.label.textContent = `Current Label: ${labelName}`
    }
  }

  showProgress(message) {
    if (this.statusElements.progress) {
      this.statusElements.progress.style.display = "block"
      this.statusElements.progressFill.style.width = "0%"
      this.statusElements.progressText.textContent = message
    }

    // Also update video processing progress
    const frameProgressText = document.getElementById("frame-progress-text")
    if (frameProgressText) {
      frameProgressText.textContent = message
    }

    // Enable stop button during processing
    const stopBtn = document.getElementById("stop-processing")
    if (stopBtn) {
      stopBtn.disabled = false
    }
  }

  hideProgress() {
    if (this.statusElements.progress) {
      setTimeout(() => {
        this.statusElements.progress.style.display = "none"
      }, 1000)
    }

    // Reset video processing progress
    const frameProgressText = document.getElementById("frame-progress-text")
    const frameProgressFill = document.getElementById("frame-progress-fill")
    if (frameProgressText) {
      frameProgressText.textContent = "Processing completed"
    }
    if (frameProgressFill) {
      frameProgressFill.style.width = "100%"
    }

    // Disable stop button
    const stopBtn = document.getElementById("stop-processing")
    if (stopBtn) {
      stopBtn.disabled = true
    }
  }

  // Video processing methods (simplified versions of Python equivalents)
  async setupVideoElement(file) {
    // Similar to Python version but using HTML5 video
    if (this.videoElement) {
      URL.revokeObjectURL(this.videoElement.src)
      this.videoElement.remove()
    }

    this.videoElement = document.createElement("video")
    this.videoElement.style.display = "none"
    this.videoElement.preload = "metadata"
    this.videoElement.crossOrigin = "anonymous"
    this.videoElement.muted = true
    document.body.appendChild(this.videoElement)

    const url = URL.createObjectURL(file)
    this.videoElement.src = url

    return new Promise((resolve, reject) => {
      this.videoElement.addEventListener("loadedmetadata", () => {
        this.videoLoaded = true
        this.actualFPS = 30 // Default, could be extracted from video
        this.statusElements.file.textContent = `Video loaded: ${file.name} (${this.videoElement.duration.toFixed(1)}s)`
        resolve()
      })

      this.videoElement.addEventListener("error", (e) => {
        console.error("Video loading error:", e)
        reject(new Error("Error loading video"))
      })
    })
  }

  async processAllVideoFrames() {
    if (!this.videoFrames || this.videoFrames.length === 0) {
      alert("No video frames loaded")
      return
    }

    if (!this.classifier || !this.classifier.trained) {
      alert("Please train the classifier first using a reference frame")
      return
    }

    try {
      // Create output folder name with timestamp
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5)
      this.outputFolderName = `segmented_frames_${timestamp}`

      this.videoProcessingActive = true
      this.processedFrameCount = 0
      this.totalFramesToProcess = this.videoFrames.length

      this.showProgress(`Processing ${this.totalFramesToProcess} video frames...`)
      this.updateProgressBar(0)

      // Create array to store all segmented frames
      const segmentedFrames = []

      // Process each frame
      for (let i = 0; i < this.videoFrames.length; i++) {
        if (!this.videoProcessingActive) {
          break // Stop if user cancelled
        }

        const frameData = this.videoFrames[i]
        const frameImage = frameData[1] // Get the image data
        const frameName = frameData[2] // Get the frame name
        const timestamp = frameData[3] // Get the timestamp

        // Create canvas for this frame
        const frameCanvas = document.createElement("canvas")
        frameCanvas.width = frameImage.width || this.currentImage.width
        frameCanvas.height = frameImage.height || this.currentImage.height
        const frameCtx = frameCanvas.getContext("2d")

        // Draw frame to canvas
        if (frameImage instanceof HTMLImageElement) {
          frameCtx.drawImage(frameImage, 0, 0)
        } else {
          // Handle ImageData or other formats
          const tempImg = new Image()
          tempImg.onload = () => frameCtx.drawImage(tempImg, 0, 0)
          tempImg.src = this.arrayToDataURL(frameImage)
        }

        try {
          let segmentedResult

          if (this.useBackend && this.classifier.backend) {
            // Use Python backend for processing
            segmentedResult = await this.backend.processVideoFrame(frameCanvas, this.selectedFeatures)
          } else {
            // Use frontend processing
            segmentedResult = await this.processFrameLocally(frameCanvas)
          }

          // Store the segmented frame
          segmentedFrames.push({
            name: frameName.replace(".png", "_segmented.png"),
            data: segmentedResult,
            originalName: frameName,
            timestamp: timestamp,
            frameIndex: i,
          })

          this.processedFrameCount++
          const progress = (this.processedFrameCount / this.totalFramesToProcess) * 100
          this.updateProgressBar(progress)
          this.updateProgressText(`Processed ${this.processedFrameCount}/${this.totalFramesToProcess} frames`)

          // Small delay to prevent UI blocking
          await new Promise((resolve) => setTimeout(resolve, 10))
        } catch (error) {
          console.error(`Error processing frame ${i}:`, error)
          // Continue with next frame
        }
      }

      if (this.videoProcessingActive) {
        // Create and download the results
        await this.createOutputFolder(segmentedFrames)
        this.hideProgress()
        alert(`Video processing completed! ${segmentedFrames.length} frames processed.`)
      } else {
        this.hideProgress()
        alert("Video processing cancelled.")
      }
    } catch (error) {
      this.hideProgress()
      console.error("Video processing failed:", error)
      alert(`Video processing failed: ${error.message}`)
    } finally {
      this.videoProcessingActive = false
    }
  }

  async processFrameLocally(frameCanvas) {
    // Extract features from the frame
    const features = await this.extractFeatures(frameCanvas)

    // Segment the frame
    const segmentedData = await this.segmentImage(features)

    // Convert to binary output
    const binaryMask = this.convertToBinaryMask(segmentedData)

    // Convert to base64 for storage
    const resultCanvas = document.createElement("canvas")
    resultCanvas.width = frameCanvas.width
    resultCanvas.height = frameCanvas.height
    const resultCtx = resultCanvas.getContext("2d")

    const imageData = resultCtx.createImageData(resultCanvas.width, resultCanvas.height)

    for (let i = 0; i < binaryMask.length; i++) {
      const pixelIndex = i * 4
      const value = binaryMask[i]
      imageData.data[pixelIndex] = value // R
      imageData.data[pixelIndex + 1] = value // G
      imageData.data[pixelIndex + 2] = value // B
      imageData.data[pixelIndex + 3] = 255 // A
    }

    resultCtx.putImageData(imageData, 0, 0)
    return resultCanvas.toDataURL("image/png").split(",")[1] // Return base64 without prefix
  }

  convertToBinaryMask(segmentedData) {
    const binaryMask = new Uint8Array(segmentedData.length)

    for (let i = 0; i < segmentedData.length; i++) {
      const label = segmentedData[i]
      // Convert labels to binary: cells=255, background=0
      if (label === 1 || label === 3 || label === 4) {
        // Cell, Nucleus, Membrane
        binaryMask[i] = 255
      } else {
        binaryMask[i] = 0
      }
    }

    return binaryMask
  }

  async createOutputFolder(segmentedFrames) {
    // Create a ZIP file containing all segmented frames
    const JSZip = window.JSZip // Declare JSZip variable
    if (typeof JSZip !== "undefined") {
      const zip = new JSZip()
      const folder = zip.folder(this.outputFolderName)

      // Add each segmented frame to the ZIP
      for (const frame of segmentedFrames) {
        folder.file(frame.name, frame.data, { base64: true })
      }

      // Add metadata file
      const metadata = {
        processing_date: new Date().toISOString(),
        total_frames: segmentedFrames.length,
        selected_features: Array.from(this.selectedFeatures),
        label_types: this.labelNames,
        video_info: {
          fps: this.actualFPS,
          total_original_frames: this.videoFrames.length,
        },
      }

      folder.file("processing_metadata.json", JSON.stringify(metadata, null, 2))

      // Generate and download ZIP
      const content = await zip.generateAsync({ type: "blob" })
      this.downloadBlob(content, `${this.outputFolderName}.zip`)
    } else {
      // Fallback: download frames individually
      for (const frame of segmentedFrames) {
        const blob = this.base64ToBlob(frame.data, "image/png")
        this.downloadBlob(blob, frame.name)
        await new Promise((resolve) => setTimeout(resolve, 100)) // Small delay between downloads
      }
    }
  }

  base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64)
    const byteNumbers = new Array(byteCharacters.length)

    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i)
    }

    const byteArray = new Uint8Array(byteNumbers)
    return new Blob([byteArray], { type: mimeType })
  }

  downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  arrayToDataURL(imageArray) {
    const canvas = document.createElement("canvas")
    canvas.width = imageArray.width
    canvas.height = imageArray.height
    const ctx = canvas.getContext("2d")

    if (imageArray instanceof ImageData) {
      ctx.putImageData(imageArray, 0, 0)
    } else {
      // Handle other array formats
      const imageData = ctx.createImageData(canvas.width, canvas.height)
      imageData.data.set(imageArray)
      ctx.putImageData(imageData, 0, 0)
    }

    return canvas.toDataURL()
  }

  stopVideoProcessing() {
    this.videoProcessingActive = false
    this.hideProgress()
  }

  updateProgressBar(percentage) {
    if (this.statusElements.progressFill) {
      this.statusElements.progressFill.style.width = `${percentage}%`
    }
  }

  updateProgressText(text) {
    if (this.statusElements.progressText) {
      this.statusElements.progressText.textContent = text
    }
  }

  // Update the load_video_frames method to store frames properly
  async loadVideoFrames(frameInterval = 1, maxFrames = 50) {
    try {
      if (!this.videoElement) {
        throw new Error("Video not loaded")
      }

      this.videoFrames = []
      const canvas = document.createElement("canvas")
      const ctx = canvas.getContext("2d")

      const duration = this.videoElement.duration
      const totalFrames = Math.floor(duration * this.actualFPS)
      const actualMaxFrames = Math.min(maxFrames, Math.floor(totalFrames / frameInterval))

      canvas.width = this.videoElement.videoWidth
      canvas.height = this.videoElement.videoHeight

      this.showProgress(`Extracting ${actualMaxFrames} frames from video...`)

      for (let i = 0; i < actualMaxFrames; i++) {
        const timePosition = (i * frameInterval) / this.actualFPS

        if (timePosition >= duration) break

        // Seek to specific time
        this.videoElement.currentTime = timePosition

        // Wait for seek to complete
        await new Promise((resolve) => {
          const onSeeked = () => {
            this.videoElement.removeEventListener("seeked", onSeeked)
            resolve()
          }
          this.videoElement.addEventListener("seeked", onSeeked)
        })

        // Draw frame to canvas
        ctx.drawImage(this.videoElement, 0, 0)

        // Create image from canvas
        const frameImage = new Image()
        frameImage.src = canvas.toDataURL()

        await new Promise((resolve) => {
          frameImage.onload = resolve
        })

        // Store frame data
        const frameData = [
          i * frameInterval, // frame number
          frameImage, // image data
          `frame_${String(i * frameInterval).padStart(4, "0")}.png`, // filename
          timePosition, // timestamp
        ]

        this.videoFrames.push(frameData)

        // Update progress
        const progress = ((i + 1) / actualMaxFrames) * 100
        this.updateProgressBar(progress)
        this.updateProgressText(`Extracted ${i + 1}/${actualMaxFrames} frames`)
      }

      this.hideProgress()

      if (this.videoFrames.length === 0) {
        throw new Error("No frames extracted from video")
      }

      this.showFrameSelector()
      // Removed tk.NORMAL as it's not used in JavaScript
    } catch (error) {
      this.hideProgress()
      console.error("Video frame extraction failed:", error)
      alert(`Failed to extract video frames: ${error.message}`)
    }
  }
}

// Initialize the application when the page loads
document.addEventListener("DOMContentLoaded", () => {
  new AdvancedCellSegmentationTool()
})
