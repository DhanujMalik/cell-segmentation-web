document.addEventListener("DOMContentLoaded", () => {
  // Initialize the application
  const app = new AdvancedCellSegmentationTool()
})

class AdvancedCellSegmentationTool {
  constructor() {
    // Initialize state variables
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
      2: [0, 255, 0], // Green - Background
      3: [0, 0, 255], // Blue - Nucleus
      4: [255, 255, 0], // Yellow - Membrane
    }
    this.labelNames = {
      1: "Cell",
      2: "Background",
      3: "Nucleus",
      4: "Membrane",
    }
    this.nextLabelId = 5

    // Feature extraction parameters
    this.sigmaValues = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0]
    this.selectedFeatures = new Set()

    // Video processing
    this.videoProcessingActive = false
    this.processedFrameCount = 0
    this.totalFramesToProcess = 0
    this.outputFolderName = ""

    // Batch processing
    this.outputFolder = null
    this.batchQueue = []
    this.batchRunning = false
    this.batchProcessed = 0
    this.batchTotal = 0

    // UI state management
    this.stepStates = {
      0: "input-card",
      1: "preprocess-card",
      2: "labeling-card",
      3: "features-card",
      4: "processing-card",
    }

    // Initialize UI
    this.initializeElements()
    this.bindEvents()
    this.initializeTabs()

    // Initialize label preview
    setTimeout(() => {
      this.updateLabelPreview()
    }, 100)
  }

  initializeElements() {
    // Get DOM elements
    this.fileInput = document.getElementById("file-input")
    this.mainCanvas = document.getElementById("main-canvas")
    this.labelCanvas = document.getElementById("label-canvas")
    this.canvasContainer = document.getElementById("canvas-container")
    this.uploadPlaceholder = document.getElementById("upload-placeholder")
    this.cropOverlay = document.getElementById("crop-overlay")
    this.previewCanvas = document.getElementById("preview-canvas")

    // Get canvas contexts
    this.mainCtx = this.mainCanvas.getContext("2d")
    this.labelCtx = this.labelCanvas.getContext("2d")
    this.previewCtx = this.previewCanvas.getContext("2d")

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
      frameProgress: document.getElementById("frame-progress-fill"),
      frameProgressText: document.getElementById("frame-progress-text"),
    }

    // Control elements
    this.controls = {
      brushSize: document.getElementById("brush-size"),
      brushSizeValue: document.getElementById("brush-size-value"),
      currentLabel: document.getElementById("current-label"),
      enableCrop: document.getElementById("enable-crop"),
      cropControls: document.getElementById("crop-controls"),
      videoSettings: document.getElementById("video-settings"),
      frameSelector: document.getElementById("frame-selector"),
      videoBatchControls: document.getElementById("video-batch-controls"),
    }

    // Label preview elements
    this.labelPreviewCanvas = document.getElementById("label-preview")
    this.labelPreviewCtx = this.labelPreviewCanvas.getContext("2d")
    this.currentLabelText = document.getElementById("current-label-text")

    // Batch processing elements
    this.outputFolderStatus = document.getElementById("output-folder-status")
    this.batchStatus = document.getElementById("batch-status")
  }

  bindEvents() {
    // Input selection events
    document.querySelectorAll('input[name="inputType"]').forEach((radio) => {
      radio.addEventListener("change", () => this.handleInputTypeChange())
    })

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

    // Video settings events
    document.querySelectorAll('input[name="videoMode"]').forEach((radio) => {
      radio.addEventListener("change", () => this.handleVideoModeChange())
    })

    document.getElementById("load-video-frames").addEventListener("click", () => {
      this.loadVideoFrames()
    })

    // Frame selector events
    document.getElementById("frame-prev-10").addEventListener("click", () => {
      this.navigateFrames(-10)
    })

    document.getElementById("frame-prev").addEventListener("click", () => {
      this.navigateFrames(-1)
    })

    document.getElementById("frame-next").addEventListener("click", () => {
      this.navigateFrames(1)
    })

    document.getElementById("frame-next-10").addEventListener("click", () => {
      this.navigateFrames(10)
    })

    document.getElementById("frame-number").addEventListener("change", () => {
      this.jumpToFrame()
    })

    document.getElementById("frame-slider").addEventListener("input", (e) => {
      this.updateFramePreview(Number.parseInt(e.target.value))
    })

    document.getElementById("cancel-frame-selection").addEventListener("click", () => {
      this.cancelFrameSelection()
    })

    document.getElementById("select-reference-frame").addEventListener("click", () => {
      this.selectReferenceFrame()
    })

    // Preprocessing events
    document.getElementById("enable-crop").addEventListener("change", () => {
      this.toggleCrop()
    })

    document.getElementById("select-area-btn").addEventListener("click", () => {
      this.startCropSelection()
    })

    // Labeling events
    document.getElementById("brush-size").addEventListener("input", (e) => {
      this.brushSize = Number.parseInt(e.target.value)
      document.getElementById("brush-size-value").textContent = this.brushSize
      this.updateStatusBar()
    })

    // Label preview update
    document.getElementById("current-label").addEventListener("change", (e) => {
      this.currentLabel = Number.parseInt(e.target.value)
      this.updateStatusBar()
      this.updateLabelPreview()
    })

    document.getElementById("current-label").addEventListener("change", (e) => {
      this.currentLabel = Number.parseInt(e.target.value)
      this.updateStatusBar()
    })

    document.getElementById("brush-tool").addEventListener("click", () => {
      this.setTool("brush")
    })

    document.getElementById("erase-tool").addEventListener("click", () => {
      this.setTool("erase")
    })

    document.getElementById("crop-tool").addEventListener("click", () => {
      this.setTool("crop")
    })

    document.getElementById("reset-view").addEventListener("click", () => {
      this.resetZoom()
    })

    document.getElementById("clear-labels").addEventListener("click", () => {
      this.clearLabels()
    })

    document.getElementById("train-classifier").addEventListener("click", () => {
      this.trainClassifier()
    })

    // Feature selection events
    document.querySelectorAll('input[name="features"]').forEach((checkbox) => {
      checkbox.addEventListener("change", (e) => {
        if (e.target.checked) {
          this.selectedFeatures.add(e.target.value)
        } else {
          this.selectedFeatures.delete(e.target.value)
        }
      })
    })

    document.getElementById("apply-features").addEventListener("click", () => {
      this.applyFeatures()
    })

    // Processing events
    document.getElementById("run-segmentation").addEventListener("click", () => {
      this.runSegmentation()
    })

    document.getElementById("download-results").addEventListener("click", () => {
      this.downloadResults()
    })

    // Video batch processing events
    document.getElementById("process-all-frames").addEventListener("click", () => {
      this.processAllVideoFrames()
    })

    document.getElementById("stop-processing").addEventListener("click", () => {
      this.stopVideoProcessing()
    })

    // Canvas events
    this.labelCanvas.addEventListener("mousedown", (e) => {
      this.handleCanvasMouseDown(e)
    })

    this.labelCanvas.addEventListener("mousemove", (e) => {
      this.handleCanvasMouseMove(e)
    })

    this.labelCanvas.addEventListener("mouseup", () => {
      this.handleCanvasMouseUp()
    })

    this.labelCanvas.addEventListener("mouseleave", () => {
      this.isDrawing = false
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

    // Batch processing events
    document.getElementById("select-output-folder").addEventListener("click", () => {
      this.selectOutputFolder()
    })

    document.getElementById("start-batch-processing").addEventListener("click", () => {
      this.startBatchProcessing()
    })

    document.getElementById("stop-batch-processing").addEventListener("click", () => {
      this.stopBatchProcessing()
    })
  }

  initializeTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn")
    tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const tabId = button.getAttribute("data-tab")

        // Update active tab button
        tabButtons.forEach((btn) => btn.classList.remove("active"))
        button.classList.add("active")

        // Update active tab panel
        document.querySelectorAll(".tab-panel").forEach((panel) => {
          panel.classList.remove("active")
        })
        document.getElementById(`${tabId}-tab`).classList.add("active")
      })
    })
  }

  handleInputTypeChange() {
    const inputType = document.querySelector('input[name="inputType"]:checked').value

    // Update file input accept attribute
    if (inputType === "image") {
      this.fileInput.setAttribute("accept", "image/*")
      this.fileInput.removeAttribute("multiple")
      this.controls.videoSettings.style.display = "none"
    } else if (inputType === "folder") {
      this.fileInput.setAttribute("accept", "image/*")
      this.fileInput.setAttribute("multiple", "multiple")
      this.controls.videoSettings.style.display = "none"
    } else if (inputType === "video") {
      this.fileInput.setAttribute("accept", "video/*")
      this.fileInput.removeAttribute("multiple")
      this.controls.videoSettings.style.display = "block"
    }
  }

  handleVideoModeChange() {
    const videoMode = document.querySelector('input[name="videoMode"]:checked').value
    const timeIntervalControl = document.getElementById("time-interval-control")

    if (videoMode === "time") {
      timeIntervalControl.style.display = "block"
    } else {
      timeIntervalControl.style.display = "none"
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

  async setupVideoElement(file) {
    // Create video element for processing
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

  async loadVideoFrames() {
    if (!this.videoElement || !this.videoLoaded) {
      alert("Please select a video file first")
      return
    }

    try {
      // Get settings
      const videoMode = document.querySelector('input[name="videoMode"]:checked').value
      let frameInterval, maxFrames

      if (videoMode === "time") {
        const timeInterval = Number.parseFloat(document.getElementById("time-interval").value)
        frameInterval = Math.max(1, Math.round(timeInterval * this.actualFPS))
        const maxDuration = Math.min(this.videoElement.duration, 300) // Limit to 5 minutes
        maxFrames = Math.floor((maxDuration * this.actualFPS) / frameInterval)
      } else {
        frameInterval = Number.parseInt(document.getElementById("frame-interval").value)
        maxFrames = Number.parseInt(document.getElementById("max-frames").value)
      }

      frameInterval = Math.max(1, frameInterval)
      maxFrames = Math.min(1000, Math.max(1, maxFrames))

      this.showProgress(`Extracting ${maxFrames} frames from video...`)

      this.videoFrames = []
      const canvas = document.createElement("canvas")
      const ctx = canvas.getContext("2d")

      const duration = this.videoElement.duration
      const totalFrames = Math.floor(duration * this.actualFPS)
      const actualMaxFrames = Math.min(maxFrames, Math.floor(totalFrames / frameInterval))

      canvas.width = this.videoElement.videoWidth
      canvas.height = this.videoElement.videoHeight

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

      // Show frame selector
      this.showFrameSelector()

      // Enable video batch controls
      this.controls.videoBatchControls.style.display = "block"
    } catch (error) {
      this.hideProgress()
      console.error("Video frame extraction failed:", error)
      alert(`Failed to extract video frames: ${error.message}`)
    }
  }

  showFrameSelector() {
    if (this.videoFrames.length === 0) return

    // Update frame selector UI
    this.controls.frameSelector.style.display = "block"

    // Update slider
    const frameSlider = document.getElementById("frame-slider")
    frameSlider.min = 0
    frameSlider.max = this.videoFrames.length - 1
    frameSlider.value = 0

    // Update frame count
    document.getElementById("frame-total").textContent = `/ ${this.videoFrames.length}`
    document.getElementById("frame-number").value = 1
    document.getElementById("frame-number").max = this.videoFrames.length

    // Show first frame
    this.currentFrameIndex = 0
    this.updateFramePreview(0)
  }

  updateFramePreview(frameIndex) {
    if (frameIndex < 0 || frameIndex >= this.videoFrames.length) return

    this.currentFrameIndex = frameIndex
    const frameData = this.videoFrames[frameIndex]
    const frameImage = frameData[1]

    // Update preview canvas
    this.previewCanvas.width = frameImage.width
    this.previewCanvas.height = frameImage.height
    this.previewCtx.clearRect(0, 0, this.previewCanvas.width, this.previewCanvas.height)
    this.previewCtx.drawImage(frameImage, 0, 0)

    // Update frame info
    document.getElementById("current-frame-info").textContent = `Frame ${frameIndex + 1} of ${this.videoFrames.length}`
    document.getElementById("frame-number").value = frameIndex + 1
    document.getElementById("frame-slider").value = frameIndex
  }

  navigateFrames(delta) {
    const newIndex = this.currentFrameIndex + delta
    if (newIndex >= 0 && newIndex < this.videoFrames.length) {
      this.updateFramePreview(newIndex)
    }
  }

  jumpToFrame() {
    const frameNumber = Number.parseInt(document.getElementById("frame-number").value)
    if (frameNumber >= 1 && frameNumber <= this.videoFrames.length) {
      this.updateFramePreview(frameNumber - 1)
    }
  }

  cancelFrameSelection() {
    this.controls.frameSelector.style.display = "none"
  }

  selectReferenceFrame() {
    if (this.currentFrameIndex < 0 || this.currentFrameIndex >= this.videoFrames.length) return

    const frameData = this.videoFrames[this.currentFrameIndex]
    const frameImage = frameData[1]

    // Set as current image
    this.originalImage = frameImage
    this.currentImage = frameImage

    // Hide frame selector
    this.controls.frameSelector.style.display = "none"

    // Display image
    this.drawImageOnCanvas()
    this.initializeLabelMask()
    this.uploadPlaceholder.style.display = "none"

    // Update UI
    this.setCurrentStep(2) // Enable up to labeling
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
    this.labelCanvas.width = width
    this.labelCanvas.height = height

    this.mainCtx.clearRect(0, 0, width, height)
    this.labelCtx.clearRect(0, 0, width, height)
    this.mainCtx.drawImage(this.currentImage, 0, 0, width, height)

    // Enable pointer events on label canvas
    this.labelCanvas.style.pointerEvents = "auto"
    this.labelCanvas.style.cursor = "crosshair"
  }

  initializeLabelMask() {
    if (!this.currentImage) return

    const width = this.currentImage.width
    const height = this.currentImage.height

    // Create label mask as ImageData
    this.labelMask = new ImageData(width, height)
  }

  toggleCrop() {
    const enableCrop = this.controls.enableCrop.checked
    this.controls.cropControls.style.display = enableCrop ? "block" : "none"
  }

  startCropSelection() {
    this.setTool("crop")
    this.cropMode = true
    this.cropOverlay.style.display = "none"
    this.statusElements.tool.textContent = "Tool: crop (click and drag to select area)"
  }

  handleCanvasMouseDown(event) {
    const rect = this.labelCanvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    if (this.currentTool === "crop" && this.cropMode) {
      this.cropStartX = x
      this.cropStartY = y
      this.cropOverlay.style.display = "block"
      this.cropOverlay.style.left = `${x}px`
      this.cropOverlay.style.top = `${y}px`
      this.cropOverlay.style.width = "0px"
      this.cropOverlay.style.height = "0px"
    } else {
      this.isDrawing = true
      this.paintLabel(x, y)
    }
  }

  handleCanvasMouseMove(event) {
    const rect = this.labelCanvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    if (this.currentTool === "crop" && this.cropMode && this.cropStartX !== undefined) {
      const width = x - this.cropStartX
      const height = y - this.cropStartY

      this.cropOverlay.style.width = `${Math.abs(width)}px`
      this.cropOverlay.style.height = `${Math.abs(height)}px`

      if (width < 0) {
        this.cropOverlay.style.left = `${x}px`
      }

      if (height < 0) {
        this.cropOverlay.style.top = `${y}px`
      }
    } else if (this.isDrawing) {
      this.paintLabel(x, y)
    }
  }

  handleCanvasMouseUp() {
    if (this.currentTool === "crop" && this.cropMode && this.cropStartX !== undefined) {
      const cropOverlayRect = this.cropOverlay.getBoundingClientRect()
      const canvasRect = this.labelCanvas.getBoundingClientRect()

      const x1 = Math.max(0, (cropOverlayRect.left - canvasRect.left) / this.zoomLevel)
      const y1 = Math.max(0, (cropOverlayRect.top - canvasRect.top) / this.zoomLevel)
      const x2 = Math.min(this.currentImage.width, (cropOverlayRect.right - canvasRect.left) / this.zoomLevel)
      const y2 = Math.min(this.currentImage.height, (cropOverlayRect.bottom - canvasRect.top) / this.zoomLevel)

      if (x2 - x1 > 10 && y2 - y1 > 10) {
        this.applyCrop(x1, y1, x2, y2)
      }

      this.cropMode = false
      this.cropStartX = undefined
      this.cropStartY = undefined
      this.cropOverlay.style.display = "none"
      this.setTool("brush")
    }

    this.isDrawing = false
  }

  applyCrop(x1, y1, x2, y2) {
    if (!this.currentImage) return

    const width = x2 - x1
    const height = y2 - y1

    const canvas = document.createElement("canvas")
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext("2d")

    ctx.drawImage(this.currentImage, x1, y1, width, height, 0, 0, width, height)

    const croppedImage = new Image()
    croppedImage.onload = () => {
      this.originalImage = croppedImage
      this.currentImage = croppedImage
      this.drawImageOnCanvas()
      this.initializeLabelMask()
    }
    croppedImage.src = canvas.toDataURL()

    this.cropCoords = { x1, y1, x2, y2 }
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

  setTool(tool) {
    this.currentTool = tool

    // Update tool button states
    document.querySelectorAll(".btn-tool").forEach((btn) => {
      btn.classList.remove("active")
    })

    if (tool !== "crop") {
      document.getElementById(`${tool}-tool`).classList.add("active")
    }

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

  updateStatusBar() {
    this.statusElements.zoom.textContent = `Zoom: ${Math.round(this.zoomLevel * 100)}%`
    this.statusElements.tool.textContent = `Tool: ${this.currentTool}`
    this.statusElements.brush.textContent = `Brush Size: ${this.brushSize}`

    const labelName = this.labelNames[this.currentLabel] || "Unknown"
    this.statusElements.label.textContent = `Current Label: ${labelName}`
  }

  resetInput() {
    this.uploadedFiles = []
    this.currentImage = null
    this.originalImage = null
    this.labelMask = null
    this.classifier = null
    this.videoFrames = []
    this.statusElements.file.textContent = "No files selected"
    this.fileInput.value = ""

    if (this.mainCanvas) {
      this.mainCtx.clearRect(0, 0, this.mainCanvas.width, this.mainCanvas.height)
    }

    if (this.labelCanvas) {
      this.labelCtx.clearRect(0, 0, this.labelCanvas.width, this.labelCanvas.height)
    }

    this.uploadPlaceholder.style.display = "flex"
    this.controls.frameSelector.style.display = "none"
    this.controls.videoBatchControls.style.display = "none"

    // Reset video element if exists
    if (this.videoElement) {
      URL.revokeObjectURL(this.videoElement.src)
      this.videoElement.remove()
      this.videoElement = null
      this.videoLoaded = false
    }

    this.setCurrentStep(0)
  }

  setCurrentStep(step) {
    this.currentStep = step
    this.updateUI()
    this.updateBatchProcessingState()
  }

  updateUI() {
    const cards = ["input-card", "preprocess-card", "labeling-card", "features-card", "processing-card"]

    cards.forEach((cardId, index) => {
      const card = document.getElementById(cardId)

      // Remove all state classes
      card.classList.remove("disabled", "enabled", "current", "completed")

      if (index < this.currentStep) {
        card.classList.add("completed")
      } else if (index === this.currentStep) {
        card.classList.add("current", "enabled")
      } else if (index <= this.currentStep) {
        card.classList.add("enabled")
      } else {
        card.classList.add("disabled")
      }
    })

    // Always enable labeling if we have an image
    if (this.currentImage) {
      document.getElementById("labeling-card").classList.remove("disabled")
      document.getElementById("labeling-card").classList.add("enabled")
      document.getElementById("features-card").classList.remove("disabled")
      document.getElementById("features-card").classList.add("enabled")
    }

    // Enable processing if classifier is trained
    if (this.classifier && this.classifier.trained) {
      document.getElementById("processing-card").classList.remove("disabled")
      document.getElementById("processing-card").classList.add("enabled")
    }

    // Enable video batch processing button if classifier is trained and video frames are loaded
    if (this.classifier && this.classifier.trained && this.videoFrames.length > 0) {
      document.getElementById("process-all-frames").disabled = false
    }
  }

  showProgress(message) {
    this.statusElements.progress.style.display = "block"
    this.statusElements.progressFill.style.width = "0%"
    this.statusElements.progressText.textContent = message
  }

  hideProgress() {
    setTimeout(() => {
      this.statusElements.progress.style.display = "none"
    }, 1000)
  }

  updateProgressBar(percentage) {
    this.statusElements.progressFill.style.width = `${percentage}%`
  }

  updateProgressText(text) {
    this.statusElements.progressText.textContent = text
  }

  async trainClassifier() {
    if (!this.currentImage || !this.labelMask) {
      alert("Please load an image and add some labels first")
      return
    }

    // Check if we have labeled pixels
    let hasLabels = false
    const labelCount = {}

    for (let i = 3; i < this.labelMask.data.length; i += 4) {
      const label = this.labelMask.data[i]
      if (label > 0) {
        hasLabels = true
        labelCount[label] = (labelCount[label] || 0) + 1
      }
    }

    if (!hasLabels) {
      alert("Please label some pixels first using the brush tool")
      return
    }

    // Check if we have at least 2 different labels
    const uniqueLabels = Object.keys(labelCount)
    if (uniqueLabels.length < 2) {
      alert("Please label pixels with at least 2 different classes (e.g., Cell and Background)")
      return
    }

    // Check if we have selected features
    if (this.selectedFeatures.size === 0) {
      alert("Please select at least one feature from the Feature Selection tab")
      return
    }

    try {
      this.showProgress("Training classifier...")

      console.log("Label distribution:", labelCount)
      console.log("Selected features:", Array.from(this.selectedFeatures))

      // Extract features from the current image
      const features = await this.extractFeatures(this.currentImage)

      console.log("Extracted features:", Object.keys(features))

      // Prepare training data
      const trainingData = []
      const trainingLabels = []

      for (let y = 0; y < this.currentImage.height; y++) {
        for (let x = 0; x < this.currentImage.width; x++) {
          const index = (y * this.currentImage.width + x) * 4
          const label = this.labelMask.data[index + 3]

          if (label > 0) {
            const pixelFeatures = this.getPixelFeatures(features, x, y)
            if (pixelFeatures.length > 0) {
              trainingData.push(pixelFeatures)
              trainingLabels.push(label)
            }
          }
        }
      }

      if (trainingData.length === 0) {
        throw new Error("No valid training data extracted. Please check your labels and selected features.")
      }

      console.log(`Training data: ${trainingData.length} samples with ${trainingData[0].length} features each`)

      // Train a simple classifier
      this.classifier = {
        trainingData: trainingData,
        trainingLabels: trainingLabels,
        features: Object.keys(features),
        trained: true,
      }

      this.hideProgress()

      const labelSummary = Object.entries(labelCount)
        .map(([label, count]) => `${this.labelNames[label] || `Label ${label}`}: ${count} pixels`)
        .join("\n")

      alert(
        `Classifier trained successfully!\n\nTraining summary:\n${labelSummary}\nTotal samples: ${trainingData.length}\nFeatures used: ${trainingData[0].length}`,
      )

      this.setCurrentStep(4)

      // Enable video batch processing if video frames are loaded
      if (this.videoFrames.length > 0) {
        document.getElementById("process-all-frames").disabled = false
      }
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

    // Always include raw intensity as a baseline
    features.raw = grayData

    // Apply selected features
    if (this.selectedFeatures.has("gaussian")) {
      features.gaussian = this.applyGaussianFilter(grayData, image.width, image.height, 1.0)
      console.log("Applied Gaussian filter")
    }

    if (this.selectedFeatures.has("laplacian")) {
      features.laplacian = this.applyLaplacianFilter(grayData, image.width, image.height)
      console.log("Applied Laplacian filter")
    }

    if (this.selectedFeatures.has("gradient")) {
      features.gradient = this.applyGradientFilter(grayData, image.width, image.height)
      console.log("Applied Gradient filter")
    }

    if (this.selectedFeatures.has("sobel")) {
      features.sobel = this.applySobelFilter(grayData, image.width, image.height)
      console.log("Applied Sobel filter")
    }

    // Add more robust features if selected
    if (this.selectedFeatures.has("gabor")) {
      features.gabor = this.applyGaborFilter(grayData, image.width, image.height)
      console.log("Applied Gabor filter")
    }

    if (this.selectedFeatures.has("hessian")) {
      features.hessian = this.applyHessianFilter(grayData, image.width, image.height)
      console.log("Applied Hessian filter")
    }

    console.log("Feature extraction completed. Available features:", Object.keys(features))
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

  async runSegmentation() {
    if (!this.classifier || !this.classifier.trained) {
      alert("Please train the classifier first")
      return
    }

    try {
      this.showProgress("Running segmentation...")

      // Debug: Check classifier training data
      console.log("Classifier training data summary:")
      console.log("- Training samples:", this.classifier.trainingData.length)
      console.log("- Feature dimensions:", this.classifier.trainingData[0]?.length || 0)

      const labelCounts = {}
      this.classifier.trainingLabels.forEach((label) => {
        labelCounts[label] = (labelCounts[label] || 0) + 1
      })
      console.log("- Training label distribution:", labelCounts)

      // Extract features from current image
      const features = await this.extractFeatures(this.currentImage)

      // Segment the image
      const segmentedData = await this.segmentImage(features)

      // Display result
      this.displaySegmentationResult(segmentedData)

      this.hideProgress()
      alert("Segmentation completed! Check the console for debug information.")
    } catch (error) {
      this.hideProgress()
      console.error("Segmentation failed:", error)
      alert(`Segmentation failed: ${error.message}`)
    }
  }

  async applyFeatures() {
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

  applyGaborFilter(data, width, height) {
    const result = new Float32Array(data.length)
    const sigma = 2.0
    const theta = 0 // Orientation
    const lambda = 4.0 // Wavelength
    const gamma = 0.5 // Aspect ratio

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0
        let weightSum = 0

        // Simple Gabor-like filter (simplified implementation)
        for (let dy = -3; dy <= 3; dy++) {
          for (let dx = -3; dx <= 3; dx++) {
            const ny = y + dy
            const nx = x + dx

            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
              const distance = Math.sqrt(dx * dx + dy * dy)
              if (distance <= 3) {
                const weight =
                  Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)) * Math.cos((2 * Math.PI * dx) / lambda)
                sum += data[ny * width + nx] * weight
                weightSum += Math.abs(weight)
              }
            }
          }
        }

        result[y * width + x] = weightSum > 0 ? Math.abs(sum / weightSum) : 0
      }
    }

    return result
  }

  applyHessianFilter(data, width, height) {
    const result = new Float32Array(data.length)

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        // Compute second derivatives (Hessian matrix elements)
        const fxx = data[y * width + (x + 1)] - 2 * data[y * width + x] + data[y * width + (x - 1)]
        const fyy = data[(y + 1) * width + x] - 2 * data[y * width + x] + data[(y - 1) * width + x]
        const fxy =
          (data[(y + 1) * width + (x + 1)] -
            data[(y + 1) * width + (x - 1)] -
            data[(y - 1) * width + (x + 1)] +
            data[(y - 1) * width + (x - 1)]) /
          4

        // Compute determinant of Hessian matrix
        const det = fxx * fyy - fxy * fxy
        result[y * width + x] = Math.abs(det)
      }
    }

    return result
  }

  getPixelFeatures(features, x, y) {
    const pixelFeatures = []
    const width = this.currentImage.width
    const index = y * width + x

    // Ensure we're within bounds
    if (index < 0 || index >= width * this.currentImage.height) {
      return []
    }

    for (const [featureName, featureData] of Object.entries(features)) {
      if (featureData && index < featureData.length) {
        const value = featureData[index]
        if (!isNaN(value) && isFinite(value)) {
          pixelFeatures.push(value)
        }
      }
    }

    return pixelFeatures
  }

  classifyPixel(pixelFeatures) {
    if (!this.classifier || !this.classifier.trained || pixelFeatures.length === 0) {
      return 2 // Default to background
    }

    // Improved nearest neighbor classification with distance weighting
    let bestLabel = 2
    const k = Math.min(5, this.classifier.trainingData.length) // Use k=5 for k-NN

    // Find k nearest neighbors
    const distances = []

    for (let i = 0; i < this.classifier.trainingData.length; i++) {
      const trainFeatures = this.classifier.trainingData[i]
      let distance = 0

      const minLength = Math.min(pixelFeatures.length, trainFeatures.length)
      for (let j = 0; j < minLength; j++) {
        const diff = pixelFeatures[j] - trainFeatures[j]
        distance += diff * diff
      }

      distance = Math.sqrt(distance)
      distances.push({ distance, label: this.classifier.trainingLabels[i] })
    }

    // Sort by distance and take k nearest
    distances.sort((a, b) => a.distance - b.distance)
    const kNearest = distances.slice(0, k)

    // Vote by labels (weighted by inverse distance)
    const votes = {}

    for (const neighbor of kNearest) {
      const weight = neighbor.distance > 0 ? 1 / (neighbor.distance + 1e-6) : 1000
      votes[neighbor.label] = (votes[neighbor.label] || 0) + weight
    }

    // Find label with highest weighted vote
    let maxVote = -1
    for (const [label, vote] of Object.entries(votes)) {
      if (vote > maxVote) {
        maxVote = vote
        bestLabel = Number.parseInt(label)
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

    // Debug: Check what labels we're getting
    const labelCounts = {}
    for (let i = 0; i < segmentedData.length; i++) {
      const label = segmentedData[i]
      labelCounts[label] = (labelCounts[label] || 0) + 1
    }
    console.log("Segmentation label distribution:", labelCounts)

    // Convert segmentation to RGB with better color mapping
    for (let i = 0; i < segmentedData.length; i++) {
      const label = segmentedData[i]
      let color = [128, 128, 128] // Default gray

      // Use the predefined label colors
      if (this.labelColors[label]) {
        color = this.labelColors[label]
      } else {
        // Fallback colors for unexpected labels
        switch (label) {
          case 0:
            color = [0, 0, 0]
            break // Black for unlabeled
          case 1:
            color = [255, 0, 0]
            break // Red for Cell
          case 2:
            color = [0, 255, 0]
            break // Green for Background
          case 3:
            color = [0, 0, 255]
            break // Blue for Nucleus
          case 4:
            color = [255, 255, 0]
            break // Yellow for Membrane
          default:
            color = [255, 255, 255]
            break // White for unknown
        }
      }

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

  downloadResults() {
    if (!this.mainCanvas) return

    const link = document.createElement("a")
    link.download = "segmentation_result.png"
    link.href = this.mainCanvas.toDataURL("image/png")
    link.click()
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

      // Update UI
      document.getElementById("process-all-frames").disabled = true
      document.getElementById("stop-processing").disabled = false
      this.statusElements.frameProgressText.textContent = `Processing ${this.totalFramesToProcess} video frames...`
      this.statusElements.frameProgress.style.width = "0%"

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
        frameCtx.drawImage(frameImage, 0, 0)

        try {
          // Extract features from the frame
          const features = await this.extractFeatures(frameImage)

          // Segment the frame
          const segmentedData = await this.segmentImage(features)

          // Convert to binary if needed
          const binaryOutput = document.getElementById("binary-output").checked
          let resultData

          if (binaryOutput) {
            resultData = this.convertToBinaryMask(segmentedData)
          } else {
            resultData = this.convertToColorMask(segmentedData)
          }

          // Create result canvas
          const resultCanvas = document.createElement("canvas")
          resultCanvas.width = frameCanvas.width
          resultCanvas.height = frameCanvas.height
          const resultCtx = resultCanvas.getContext("2d")

          // Draw result to canvas
          const imageData = resultCtx.createImageData(resultCanvas.width, resultCanvas.height)

          if (binaryOutput) {
            for (let j = 0; j < resultData.length; j++) {
              const pixelIndex = j * 4
              const value = resultData[j]
              imageData.data[pixelIndex] = value // R
              imageData.data[pixelIndex + 1] = value // G
              imageData.data[pixelIndex + 2] = value // B
              imageData.data[pixelIndex + 3] = 255 // A
            }
          } else {
            imageData.data.set(resultData)
          }

          resultCtx.putImageData(imageData, 0, 0)

          // Store the segmented frame
          segmentedFrames.push({
            name: frameName.replace(".png", "_segmented.png"),
            data: resultCanvas.toDataURL("image/png").split(",")[1], // Base64 without prefix
            originalName: frameName,
            timestamp: timestamp,
            frameIndex: i,
          })

          // Update progress
          this.processedFrameCount++
          const progress = (this.processedFrameCount / this.totalFramesToProcess) * 100
          this.statusElements.frameProgress.style.width = `${progress}%`
          this.statusElements.frameProgressText.textContent = `Processed ${this.processedFrameCount}/${this.totalFramesToProcess} frames`

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
        this.statusElements.frameProgressText.textContent = "Processing completed!"
        alert(`Video processing completed! ${segmentedFrames.length} frames processed.`)
      } else {
        this.statusElements.frameProgressText.textContent = "Processing cancelled."
        alert("Video processing cancelled.")
      }
    } catch (error) {
      console.error("Video processing failed:", error)
      this.statusElements.frameProgressText.textContent = `Error: ${error.message}`
      alert(`Video processing failed: ${error.message}`)
    } finally {
      this.videoProcessingActive = false
      document.getElementById("process-all-frames").disabled = false
      document.getElementById("stop-processing").disabled = true
    }
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

  convertToColorMask(segmentedData) {
    const width = this.currentImage.width
    const height = this.currentImage.height
    const colorMask = new Uint8ClampedArray(width * height * 4)

    for (let i = 0; i < segmentedData.length; i++) {
      const label = segmentedData[i]
      const color = this.labelColors[label] || [128, 128, 128]
      const pixelIndex = i * 4

      colorMask[pixelIndex] = color[0] // R
      colorMask[pixelIndex + 1] = color[1] // G
      colorMask[pixelIndex + 2] = color[2] // B
      colorMask[pixelIndex + 3] = 255 // A
    }

    return colorMask
  }

  async createOutputFolder(segmentedFrames) {
    // Check if JSZip is available
    if (window.JSZip) {
      const zip = new window.JSZip()
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

  stopVideoProcessing() {
    this.videoProcessingActive = false
    document.getElementById("stop-processing").disabled = true
    this.statusElements.frameProgressText.textContent = "Stopping processing..."
  }

  updateLabelPreview() {
    if (this.labelColors[this.currentLabel]) {
      const color = this.labelColors[this.currentLabel]
      this.labelPreviewCtx.clearRect(0, 0, 20, 20)
      this.labelPreviewCtx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`
      this.labelPreviewCtx.fillRect(0, 0, 20, 20)

      const labelName = this.labelNames[this.currentLabel] || "Unknown"
      this.currentLabelText.textContent = `${labelName} (${this.currentLabel})`
    }
  }

  selectOutputFolder() {
    // Since we can't access file system directly in browser, we'll use a different approach
    // For now, we'll just enable batch processing and use downloads
    this.outputFolder = "downloads"
    this.outputFolderStatus.textContent = "Will download to browser's download folder"
    this.updateBatchProcessingState()
  }

  updateBatchProcessingState() {
    const canProcess =
      this.outputFolder &&
      this.classifier &&
      this.classifier.trained &&
      (this.uploadedFiles.length > 0 || this.videoFrames.length > 0)

    document.getElementById("start-batch-processing").disabled = !canProcess
  }

  async startBatchProcessing() {
    if (!this.classifier || !this.classifier.trained) {
      alert("Please train the classifier first")
      return
    }

    let files = []
    const inputType = document.querySelector('input[name="inputType"]:checked').value

    if (inputType === "image" && this.uploadedFiles.length > 0) {
      files = this.uploadedFiles
    } else if (inputType === "folder" && this.uploadedFiles.length > 0) {
      files = this.uploadedFiles.filter((f) => f.name.toLowerCase().match(/\.(png|jpg|jpeg|bmp|tiff)$/))
    } else if (inputType === "video" && this.videoFrames.length > 0) {
      // For video, we'll process the extracted frames
      files = this.videoFrames.map((frame) => ({
        name: frame[2],
        data: frame[1],
      }))
    }

    if (files.length === 0) {
      alert("No files to process")
      return
    }

    this.batchQueue = [...files]
    this.batchRunning = true
    this.batchProcessed = 0
    this.batchTotal = files.length

    document.getElementById("start-batch-processing").disabled = true
    document.getElementById("stop-batch-processing").disabled = false

    this.batchStatus.textContent = `Processing 0/${this.batchTotal} files...`

    await this.processBatch()
  }

  async processBatch() {
    const outputFormat = document.getElementById("output-format").value.toLowerCase()
    const segmentedFiles = []

    for (let i = 0; i < this.batchQueue.length && this.batchRunning; i++) {
      const file = this.batchQueue[i]

      try {
        this.batchStatus.textContent = `Processing ${file.name || `file ${i + 1}`}...`

        let image
        if (file.data instanceof HTMLImageElement) {
          // Video frame
          image = file.data
        } else {
          // Regular file
          image = await this.loadImageFromFile(file)
        }

        // Apply crop if enabled
        if (this.cropCoords) {
          image = this.applyCropToImage(image, this.cropCoords)
        }

        // Extract features and segment
        const features = await this.extractFeatures(image)
        const segmentedData = await this.segmentImage(features)

        // Convert to desired output format
        let resultData
        if (document.getElementById("binary-output").checked) {
          resultData = this.convertToBinaryMask(segmentedData)
        } else {
          resultData = this.convertToColorMask(segmentedData)
        }

        // Create result canvas
        const resultCanvas = document.createElement("canvas")
        resultCanvas.width = image.width
        resultCanvas.height = image.height
        const resultCtx = resultCanvas.getContext("2d")

        const imageData = resultCtx.createImageData(resultCanvas.width, resultCanvas.height)

        if (document.getElementById("binary-output").checked) {
          for (let j = 0; j < resultData.length; j++) {
            const pixelIndex = j * 4
            const value = resultData[j]
            imageData.data[pixelIndex] = value
            imageData.data[pixelIndex + 1] = value
            imageData.data[pixelIndex + 2] = value
            imageData.data[pixelIndex + 3] = 255
          }
        } else {
          imageData.data.set(resultData)
        }

        resultCtx.putImageData(imageData, 0, 0)

        // Store result
        const fileName = (file.name || `frame_${i}`).replace(/\.[^/.]+$/, "") + `_segmented.${outputFormat}`
        segmentedFiles.push({
          name: fileName,
          data: resultCanvas.toDataURL(`image/${outputFormat}`).split(",")[1],
        })

        this.batchProcessed++
        this.batchStatus.textContent = `Processing ${this.batchProcessed}/${this.batchTotal} files...`

        // Small delay to prevent UI blocking
        await new Promise((resolve) => setTimeout(resolve, 50))
      } catch (error) {
        console.error(`Error processing file ${file.name}:`, error)
      }
    }

    if (this.batchRunning) {
      // Create and download ZIP file
      await this.downloadBatchResults(segmentedFiles)
      this.batchStatus.textContent = `Batch processing completed! ${this.batchProcessed}/${this.batchTotal} files processed.`
    } else {
      this.batchStatus.textContent = "Batch processing stopped."
    }

    this.batchRunning = false
    document.getElementById("start-batch-processing").disabled = false
    document.getElementById("stop-batch-processing").disabled = true
  }

  async loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const img = new Image()
        img.onload = () => resolve(img)
        img.onerror = reject
        img.src = e.target.result
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  applyCropToImage(image, cropCoords) {
    const canvas = document.createElement("canvas")
    const ctx = canvas.getContext("2d")

    const { x1, y1, x2, y2 } = cropCoords
    const width = x2 - x1
    const height = y2 - y1

    canvas.width = width
    canvas.height = height
    ctx.drawImage(image, x1, y1, width, height, 0, 0, width, height)

    const croppedImage = new Image()
    croppedImage.src = canvas.toDataURL()
    return croppedImage
  }

  async downloadBatchResults(segmentedFiles) {
    if (window.JSZip && segmentedFiles.length > 1) {
      const zip = new window.JSZip()
      const folder = zip.folder("batch_segmentation_results")

      for (const file of segmentedFiles) {
        folder.file(file.name, file.data, { base64: true })
      }

      // Add metadata
      const metadata = {
        processing_date: new Date().toISOString(),
        total_files: segmentedFiles.length,
        selected_features: Array.from(this.selectedFeatures),
        label_types: this.labelNames,
        output_format: document.getElementById("output-format").value,
      }

      folder.file("processing_metadata.json", JSON.stringify(metadata, null, 2))

      const content = await zip.generateAsync({ type: "blob" })
      this.downloadBlob(content, "batch_segmentation_results.zip")
    } else {
      // Download files individually
      for (const file of segmentedFiles) {
        const blob = this.base64ToBlob(
          file.data,
          `image/${document.getElementById("output-format").value.toLowerCase()}`,
        )
        this.downloadBlob(blob, file.name)
        await new Promise((resolve) => setTimeout(resolve, 100))
      }
    }
  }

  stopBatchProcessing() {
    this.batchRunning = false
    this.batchStatus.textContent = "Stopping batch processing..."
    document.getElementById("stop-batch-processing").disabled = true
  }
}

// Add JSZip library for ZIP file creation
document.addEventListener("DOMContentLoaded", () => {
  const script = document.createElement("script")
  script.src = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.7.1/jszip.min.js"
  script.integrity = "sha512-xQBQYt9UcgblF6aCMrwU1NkVA7HCXaSN2oq0so80KO+y68M+n64FOcqgav4igHe6D5ObBLIf68DWv+gfBowczg=="
  script.crossOrigin = "anonymous"
  script.referrerPolicy = "no-referrer"
  document.head.appendChild(script)
})
