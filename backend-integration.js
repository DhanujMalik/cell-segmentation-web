// Backend integration for advanced image processing
class BackendIntegration {
  constructor(baseUrl = "http://localhost:5000") {
    this.baseUrl = baseUrl
  }

  async extractFeatures(imageCanvas, selectedFeatures, sigmaValues) {
    try {
      const imageData = imageCanvas.toDataURL("image/png")

      const response = await fetch(`${this.baseUrl}/api/extract_features`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: imageData,
          features: Array.from(selectedFeatures),
          sigma_values: sigmaValues,
        }),
      })

      const result = await response.json()
      if (!result.success) {
        throw new Error(result.error)
      }

      return result.features
    } catch (error) {
      console.error("Feature extraction failed:", error)
      throw error
    }
  }

  async trainClassifier(imageCanvas, labelMask, selectedFeatures) {
    try {
      const imageData = imageCanvas.toDataURL("image/png")

      const response = await fetch(`${this.baseUrl}/api/train_classifier`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: imageData,
          labels: Array.from(labelMask),
          features: Array.from(selectedFeatures),
        }),
      })

      const result = await response.json()
      if (!result.success) {
        throw new Error(result.error)
      }

      return result
    } catch (error) {
      console.error("Classifier training failed:", error)
      throw error
    }
  }

  async segmentImage(imageCanvas, selectedFeatures, binaryOutput = true) {
    try {
      const imageData = imageCanvas.toDataURL("image/png")

      const response = await fetch(`${this.baseUrl}/api/segment_image`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: imageData,
          features: Array.from(selectedFeatures),
          binary_output: binaryOutput,
        }),
      })

      const result = await response.json()
      if (!result.success) {
        throw new Error(result.error)
      }

      return result.segmented_image
    } catch (error) {
      console.error("Image segmentation failed:", error)
      throw error
    }
  }

  async processVideoFrame(frameCanvas, selectedFeatures) {
    try {
      const frameData = frameCanvas.toDataURL("image/png")

      const response = await fetch(`${this.baseUrl}/api/process_video_frame`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          frame: frameData,
          features: Array.from(selectedFeatures),
        }),
      })

      const result = await response.json()
      if (!result.success) {
        throw new Error(result.error)
      }

      return result.segmented_frame
    } catch (error) {
      console.error("Video frame processing failed:", error)
      throw error
    }
  }

  async saveClassifier() {
    try {
      const response = await fetch(`${this.baseUrl}/api/save_classifier`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      })

      const result = await response.json()
      if (!result.success) {
        throw new Error(result.error)
      }

      return result
    } catch (error) {
      console.error("Classifier saving failed:", error)
      throw error
    }
  }

  async loadClassifier() {
    try {
      const response = await fetch(`${this.baseUrl}/api/load_classifier`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      })

      const result = await response.json()
      if (!result.success) {
        throw new Error(result.error)
      }

      return result
    } catch (error) {
      console.error("Classifier loading failed:", error)
      throw error
    }
  }

  async processVideoBatch(framesData, selectedFeatures, binaryOutput = true) {
    try {
      const response = await fetch(`${this.baseUrl}/api/process_video_batch`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          frames: framesData,
          features: Array.from(selectedFeatures),
          binary_output: binaryOutput,
        }),
      })

      const result = await response.json()
      if (!result.success) {
        throw new Error(result.error)
      }

      return result
    } catch (error) {
      console.error("Video batch processing failed:", error)
      throw error
    }
  }
}

// Enhanced segmentation tool with backend integration
class EnhancedSegmentationTool {
  constructor() {
    this.backend = new BackendIntegration()
    this.useBackend = true // Toggle for backend vs frontend processing
  }

  async trainClassifier() {
    if (!this.currentImage || !this.labelMask) {
      alert("Please load an image and add some labels first")
      return
    }

    try {
      this.showProgress("Training classifier...")

      if (this.useBackend) {
        // Use Python backend for training
        const imageCanvas = document.createElement("canvas")
        imageCanvas.width = this.currentImage.width
        imageCanvas.height = this.currentImage.height
        const ctx = imageCanvas.getContext("2d")
        ctx.drawImage(this.currentImage, 0, 0)

        // Convert label mask to array
        const labelArray = new Array(this.currentImage.width * this.currentImage.height)
        for (let i = 0; i < this.labelMask.data.length; i += 4) {
          labelArray[i / 4] = this.labelMask.data[i + 3]
        }

        const result = await this.backend.trainClassifier(imageCanvas, labelArray, this.selectedFeatures)

        this.classifier = { trained: true, backend: true }
        alert("Classifier trained successfully using advanced backend!")
      } else {
        // Use frontend training (original implementation)
        await this.frontendTrainClassifier()
      }

      this.hideProgress()
      this.setCurrentStep(4)
    } catch (error) {
      this.hideProgress()
      console.error("Training failed:", error)
      alert(`Training failed: ${error.message}`)
    }
  }

  async runSegmentation() {
    if (!this.classifier || !this.classifier.trained) {
      alert("Please train the classifier first")
      return
    }

    try {
      this.showProgress("Running segmentation...")

      if (this.useBackend && this.classifier.backend) {
        // Use Python backend for segmentation
        const imageCanvas = document.createElement("canvas")
        imageCanvas.width = this.currentImage.width
        imageCanvas.height = this.currentImage.height
        const ctx = imageCanvas.getContext("2d")
        ctx.drawImage(this.currentImage, 0, 0)

        const segmentedImageBase64 = await this.backend.segmentImage(
          imageCanvas,
          this.selectedFeatures,
          true, // binary output
        )

        // Display the segmented result
        const img = new Image()
        img.onload = () => {
          this.mainCtx.clearRect(0, 0, this.mainCanvas.width, this.mainCanvas.height)
          this.mainCtx.drawImage(img, 0, 0, this.mainCanvas.width, this.mainCanvas.height)
        }
        img.src = `data:image/png;base64,${segmentedImageBase64}`
      } else {
        // Use frontend segmentation (original implementation)
        await this.frontendRunSegmentation()
      }

      this.hideProgress()
      alert("Segmentation completed!")
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

    try {
      this.showProgress("Applying features...")

      if (this.useBackend) {
        // Use Python backend for feature extraction
        const imageCanvas = document.createElement("canvas")
        imageCanvas.width = this.currentImage.width
        imageCanvas.height = this.currentImage.height
        const ctx = imageCanvas.getContext("2d")
        ctx.drawImage(this.currentImage, 0, 0)

        const features = await this.backend.extractFeatures(imageCanvas, this.selectedFeatures, this.sigmaValues)

        // Display first feature as preview
        const firstFeatureName = Object.keys(features)[0]
        if (firstFeatureName) {
          const img = new Image()
          img.onload = () => {
            this.mainCtx.clearRect(0, 0, this.mainCanvas.width, this.mainCanvas.height)
            this.mainCtx.drawImage(img, 0, 0, this.mainCanvas.width, this.mainCanvas.height)
          }
          img.src = `data:image/png;base64,${features[firstFeatureName]}`
        }
      } else {
        // Use frontend feature extraction (original implementation)
        await this.frontendApplyFeatures()
      }

      this.hideProgress()
      this.setCurrentStep(3)
    } catch (error) {
      this.hideProgress()
      console.error("Feature application failed:", error)
      alert(`Feature application failed: ${error.message}`)
    }
  }

  // Add method to toggle between backend and frontend processing
  toggleProcessingMode() {
    this.useBackend = !this.useBackend
    const mode = this.useBackend ? "Backend (Python)" : "Frontend (JavaScript)"
    alert(`Processing mode switched to: ${mode}`)
  }

  // Placeholder methods for frontend processing
  async frontendTrainClassifier() {
    // Implement frontend classifier training logic here
  }

  async frontendRunSegmentation() {
    // Implement frontend segmentation logic here
  }

  async frontendApplyFeatures() {
    // Implement frontend feature extraction logic here
  }

  showProgress(message) {
    console.log(message)
  }

  hideProgress() {
    console.log("Progress hidden")
  }

  setCurrentStep(step) {
    console.log(`Current step set to: ${step}`)
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
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5)
      this.outputFolderName = `segmented_frames_${timestamp}`

      this.videoProcessingActive = true
      this.processedFrameCount = 0
      this.totalFramesToProcess = this.videoFrames.length

      this.showProgress(`Processing ${this.totalFramesToProcess} video frames...`)
      this.updateProgressBar(0)

      if (this.useBackend && this.classifier.backend) {
        // Use backend batch processing for better performance
        await this.processVideoFramesBatch()
      } else {
        // Use frontend processing frame by frame
        await this.processVideoFramesSequential()
      }
    } catch (error) {
      this.hideProgress()
      console.error("Video processing failed:", error)
      alert(`Video processing failed: ${error.message}`)
    } finally {
      this.videoProcessingActive = false
    }
  }

  async processVideoFramesBatch() {
    const batchSize = 10 // Process frames in batches to avoid memory issues
    const segmentedFrames = []

    for (let i = 0; i < this.videoFrames.length; i += batchSize) {
      if (!this.videoProcessingActive) break

      const batch = this.videoFrames.slice(i, Math.min(i + batchSize, this.videoFrames.length))
      const batchData = []

      // Prepare batch data
      for (const frameData of batch) {
        const frameImage = frameData[1]
        const frameName = frameData[2]
        const timestamp = frameData[3]

        // Convert frame to canvas and then to base64
        const frameCanvas = document.createElement("canvas")
        frameCanvas.width = frameImage.width
        frameCanvas.height = frameImage.height
        const frameCtx = frameCanvas.getContext("2d")
        frameCtx.drawImage(frameImage, 0, 0)

        batchData.push({
          image: frameCanvas.toDataURL("image/png"),
          name: frameName,
          timestamp: timestamp,
        })
      }

      try {
        // Process batch
        const batchResults = await this.backend.processVideoBatch(
          batchData,
          this.selectedFeatures,
          document.getElementById("binary-output").checked,
        )

        // Add successful results to segmented frames
        for (const result of batchResults.results) {
          if (result.success) {
            segmentedFrames.push({
              name: result.segmented_name,
              data: result.segmented_image,
              originalName: result.original_name,
              timestamp: result.timestamp,
              frameIndex: result.frame_index,
            })
          }
        }

        this.processedFrameCount += batch.length
        const progress = (this.processedFrameCount / this.totalFramesToProcess) * 100
        this.updateProgressBar(progress)
        this.updateProgressText(`Processed ${this.processedFrameCount}/${this.totalFramesToProcess} frames`)
      } catch (error) {
        console.error(`Error processing batch starting at frame ${i}:`, error)
      }
    }

    if (this.videoProcessingActive && segmentedFrames.length > 0) {
      await this.createOutputFolder(segmentedFrames)
      this.hideProgress()
      alert(
        `Video processing completed! ${segmentedFrames.length} frames processed and saved to ${this.outputFolderName}.zip`,
      )
    }
  }
}

// Initialize the enhanced application
document.addEventListener("DOMContentLoaded", () => {
  window.segmentationTool = new EnhancedSegmentationTool()

  // Add toggle button for processing mode
  const toggleBtn = document.createElement("button")
  toggleBtn.textContent = "Toggle Processing Mode"
  toggleBtn.className = "btn btn-secondary"
  toggleBtn.onclick = () => window.segmentationTool.toggleProcessingMode()

  const controlPanel = document.querySelector(".control-panel")
  if (controlPanel) {
    controlPanel.appendChild(toggleBtn)
  }
})
