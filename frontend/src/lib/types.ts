export type DeviceConfig = {
  gpu_available: boolean
  device: string
  api_url: string
}

export type PredictResponse = {
  filename: string
  success: boolean
  prediction: {
    metadata: {
      image_path: string
      inference_time_ms: number
    }
    prediction: {
      class: string
      class_id: number
      confidence: number
      confidence_percent: number
    }
    probabilities: Record<string, number>
  }
}

export type HistoryEntry = {
  image: string
  classCode: string
  className: string
  confidence: number
}

export type SessionStats = {
  totalAnalyses: number
  totalConfidence: number
  totalTime: number
}
