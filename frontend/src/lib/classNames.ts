export const CLASS_LABELS: Record<string, string> = {
  mel: 'Melanoma',
  nv: 'Melanocytic Nevus',
  bcc: 'Basal Cell Carcinoma',
  akiec: 'Actinic Keratosis',
  bkl: 'Benign Keratosis',
  df: 'Dermatofibroma',
  vasc: 'Vascular Lesion',
}

export function labelForClass(code: string): string {
  return CLASS_LABELS[code] ?? code
}
