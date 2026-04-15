#!/bin/bash
# Skin Cancer Detection API - cURL Examples
# 
# These examples demonstrate how to interact with the Skin Cancer Detection API
# using cURL commands. All examples assume the API is running on http://localhost:5000

API_URL="http://localhost:5000"

echo "=================================="
echo "SKIN CANCER API - cURL EXAMPLES"
echo "=================================="
echo ""

# Example 1: Health Check
echo "1. Health Check"
echo "   Command: curl -X GET http://localhost:5000/health"
echo "   Description: Verify API is running and healthy"
echo ""
# curl -X GET "${API_URL}/health"

# Example 2: Get Model Info
echo "2. Get Model Information"
echo "   Command: curl -X GET http://localhost:5000/info"
echo "   Description: Get model architecture, accuracy, and class information"
echo ""
# curl -X GET "${API_URL}/info" | jq '.'

# Example 3: Single Image Prediction
echo "3. Single Image Prediction"
echo "   Command: curl -X POST http://localhost:5000/predict -F 'image=@skin_lesion.jpg'"
echo "   Description: Predict disease from a single image"
echo ""
# curl -X POST "${API_URL}/predict" \
#   -F "image=@skin_lesion.jpg" | jq '.'

# Example 4: Batch Prediction
echo "4. Batch Image Prediction"
echo "   Command: curl -X POST http://localhost:5000/batch-predict \\"
echo "            -F 'images=@img1.jpg' -F 'images=@img2.jpg' -F 'images=@img3.jpg' \\"
echo "            -F 'return_all_predictions=false'"
echo "   Description: Predict diseases from multiple images at once"
echo ""
# curl -X POST "${API_URL}/batch-predict" \
#   -F "images=@img1.jpg" \
#   -F "images=@img2.jpg" \
#   -F "images=@img3.jpg" \
#   -F "return_all_predictions=false" | jq '.'

# Example 5: Pretty-printed Results
echo "5. Pretty-Print JSON Results"
echo "   Add '| jq .' to any command to format JSON output"
echo ""
echo "   Example:"
echo "   curl -s http://localhost:5000/info | jq '.'"
echo ""

# Example 6: Save Results to File
echo "6. Save Results to File"
echo "   curl -s http://localhost:5000/predict -F 'image=@img.jpg' > result.json"
echo ""

# Example 7: Extract Specific Field
echo "7. Extract Specific Field"
echo "   curl -s http://localhost:5000/predict -F 'image=@img.jpg' | jq '.top_class'"
echo ""

# Example 8: Check Response Headers
echo "8. View Response Headers"
echo "   curl -i http://localhost:5000/health"
echo ""

# Example 9: Measure Request Time
echo "9. Measure Request Time"
echo "   curl -w \"@curl-format.txt\" -o /dev/null -s http://localhost:5000/health"
echo ""

# Example 10: Basic Authentication (if enabled)
echo "10. With Authentication Token (if enabled)"
echo "    curl -H 'Authorization: Bearer YOUR_TOKEN' http://localhost:5000/info"
echo ""

# Actual functional examples (commented out for safety)
cat << 'EOF'

===== FUNCTIONAL EXAMPLES (Uncomment to use) =====

# Test 1: Health Check
# curl -s http://localhost:5000/health | jq '.'

# Test 2: Model Info with Pretty Print
# curl -s http://localhost:5000/info | jq '.'

# Test 3: Get Just the Status from Health Check
# curl -s http://localhost:5000/health | jq '.status'

# Test 4: Get Just the Accuracy from Model Info
# curl -s http://localhost:5000/info | jq '.test_accuracy'

# Test 5: Predict with Image (single file)
# curl -X POST http://localhost:5000/predict \
#   -F "image=@/path/to/image.jpg" | jq '.predictions'

# Test 6: Batch Predict with Multiple Images
# curl -X POST http://localhost:5000/batch-predict \
#   -F "images=@/path/to/image1.jpg" \
#   -F "images=@/path/to/image2.jpg" \
#   -F "images=@/path/to/image3.jpg" | jq '.summary'

# Test 7: Get Response Time
# curl -s -w "Time: %{time_total}s\n" http://localhost:5000/health

# Test 8: Save Full Response
# curl -s http://localhost:5000/info > api_info.json

# Test 9: Loop Through Multiple Images (Bash)
# for img in *.jpg; do
#   echo "Processing: $img"
#   curl -s http://localhost:5000/predict -F "image=@$img" | jq '.top_class'
# done

# Test 10: Prediction with Custom Confidence Threshold
# curl -X POST http://localhost:5000/predict \
#   -F "image=@image.jpg" \
#   -F "confidence_threshold=0.7" | jq '.'

===== API ENDPOINTS =====

GET /health
  Returns: API health status
  Response: {"status": "healthy", "timestamp": "..."}

GET /info
  Returns: Model information and statistics
  Response: {"model_type": "resnet50", "test_accuracy": 0.8029, ...}

POST /predict
  Input: Single image file (form: file='image')
  Returns: Prediction result
  Response: {"top_class": "Melanoma", "confidence": 0.95, ...}

POST /batch-predict
  Input: Multiple image files (form: files='images')
  Returns: Batch prediction results
  Response: {"predictions": [...], "summary": {...}}

POST /predict-from-bytes
  Input: Image bytes (form: file='image')
  Returns: Prediction result
  Response: {"top_class": "...", "confidence": ..., ...}

===== USEFUL jq FILTERS =====

# Get just the class name
jq '.top_class'

# Get just the confidence score
jq '.confidence'

# Get predicted ID
jq '.predicted_id'

# Get all class predictions
jq '.all_predictions'

# Get inference time
jq '.inference_time_ms'

# Filter multiple fields
jq '{class: .top_class, confidence: .confidence, time: .inference_time_ms}'

# For batch results, get all class names
jq '.predictions[].top_class'

# For batch results, get average confidence
jq '.predictions[].confidence | add / length'

===== TROUBLESHOOTING =====

# Connection Refused
# - Make sure API is running: docker-compose up -d
# - Check port is correct (default: 5000)

# Command not found: jq
# - Install jq: apt-get install jq (Linux) or brew install jq (Mac)
# - Or use Python for JSON parsing: ... | python -m json.tool

# File not found
# - Use absolute path: /full/path/to/image.jpg
# - Or relative path from current directory

# Timeout Error
# - Increase timeout: curl --max-time 60
# - Check API performance

# 404 Not Found
# - Verify endpoint path is correct
# - Check API prefix path if behind reverse proxy
EOF
