# Change to the backend directory
cd "$(dirname "$0")"

echo "=== Starting Data Pipeline ==="
echo "Current directory: $(pwd)"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found, installing requirements..."
    pip install -r requirements.txt
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Run scraping
echo "\n=== Running Data Scraping ==="
python -m src.preparation.scraping
if [ $? -ne 0 ]; then
    echo "Error: Data scraping failed"
    exit 1
fi

# Run cleansing
echo "\n=== Running Data Cleansing ==="
python -m src.preparation.cleansing
if [ $? -ne 0 ]; then
    echo "Error: Data cleansing failed"
    exit 1
fi

# Run preprocessing
echo "\n=== Running Data Preprocessing ==="
python -m src.preparation.preprocessing
if [ $? -ne 0 ]; then
    echo "Error: Data preprocessing failed"
    exit 1
fi

# Run model training
echo "\n=== Running Model Training ==="
python -m src.modelling.model
if [ $? -ne 0 ]; then
    echo "Error: Model training failed"
    exit 1
fi

# Run heatmap generation
echo "\n=== Generating Heatmap Data ==="
python -m src.modelling.generate_heatmaps
if [ $? -ne 0 ]; then
    echo "Error: Heatmap generation failed"
    exit 1
fi

# Export trained models to ONNX and copy heatmaps.json to the frontend.
# This replaces the Flask API — all inference now runs client-side in the browser.
echo "\n=== Exporting Models to ONNX ==="
python export_to_onnx.py
if [ $? -ne 0 ]; then
    echo "Error: ONNX model export failed"
    exit 1
fi

echo "\n=== Pipeline completed successfully ==="
