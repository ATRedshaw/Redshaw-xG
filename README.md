# xG-Plotter: An Interactive Expected Goals (xG) Prediction and Visualisation Tool

A web application for interactively exploring a custom Expected Goals (xG) model for football. It provides a visual interface for plotting shots on a pitch, obtaining instant xG predictions and viewing pre-calculated heatmaps, all running entirely in the browser with no server dependency.

## Features

- **Interactive Pitch:** Click anywhere on a football pitch to get an instant xG prediction for that location.
- **Dynamic Model Selection:** The application automatically selects the most appropriate model based on the provided input features (shot location, situation and shot type).
- **xG Heatmaps:** Visualise the model's predictions across the pitch with interactive heatmaps that respond to different situations and shot types.
- **Match Plotter:** Create and save entire matches, plot shots for home and away teams and track the cumulative xG for each side.
- **Local Storage:** Match data is persisted in the browser using IndexedDB, allowing you to return to previously plotted matches.
- **Data Pipeline:** A complete Python pipeline for scraping, cleansing, preprocessing and training the xG models.

## Live Demo

The live version of the application is available here: [https://atredshaw.github.io/Redshaw-xG/index.html](https://atredshaw.github.io/Redshaw-xG/index.html)

## How It Works

The project is divided into two components: a Python-based data pipeline and a vanilla JavaScript frontend.

### Backend (Data Pipeline)

The backend is responsible solely for data processing and model training. There is no running server. The entire pipeline is orchestrated by the `run_pipeline.sh` script, which executes the following steps in order:

1. **`src/preparation/scraping.py`**: Scrapes shot data from football data providers. Pipeline settings are configured via `src/config.yaml`. Scraped data is saved to `data/raw/`.
2. **`src/preparation/cleansing.py`**: Cleans the raw data, handling missing values and inconsistencies. Output is saved to `data/cleansed/`.
3. **`src/preparation/preprocessing.py`**: Performs feature engineering, calculating `distance_to_goal` and `angle_to_goal` and one-hot encoding categorical features such as `situation` and `shotType`. Preprocessed data is saved to `data/preprocessed/`.
4. **`src/modelling/model.py`**: Trains four logistic regression models using scikit-learn:
    - `basic_model` — Shot coordinates, distance and angle only.
    - `situation_model` — Adds one-hot encoded game situations (e.g. Open Play, Set Piece).
    - `shottype_model` — Adds one-hot encoded shot types (e.g. Head, Right Foot).
    - `advanced_model` — Uses all available features including situation-by-shot-type interaction terms.

    Each trained model is saved as a `.joblib` file alongside a `metadata.json` describing its feature list.

5. **`src/modelling/generate_heatmaps.py`**: Pre-calculates xG values across a grid of pitch coordinates for each situation and shot type combination. Results are saved to `heatmaps/heatmaps.json`.
6. **`export_to_onnx.py`**: Converts each `.joblib` model to ONNX format using `skl2onnx` and copies the resulting `.onnx` files and their `metadata.json` counterparts into `frontend/models/`. The heatmap data is also copied to `frontend/data/heatmaps.json`. After this step, the frontend has everything it needs to run without any server.

### Frontend

The frontend is built with HTML, Tailwind CSS and vanilla JavaScript. It runs entirely in the browser by loading the ONNX models directly and performing inference client-side via [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html).

#### Why ONNX Runtime Web instead of a Flask backend

The project originally used a Flask server to handle xG predictions via a REST API. This required a persistently running server, introduced network latency on every prediction and created a hosting dependency that made the application fragile to deploy and maintain.

Migrating to ONNX Runtime Web removes all of those constraints. The trained scikit-learn pipelines are exported to the open ONNX format once, then loaded and executed entirely within the browser using WebAssembly. Predictions are instantaneous, the application works offline after the initial page load and the whole project can be hosted as a static site on GitHub Pages with no backend infrastructure required.

The `js/xg_inference.js` module is a complete client-side port of the original Python inference logic, covering input validation, coordinate normalisation, model selection and feature engineering. It exposes a single `XG_INFERENCE.predict()` function consumed by the rest of the frontend.

#### Pages

- **Home (`index.html`)**: The landing page with the interactive pitch for single-shot predictions.
- **Heatmaps (`heatmap.html`)**: Visualisation of pre-calculated xG heatmaps.
- **Plotter (`plotter.html`)**: Advanced tool for plotting and saving full matches.

The JavaScript files handle drawing the pitch on an HTML `<canvas>`, responding to user interactions, running ONNX inference and updating the DOM with results. `plotter.js` additionally manages storing and retrieving match data from IndexedDB.

## Running the Project Locally

### Backend Setup (pipeline only)

1. **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2. **Run the full pipeline** to retrain models on fresh data:
    ```bash
    ./run_pipeline.sh
    ```

    After this step, `frontend/models/` and `frontend/data/` will contain everything the frontend needs.

### Frontend Setup

1. **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2. **Serve the files** using a simple HTTP server. Python's built-in server works well:
    ```bash
    python3 -m http.server
    ```

3. **Open the application** at `http://localhost:8000`.

## Project Structure

```
.
├── backend/
│   ├── export_to_onnx.py   # Converts .joblib models to ONNX and copies assets to frontend
│   ├── run_pipeline.sh     # Orchestrates the full data and training pipeline
│   ├── requirements.txt    # Python dependencies
│   ├── data/               # Raw, cleansed and preprocessed data
│   ├── models/             # Trained scikit-learn models (.joblib)
│   ├── heatmaps/           # Pre-calculated heatmap data
│   ├── src/                # Pipeline source code
│   │   ├── preparation/    # Scraping, cleansing and preprocessing
│   │   └── modelling/      # Model training and heatmap generation
│   └── utils/              # Shared helper functions
└── frontend/
    ├── index.html          # Main page with interactive pitch
    ├── plotter.html        # Full match plotting tool
    ├── heatmap.html        # xG heatmap visualisation
    ├── css/                # Stylesheets
    ├── js/                 # JavaScript modules
    ├── models/             # ONNX models and metadata (generated by export_to_onnx.py)
    └── data/               # Heatmap data (generated by export_to_onnx.py)
```

## Technologies Used

- **Pipeline:** Python, scikit-learn, pandas, NumPy, skl2onnx
- **Frontend:** HTML, CSS, JavaScript (vanilla), Tailwind CSS, ONNX Runtime Web
- **Client-side storage:** IndexedDB
- **Deployment:** GitHub Pages
