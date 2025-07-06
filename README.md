# xG-Plotter: An Interactive Expected Goals (xG) Prediction and Visualization Tool

This project is a web application that allows users to interactively explore a custom Expected Goals (xG) model for football. It provides a visual interface to plot shots on a football pitch, get xG predictions, and view heatmaps of the model's predictions.

## Features

*   **Interactive Pitch:** Click on a football pitch to get an instant xG prediction for that location.
*   **Dynamic Model Selection:** The backend intelligently selects the most appropriate model based on the input features (shot location, situation, shot type).
*   **xG Heatmaps:** Visualize the xG model's predictions across the pitch with interactive heatmaps that respond to different situations and shot types.
*   **Match Plotter:** Create and save entire matches. Plot shots for home and away teams, and see the cumulative xG for each team.
*   **Local Storage:** Match data is saved in your browser using IndexedDB, so you can come back to your plotted matches later.
*   **REST API:** A Flask-based backend provides a REST API for xG predictions and heatmap data.
*   **Data Pipeline:** A complete data pipeline for scraping, cleansing, preprocessing, and training the xG models.

## Live Demo

You can access the live version of the application here: [https://atredshaw.github.io/xg-plotter/frontend/index.html](https://atredshaw.github.io/xg-plotter/frontend/index.html)

## How It Works

The project is divided into two main components: a Python-based backend and a vanilla JavaScript frontend.

### Backend

The backend is a Flask application that serves the xG models and heatmap data.

#### Data Pipeline

The entire data processing and model training pipeline is orchestrated by the `run_pipeline.sh` script. This script executes the following Python scripts in order:

1.  **`src/preparation/scraping.py`**: Scrapes shot data from football data providers. Settings can be managed for this via the `config.yaml` file. The scraped data is saved in the `data/raw/` directory.
2.  **`src/preparation/cleansing.py`**: Cleans the raw scraped data, handling missing values and inconsistencies, and saves the cleansed data to `data/cleansed/`.
3.  **`src/preparation/preprocessing.py`**: Performs feature engineering. It calculates features like `distance_to_goal` and `angle_to_goal`, and one-hot encodes categorical features like `situation` and `shotType`. The preprocessed data is saved to `data/preprocessed/`.
4.  **`src/modelling/model.py`**: Trains four different logistic regression models using `scikit-learn`:
    *   `basic_model`: Uses only the shot coordinates, distance, and angle.
    *   `situation_model`: Adds one-hot encoded game situations (e.g., Open Play, Set Piece).
    *   `shottype_model`: Adds one-hot encoded shot types (e.g., Head, Right Foot).
    *   `advanced_model`: Uses all available features.
    The trained models are saved as `.joblib` files.
5.  **`src/modelling/generate_heatmaps.py`**: Pre-calculates the xG values for a grid of points on the pitch for various situation and shot type combinations. The results are saved to `heatmaps/heatmaps.json` to be served by the API.

#### API Endpoints

The Flask application (`app.py`) exposes two API endpoints:

*   `POST /api/predict`: Accepts shot data (x, y, situation, shot_type) and returns a predicted xG value. It automatically determines the best model to use based on the provided inputs.
*   `GET /api/predict/grid`: Returns the pre-generated heatmap data for a given situation and shot type.

### Frontend

The frontend is built with HTML, Tailwind CSS, and vanilla JavaScript. It consists of three main pages:

*   **Home (`index.html`)**: The main landing page with the interactive pitch for single shot predictions.
*   **Heatmaps (`heatmap.html`)**: The page for visualizing the xG heatmaps.
*   **Plotter (`plotter.html`)**: The advanced tool for plotting and saving full matches.

The JavaScript files (`js/index.js`, `js/heatmap.js`, `js/plotter.js`) handle:
*   Drawing the football pitch on an HTML `<canvas>`.
*   Handling user interactions like mouse clicks and form inputs.
*   Making `fetch` requests to the backend API.
*   Dynamically updating the DOM with prediction results and heatmaps.
*   The `plotter.js` file also manages storing and retrieving match data from the browser's IndexedDB.

## Running the Project Locally

To run this project on your local machine, you'll need to run the backend server and serve the frontend files.

### Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the data pipeline (optional):**
    If you want to retrain the models with new data, run the pipeline script. This will take some time.
    ```bash
    ./run_pipeline.sh
    ```

5.  **Start the Flask server:**
    ```bash
    flask run
    ```
    The backend server will be running at `http://127.0.0.1:5000`.

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Serve the HTML files.** You can use a simple HTTP server. If you have Python installed, you can use its built-in server:
    ```bash
    python3 -m http.server
    ```

3.  **Access the application:**
    Open your web browser and go to `http://localhost:8000`.

## Project Structure

```
.
├── backend/
│   ├── app.py              # Flask API application
│   ├── run_pipeline.sh     # Main script to run the data pipeline
│   ├── requirements.txt    # Python dependencies
│   ├── data/               # Raw, cleansed, and preprocessed data
│   ├── models/             # Trained machine learning models
│   ├── src/                # Source code for the pipeline
│   │   ├── preparation/    # Scraping, cleansing, preprocessing
│   │   └── modelling/      # Model training and heatmap generation
│   └── utils/              # Helper functions
└── frontend/
    ├── index.html          # Main page with interactive plotter
    ├── plotter.html        # Page for plotting full matches
    ├── heatmap.html        # Page for viewing xG heatmaps
    ├── css/                # CSS files
    └── js/                 # JavaScript files
```

## Technologies Used

*   **Backend:** Python, Flask, scikit-learn, pandas, NumPy
*   **Frontend:** HTML, CSS, JavaScript (vanilla), Tailwind CSS
*   **Database:** IndexedDB (for client-side storage)
*   **Deployment:** Render (for the backend), GitHub Pages (for the frontend)