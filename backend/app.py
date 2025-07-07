from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.helper import load_models, load_metadata_features, determine_model, verify_valid_situation, verify_valid_shot_type, verify_all_features_present
from utils.preprocess import preprocess
import json

# Add CORS, and load models and features. Only allow requests from https://atredshaw.github.io/
app = Flask(__name__)
CORS(app)
app.config['CORS_ORIGINS'] = ['https://atredshaw.github.io']
models = load_models()
features = load_metadata_features()

# Load heatmap data once at startup
heatmap_data_path = 'heatmaps/heatmaps.json'
try:
    with open(heatmap_data_path, 'r') as f:
        heatmap_data = json.load(f)
except FileNotFoundError:
    heatmap_data = None

# ---------- ROUTES ----------
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

@app.route('/redshaw-xg/api/predict', methods=['POST'])
def predict():
    """POST endpoint to predict based on input features and select appropriate model."""
    try:
        data = request.get_json()
        
        # Extract inputs
        x = data.get('x', None)
        y = data.get('y', None)
        situation = data.get('situation', None )
        shot_type = data.get('shot_type', None)
        normalisation = data.get('normalisation', {})

        if x is None or y is None:
            return jsonify({'error': 'Missing x or y coordinate'}), 400

        # Verify valid inputs
        if not verify_valid_situation(situation):
            return jsonify({'error': 'Invalid situation - Valid values are: OpenPlay, SetPiece, DirectFreekick, FromCorner, Penalty'}), 400
        if not verify_valid_shot_type(shot_type):
            return jsonify({'error': 'Invalid shot type - Valid values are: Head, RightFoot, LeftFoot, OtherBodyPart'}), 400

        # Call the determine model function
        chosen_model_dic = determine_model(x, y, situation, shot_type, normalisation)

        if chosen_model_dic['error'] != None:
            return jsonify({'error': chosen_model_dic['error']}), 400

        chosen_model = chosen_model_dic['chosen_model']
        chosen_model_features = features[chosen_model]

        # Update the x and y values in case they've now been normalised
        x = chosen_model_dic['x']
        y = chosen_model_dic['y']

        # Check for and display any errors
        if chosen_model_dic['error'] != None:
            return jsonify({'error': chosen_model_dic['error']}), 400

        # Preprocess the input data
        X = preprocess(x, y, situation, shot_type, chosen_model, chosen_model_features)

        # Verify all features are present
        if not verify_all_features_present(X, chosen_model_features):
            return jsonify({'error': 'Not all features are present'}), 400
        else:
            X = X[chosen_model_features]

        # Make the prediction
        prediction = models[chosen_model].predict_proba(X)[:, 1]

        if situation == 'Penalty':
            prediction[0] = 0.76
            x = 0.895
            y = 0.5

        # Return the prediction
        return jsonify({
            'xG': round(prediction[0], 2),
            'inputs': {
                'x': x,
                'y': y,
                'situation': situation,
                'shot_type': shot_type,
                'normalisation': normalisation
            },
            'chosen_model': chosen_model,
            'chosen_model_features': chosen_model_features
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('redshaw-xg/api/predict/grid', methods=['GET'])
def predict_grid():
    """GET endpoint to retrieve a heatmap grid for a given situation and shot type."""
    if heatmap_data is None:
        return jsonify({'error': f'Heatmap data file not found at {heatmap_data_path}'}), 500
        
    try:
        situation = request.args.get('situation') # Returns None if not present
        shot_type = request.args.get('shot_type') # Returns None if not present
        max_length = request.args.get('max_length', type=float)
        max_width = request.args.get('max_width', type=float)

        # Custom validation, as 'Penalty' is not allowed and error message should reflect that.
        valid_situations = ['OpenPlay', 'SetPiece', 'DirectFreekick', 'FromCorner']
        if situation is not None and situation not in valid_situations:
            if situation == 'Penalty':
                 return jsonify({'error': "The 'Penalty' situation is not available for grid prediction."}), 400
            return jsonify({'error': f"Invalid situation. Valid values are: {', '.join(valid_situations)}"}), 400

        if not verify_valid_shot_type(shot_type):
            return jsonify({'error': 'Invalid shot type - Valid values are: Head, RightFoot, LeftFoot, OtherBodyPart'}), 400

        # Convert None to 'None' string for dict key access, matching heatmaps.json
        situation_key = situation if situation is not None else 'None'
        shot_type_key = shot_type if shot_type is not None else 'None'

        # Retrieve data, creating a copy of grid_def to avoid modifying the global object
        grid_definition = heatmap_data['grid_definition'].copy()
        heatmap = heatmap_data['heatmaps'][situation_key][shot_type_key]

        # Scale coordinates if dimensions are provided
        if max_length is not None and max_width is not None:
            if max_length <= 0 or max_width <= 0:
                return jsonify({'error': 'max_length and max_width must be positive numbers.'}), 400
            
            grid_definition['x_coords'] = [x * max_length for x in grid_definition['x_coords']]
            grid_definition['y_coords'] = [y * max_width for y in grid_definition['y_coords']]
        
        return jsonify({
            'grid_definition': grid_definition,
            'heatmap': heatmap
        }), 200

    except KeyError:
        return jsonify({'error': 'Data not found for the specified situation and shot_type combination.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run in production
    app.run(debug=False)
