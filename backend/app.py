from flask import Flask, jsonify, request
from utils.helper import load_models, load_metadata_features, determine_model, verify_valid_situation, verify_valid_shot_type, verify_all_features_present
from utils.preprocess import preprocess

app = Flask(__name__)
models = load_models()
features = load_metadata_features()

# ---------- ROUTES ----------
@app.route('/api/predict', methods=['POST'])
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

        # Default value to penalty average as no other changing circumstances here (76% scored on average)
        # Including approximate normalised pitch position of penalty spot
        if situation == 'Penalty':
            return jsonify({
                'xG': 0.76,
                'inputs': {
                    'x': 0.895,
                    'y': 0.5,
                    'situation': situation,
                    'shot_type': shot_type,
                    'normalisation': normalisation
                },
                'chosen_model': 'penalty_default',
                'chosen_model_features': ['is_Penalty']
            }), 200

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

if __name__ == '__main__':
    app.run(debug=True)
