/**
 * xg_inference.js
 *
 * Client-side xG prediction module using ONNX Runtime Web.
 *
 * This module is a complete port of the Python Flask backend pipeline:
 *   - Input validation  (mirrors utils/helper.py)
 *   - Coordinate normalisation and model selection (mirrors determine_model)
 *   - Feature engineering (mirrors utils/preprocess.py)
 *   - ONNX inference with lazy model caching
 *
 * Depends on: onnxruntime-web loaded globally as `ort` before this script.
 *
 * Public API:
 *   XG_INFERENCE.predict(x, y, situation, shotType, normalisation) → Promise<{xG, x, y, chosenModel}>
 */
const XG_INFERENCE = (() => {
    'use strict';

    // --- Constants (mirror preprocess.py) ---
    const GOAL_CENTER = [1.0, 0.5];
    const GOAL_POSTS  = [[1.0, 0.45], [1.0, 0.55]];

    // ORT WASM files must be loaded from the CDN path since GitHub Pages
    // does not serve them locally.
    const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

    /**
     * Feature lists for each model, in the exact column order expected by the
     * corresponding ONNX model. These mirror the 'features' arrays in each
     * metadata.json and must remain in sync with the Python training code.
     * @type {Object.<string, string[]>}
     */
    const MODEL_FEATURES = {
        basic_model: [
            'X', 'Y', 'distance_to_goal', 'angle_to_goal',
        ],
        shottype_model: [
            'X', 'Y', 'distance_to_goal', 'angle_to_goal',
            'shotType_Head', 'shotType_LeftFoot', 'shotType_OtherBodyPart', 'shotType_RightFoot',
        ],
        situation_model: [
            'X', 'Y', 'distance_to_goal', 'angle_to_goal',
            'situation_DirectFreekick', 'situation_FromCorner', 'situation_OpenPlay',
            'situation_Penalty', 'situation_SetPiece',
        ],
        advanced_model: [
            'X', 'Y', 'distance_to_goal', 'angle_to_goal',
            'situation_DirectFreekick', 'situation_FromCorner', 'situation_OpenPlay',
            'situation_Penalty', 'situation_SetPiece',
            'shotType_Head', 'shotType_LeftFoot', 'shotType_OtherBodyPart', 'shotType_RightFoot',
            'interaction_DirectFreekick_LeftFoot', 'interaction_DirectFreekick_RightFoot',
            'interaction_FromCorner_Head', 'interaction_FromCorner_LeftFoot',
            'interaction_FromCorner_OtherBodyPart', 'interaction_FromCorner_RightFoot',
            'interaction_OpenPlay_Head', 'interaction_OpenPlay_LeftFoot',
            'interaction_OpenPlay_OtherBodyPart', 'interaction_OpenPlay_RightFoot',
            'interaction_Penalty_LeftFoot', 'interaction_Penalty_RightFoot',
            'interaction_SetPiece_Head', 'interaction_SetPiece_LeftFoot',
            'interaction_SetPiece_OtherBodyPart', 'interaction_SetPiece_RightFoot',
        ],
    };

    // Lazy-loaded ONNX InferenceSession cache, keyed by model name.
    const _sessionCache = new Map();

    // ---------------------------------------------------------------------------
    // Validation helpers (mirror helper.py)
    // ---------------------------------------------------------------------------

    const VALID_SITUATIONS = new Set(['OpenPlay', 'SetPiece', 'DirectFreekick', 'FromCorner', 'Penalty']);
    const VALID_SHOT_TYPES = new Set(['Head', 'RightFoot', 'LeftFoot', 'OtherBodyPart']);

    /**
     * Returns true when the situation is null/empty or a recognised value.
     * @param {string|null|undefined} situation
     * @returns {boolean}
     */
    function isValidSituation(situation) {
        return situation == null || situation === '' || VALID_SITUATIONS.has(situation);
    }

    /**
     * Returns true when the shot type is null/empty or a recognised value.
     * @param {string|null|undefined} shotType
     * @returns {boolean}
     */
    function isValidShotType(shotType) {
        return shotType == null || shotType === '' || VALID_SHOT_TYPES.has(shotType);
    }

    // ---------------------------------------------------------------------------
    // Normalisation and model selection (mirror determine_model in helper.py)
    // ---------------------------------------------------------------------------

    /**
     * Validates and normalises coordinates, then selects the appropriate model
     * name based on which categorical inputs are present.
     *
     * @param {number}      x             Raw or normalised x-coordinate.
     * @param {number}      y             Raw or normalised y-coordinate.
     * @param {string|null} situation     Shot situation or null.
     * @param {string|null} shotType      Shot type or null.
     * @param {Object}      normalisation Normalisation config object.
     * @returns {{ chosenModel: string, x: number, y: number,
     *             situation: string|null, shotType: string|null, error: string|null }}
     */
    function determineModel(x, y, situation, shotType, normalisation) {
        if (normalisation == null || !Object.prototype.hasOwnProperty.call(normalisation, 'is_normalised')) {
            return { error: "Normalisation object must contain an 'is_normalised' key." };
        }

        if (normalisation.is_normalised === false) {
            const maxWidth  = normalisation.max_pitch_width;
            const maxLength = normalisation.max_pitch_length;

            if (maxWidth == null || maxLength == null) {
                return {
                    error: 'max_pitch_width and max_pitch_length are required when is_normalised is false.',
                };
            }

            const w = parseFloat(maxWidth);
            const l = parseFloat(maxLength);

            if (isNaN(w) || isNaN(l)) {
                return { error: 'max_pitch_width and max_pitch_length must be positive numbers.' };
            }

            if (w <= 0 || l <= 0) {
                return { error: 'The maximum width and length of the pitch cannot be a negative value.' };
            }

            x = x / l;
            y = y / w;
        }

        if (x < 0 || x > 1 || y < 0 || y > 1) {
            return {
                error: 'Coordinates have been incorrectly claimed as normalised — ensure values are between 0 and 1.',
            };
        }

        // Normalise empty strings to null so downstream logic is consistent.
        const sit = (situation === '' || situation == null) ? null : situation;
        const sht = (shotType  === '' || shotType  == null) ? null : shotType;

        let chosenModel;
        if (sit === null && sht === null)      chosenModel = 'basic_model';
        else if (sit === null && sht !== null) chosenModel = 'shottype_model';
        else if (sit !== null && sht === null) chosenModel = 'situation_model';
        else                                   chosenModel = 'advanced_model';

        return { chosenModel, x, y, situation: sit, shotType: sht, error: null };
    }

    // ---------------------------------------------------------------------------
    // Feature engineering (mirror preprocess.py)
    // ---------------------------------------------------------------------------

    /**
     * Builds an ordered Float32Array of model inputs for a single shot.
     * All feature computation mirrors the Python preprocess() function exactly.
     *
     * @param {number}      x         Normalised x-coordinate (0–1).
     * @param {number}      y         Normalised y-coordinate (0–1).
     * @param {string|null} situation Situation key or null.
     * @param {string|null} shotType  Shot-type key or null.
     * @param {string[]}    features  Ordered feature name list from MODEL_FEATURES.
     * @returns {Float32Array}
     */
    function buildFeatureVector(x, y, situation, shotType, features) {
        // Initialise all features to 0.0.
        const featureMap = Object.fromEntries(features.map(f => [f, 0.0]));

        // Coordinate features.
        if ('X' in featureMap) featureMap['X'] = x;
        if ('Y' in featureMap) featureMap['Y'] = y;

        // Euclidean distance to goal centre.
        if ('distance_to_goal' in featureMap) {
            featureMap['distance_to_goal'] = Math.sqrt(
                Math.pow(x - GOAL_CENTER[0], 2) + Math.pow(y - GOAL_CENTER[1], 2),
            );
        }

        // Angle between vectors to the two goalposts (radians).
        if ('angle_to_goal' in featureMap) {
            const v1x = GOAL_POSTS[0][0] - x;
            const v1y = GOAL_POSTS[0][1] - y;
            const v2x = GOAL_POSTS[1][0] - x;
            const v2y = GOAL_POSTS[1][1] - y;

            const dot  = v1x * v2x + v1y * v2y;
            const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
            const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);

            if (mag1 === 0 || mag2 === 0) {
                featureMap['angle_to_goal'] = 0;
            } else {
                const cosAngle = Math.max(-1.0, Math.min(1.0, dot / (mag1 * mag2)));
                featureMap['angle_to_goal'] = Math.acos(cosAngle);
            }
        }

        // One-hot situation feature.
        if (situation !== null) {
            const key = `situation_${situation}`;
            if (key in featureMap) featureMap[key] = 1.0;
        }

        // One-hot shot-type feature.
        if (shotType !== null) {
            const key = `shotType_${shotType}`;
            if (key in featureMap) featureMap[key] = 1.0;
        }

        // One-hot interaction feature (situation × shot type).
        if (situation !== null && shotType !== null) {
            const key = `interaction_${situation}_${shotType}`;
            if (key in featureMap) featureMap[key] = 1.0;
        }

        // Return values in the exact column order expected by the ONNX model.
        return new Float32Array(features.map(f => featureMap[f]));
    }

    // ---------------------------------------------------------------------------
    // ONNX model loading with session cache
    // ---------------------------------------------------------------------------

    /**
     * Lazily creates and caches an ONNX InferenceSession for the given model name.
     * Subsequent calls for the same model return the cached session immediately.
     *
     * @param {string} modelName  Key from MODEL_FEATURES (e.g., 'basic_model').
     * @returns {Promise<ort.InferenceSession>}
     */
    async function getSession(modelName) {
        if (_sessionCache.has(modelName)) {
            return _sessionCache.get(modelName);
        }

        // Ensure ORT can locate its WASM binaries even on a CDN-hosted page.
        ort.env.wasm.wasmPaths = ORT_CDN;

        // Path is relative to the HTML document, which lives at the frontend root.
        const modelPath = `models/${modelName}/model.onnx`;
        const session = await ort.InferenceSession.create(modelPath);
        _sessionCache.set(modelName, session);
        return session;
    }

    // ---------------------------------------------------------------------------
    // Public prediction function
    // ---------------------------------------------------------------------------

    /**
     * Runs the full xG prediction pipeline entirely in the browser.
     *
     * Replicates the behaviour of the Flask /redshaw-xg/api/predict endpoint,
     * including the Penalty hard-coded override.
     *
     * @param {number}      x             X-coordinate (raw metres or normalised).
     * @param {number}      y             Y-coordinate (raw metres or normalised).
     * @param {string|null} situation     Situation string or null/''.
     * @param {string|null} shotType      Shot-type string or null/''.
     * @param {Object}      normalisation { is_normalised: bool, max_pitch_width?, max_pitch_length? }
     * @returns {Promise<{ xG: number, x: number, y: number, chosenModel: string }>}
     * @throws {Error} on invalid inputs.
     */
    async function predict(x, y, situation, shotType, normalisation) {
        if (x == null || y == null) {
            throw new Error('Missing x or y coordinate.');
        }

        x = parseFloat(x);
        y = parseFloat(y);

        if (isNaN(x) || isNaN(y)) {
            throw new Error('x and y must be numeric values.');
        }

        if (!isValidSituation(situation)) {
            throw new Error(
                `Invalid situation '${situation}'. Valid values: OpenPlay, SetPiece, DirectFreekick, FromCorner, Penalty.`,
            );
        }

        if (!isValidShotType(shotType)) {
            throw new Error(
                `Invalid shot type '${shotType}'. Valid values: Head, RightFoot, LeftFoot, OtherBodyPart.`,
            );
        }

        const determined = determineModel(x, y, situation, shotType, normalisation);
        if (determined.error) {
            throw new Error(determined.error);
        }

        const { chosenModel, x: normX, y: normY, situation: sit, shotType: sht } = determined;

        // Penalty is a hard-coded constant — mirrors the Flask route override.
        if (sit === 'Penalty') {
            return { xG: 0.76, x: 0.895, y: 0.5, chosenModel };
        }

        const features      = MODEL_FEATURES[chosenModel];
        const featureVector = buildFeatureVector(normX, normY, sit, sht, features);
        const session       = await getSession(chosenModel);

        // Input tensor: float32, shape [1, n_features].
        const tensor = new ort.Tensor('float32', featureVector, [1, features.length]);
        const feeds  = { float_input: tensor };
        const results = await session.run(feeds);

        // skl2onnx outputs 'probabilities' as float32 [N, 2] when zipmap=False.
        // Index 1 is P(class=1), i.e., the probability of a goal.
        const probsKey = 'probabilities' in results ? 'probabilities' : Object.keys(results).find(k => k !== 'label');
        const xG = parseFloat(results[probsKey].data[1].toFixed(2));

        return { xG, x: normX, y: normY, chosenModel };
    }

    // Expose only the public API.
    return { predict };
})();
