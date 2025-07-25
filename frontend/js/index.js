// --- DOM ELEMENT SELECTION ---
const canvas = document.getElementById('football-pitch');
const ctx = canvas.getContext('2d');
const pitchContainer = document.getElementById('pitch-container');
const xgDisplay = document.getElementById('xg-display');
const xgValue = document.getElementById('xg-value');
const situationSelect = document.getElementById('situation-select');
const shotTypeSelect = document.getElementById('shot-type-select');

// --- PITCH CONSTANTS (FIFA Standard) ---
const PITCH_LENGTH_METERS = 105;
const PITCH_WIDTH_METERS = 68;
const GOAL_WIDTH_METERS = 7.32;
const GOAL_DEPTH_METERS = 2;
const PADDING_METERS = 3;
const TOTAL_WIDTH_METERS = PITCH_LENGTH_METERS + 2 * PADDING_METERS;
const TOTAL_HEIGHT_METERS = PITCH_WIDTH_METERS + 2 * PADDING_METERS; // Includes top/bottom padding for visual balance
const ASPECT_RATIO = TOTAL_WIDTH_METERS / TOTAL_HEIGHT_METERS;

// --- PITCH MARKING CONSTANTS (in meters) ---
const GOAL_AREA_LENGTH = 5.5;
const GOAL_AREA_WIDTH = 18.32;
const PENALTY_AREA_LENGTH = 16.5;
const PENALTY_AREA_WIDTH = 40.32;
const CENTER_CIRCLE_RADIUS = 9.15;
const PENALTY_SPOT_DISTANCE = 11;

// --- STATE & CONFIGURATION ---
let scale; // Pixels per meter, calculated dynamically
let currentPoint = null; // Stores {x, y} in meters, for animation
let targetPoint = null; // Stores the target {x, y} in meters after a click
let animationStartTime = null;
const ANIMATION_DURATION_MS = 200; // milliseconds for animation for the dot
const POINT_RADIUS_METERS = 0.5; // Radius of the plotted point in meters
const LINE_COLOR = '#000080'; // Navy Blue
const POINT_COLOR = '#d32f2f'; // Red
const PENALTY_SPOT_RIGHT_X = PITCH_LENGTH_METERS - PENALTY_SPOT_DISTANCE;
const PENALTY_SPOT_Y = PITCH_WIDTH_METERS / 2;

// --- xG Animation State ---
let currentXG = 0; // Current displayed xG value
let targetXG = 0; // Target xG value from prediction
let xgAnimationStartTime = null;
const XG_ANIMATION_DURATION_MS = 300; // milliseconds for xG animation

/**
 * Converts meter-based coordinates to canvas pixel coordinates.
 * The X-axis is offset by padding. The Y-axis is inverted and offset.
 * @param {number} metersX - The X coordinate in meters.
 * @param {number} metersY - The Y coordinate in meters.
 * @returns {{x: number, y: number}} - The pixel coordinates.
 */
function metersToPixels(metersX, metersY) {
    const px = (metersX + PADDING_METERS) * scale;
    const py = (PITCH_WIDTH_METERS - metersY + PADDING_METERS) * scale; // Y is inverted and padded
    return { x: px, y: py };
}

function drawLine(x1m, y1m, x2m, y2m) {
    const p1 = metersToPixels(x1m, y1m);
    const p2 = metersToPixels(x2m, y2m);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
}

function drawCircle(x_m, y_m, radius_m, fill = false, startAngle = 0, endAngle = 2 * Math.PI, anticlockwise = false) {
    const p = metersToPixels(x_m, y_m);
    const radius_px = radius_m * scale;
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius_px, startAngle, endAngle, anticlockwise);
    if (fill) ctx.fill(); else ctx.stroke();
}

/**
 * Main function to draw the entire pitch and the plotted point.
 */
function drawPitch() {
    // 1. Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 2. Set drawing styles
    ctx.strokeStyle = LINE_COLOR;
    ctx.fillStyle = LINE_COLOR;
    ctx.lineWidth = 2;

    // --- Draw Goals ---
    const goalY1 = (PITCH_WIDTH_METERS - GOAL_WIDTH_METERS) / 2;
    const goalY2 = (PITCH_WIDTH_METERS + GOAL_WIDTH_METERS) / 2;
    drawLine(0, goalY1, -GOAL_DEPTH_METERS, goalY1);
    drawLine(0, goalY2, -GOAL_DEPTH_METERS, goalY2);
    drawLine(-GOAL_DEPTH_METERS, goalY1, -GOAL_DEPTH_METERS, goalY2);
    drawLine(PITCH_LENGTH_METERS, goalY1, PITCH_LENGTH_METERS + GOAL_DEPTH_METERS, goalY1);
    drawLine(PITCH_LENGTH_METERS, goalY2, PITCH_LENGTH_METERS + GOAL_DEPTH_METERS, goalY2);
    drawLine(PITCH_LENGTH_METERS + GOAL_DEPTH_METERS, goalY1, PITCH_LENGTH_METERS + GOAL_DEPTH_METERS, goalY2);

    // --- Draw Pitch Markings ---
    drawLine(0, 0, PITCH_LENGTH_METERS, 0); // Bottom touchline
    drawLine(0, PITCH_WIDTH_METERS, PITCH_LENGTH_METERS, PITCH_WIDTH_METERS); // Top touchline
    drawLine(0, 0, 0, PITCH_WIDTH_METERS); // Left goal line
    drawLine(PITCH_LENGTH_METERS, 0, PITCH_LENGTH_METERS, PITCH_WIDTH_METERS); // Right goal line
    drawLine(PITCH_LENGTH_METERS / 2, 0, PITCH_LENGTH_METERS / 2, PITCH_WIDTH_METERS); // Halfway line
    drawCircle(PITCH_LENGTH_METERS / 2, PITCH_WIDTH_METERS / 2, CENTER_CIRCLE_RADIUS); // Center circle
    drawCircle(PITCH_LENGTH_METERS / 2, PITCH_WIDTH_METERS / 2, 0.3, true); // Center spot

    // Left side penalty and goal areas
    drawLine(0, (PITCH_WIDTH_METERS - PENALTY_AREA_WIDTH) / 2, PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS - PENALTY_AREA_WIDTH) / 2);
    drawLine(0, (PITCH_WIDTH_METERS + PENALTY_AREA_WIDTH) / 2, PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS + PENALTY_AREA_WIDTH) / 2);
    drawLine(PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS - PENALTY_AREA_WIDTH) / 2, PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS + PENALTY_AREA_WIDTH) / 2);
    drawLine(0, (PITCH_WIDTH_METERS - GOAL_AREA_WIDTH) / 2, GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS - GOAL_AREA_WIDTH) / 2);
    drawLine(0, (PITCH_WIDTH_METERS + GOAL_AREA_WIDTH) / 2, GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS + GOAL_AREA_WIDTH) / 2);
    drawLine(GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS - GOAL_AREA_WIDTH) / 2, GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS + GOAL_AREA_WIDTH) / 2);
    drawCircle(PENALTY_SPOT_DISTANCE, PITCH_WIDTH_METERS / 2, 0.3, true);
    const leftArcAngle = Math.acos((PENALTY_AREA_LENGTH - PENALTY_SPOT_DISTANCE) / CENTER_CIRCLE_RADIUS);
    drawCircle(PENALTY_SPOT_DISTANCE, PITCH_WIDTH_METERS / 2, CENTER_CIRCLE_RADIUS, false, -leftArcAngle, leftArcAngle);

    // Right side penalty and goal areas
    const rightPenaltySpotX = PITCH_LENGTH_METERS - PENALTY_SPOT_DISTANCE;
    drawLine(PITCH_LENGTH_METERS, (PITCH_WIDTH_METERS - PENALTY_AREA_WIDTH) / 2, PITCH_LENGTH_METERS - PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS - PENALTY_AREA_WIDTH) / 2);
    drawLine(PITCH_LENGTH_METERS, (PITCH_WIDTH_METERS + PENALTY_AREA_WIDTH) / 2, PITCH_LENGTH_METERS - PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS + PENALTY_AREA_WIDTH) / 2);
    drawLine(PITCH_LENGTH_METERS - PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS - PENALTY_AREA_WIDTH) / 2, PITCH_LENGTH_METERS - PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS + PENALTY_AREA_WIDTH) / 2);
    drawLine(PITCH_LENGTH_METERS, (PITCH_WIDTH_METERS - GOAL_AREA_WIDTH) / 2, PITCH_LENGTH_METERS - GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS - GOAL_AREA_WIDTH) / 2);
    drawLine(PITCH_LENGTH_METERS, (PITCH_WIDTH_METERS + GOAL_AREA_WIDTH) / 2, PITCH_LENGTH_METERS - GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS + GOAL_AREA_WIDTH) / 2);
    drawLine(PITCH_LENGTH_METERS - GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS - GOAL_AREA_WIDTH) / 2, PITCH_LENGTH_METERS - GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS + GOAL_AREA_WIDTH) / 2);
    drawCircle(rightPenaltySpotX, PITCH_WIDTH_METERS / 2, 0.3, true);
    const rightArcAngle = Math.acos((rightPenaltySpotX - (PITCH_LENGTH_METERS - PENALTY_AREA_LENGTH)) / CENTER_CIRCLE_RADIUS);
    drawCircle(rightPenaltySpotX, PITCH_WIDTH_METERS / 2, CENTER_CIRCLE_RADIUS, false, Math.PI - rightArcAngle, Math.PI + rightArcAngle);

    // 3. Draw the plotted point if it exists
    drawDot();
}

/**
 * Sets up the canvas dimensions and scale based on its container size.
 */
function setupCanvas() {
    const containerWidth = pitchContainer.clientWidth;
    canvas.width = containerWidth;
    canvas.height = containerWidth / ASPECT_RATIO;
    scale = canvas.width / TOTAL_WIDTH_METERS; // Calculate scale based on total width including padding
    drawPitch();
}

/**
 * Handles click events on the canvas to plot a point.
 * @param {MouseEvent} event - The click event.
 */
function handleCanvasClick(event) {
    const rect = canvas.getBoundingClientRect();
    const pixelX = event.clientX - rect.left;
    const pixelY = event.clientY - rect.top;

    // Convert pixel coordinates to meter coordinates
    const metersX = (pixelX / scale) - PADDING_METERS;
    const metersY = PITCH_WIDTH_METERS + PADDING_METERS - (pixelY / scale);

    // If current situation is Penalty, check if click is away from the spot
    if (situationSelect.value === 'Penalty') {
        const distance = Math.sqrt(Math.pow(metersX - PENALTY_SPOT_RIGHT_X, 2) + Math.pow(metersY - PENALTY_SPOT_Y, 2));
        // If clicked more than a meter away from the penalty spot, reset situation
        if (distance > 1) {
            situationSelect.value = ''; // Reset to "All Situations"
            shotTypeSelect.disabled = false; // Re-enable shot type dropdown
        }
    }

    // Constrain the point to be within the field of play
    if (metersX >= 0 && metersX <= PITCH_LENGTH_METERS && metersY >= 0 && metersY <= PITCH_WIDTH_METERS) {
        targetPoint = { x: metersX, y: metersY };
        animationStartTime = performance.now();
        requestAnimationFrame(animateDot);
        callXGPrediction(metersX, metersY);
    }
}

/**
 * Calls the backend API to get an xG prediction for the given point.
 * @param {number} x - The X coordinate in meters.
 * @param {number} y - The Y coordinate in meters.
 */
async function callXGPrediction(x, y) {
    const situation = situationSelect.value || null;
    const shot_type = shotTypeSelect.value || null;

    const fetchBody = {
        x: x,
        y: y,
        situation: situation,
        shot_type: shot_type,
        normalisation: { is_normalised: false, max_pitch_width: 68, max_pitch_length: 105 }
    };

    try {
        const response = await fetch('https://redshaw-web-apps.onrender.com/redshaw-xg/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(fetchBody)
        });
        const data = await response.json();

        if (data.xG !== undefined) {
            targetXG = data.xG;
            xgAnimationStartTime = performance.now();
            requestAnimationFrame(animateXG);
        } else {
            throw new Error(data.error || 'Unexpected response format');
        }
    } catch (err) {
        xgValue.textContent = 'Error';
        console.error('Fetch error:', err);
        currentXG = 0; // Reset currentXG on error
        targetXG = 0; // Reset targetXG on error
    }
}

/**
 * Handles changes to the situation dropdown.
 * If "Penalty" is selected, it automatically plots the point.
 * Otherwise, it just updates the prediction for the existing point.
 */
function handleSituationChange() {
    if (situationSelect.value === 'Penalty') {
        // Automatically plot the point at the right penalty spot
        targetPoint = { x: PENALTY_SPOT_RIGHT_X, y: PENALTY_SPOT_Y };
        animationStartTime = performance.now();
        requestAnimationFrame(animateDot);
        callXGPrediction(targetPoint.x, targetPoint.y);
        // Disable shot type dropdown and reset its value
        shotTypeSelect.disabled = true;
        shotTypeSelect.value = ''; // "All Shot Types"
    } else {
        // For other situations, just update the xG if a point is already plotted
        updateXGPrediction();
        // Enable shot type dropdown
        shotTypeSelect.disabled = false;
    }
}

/**
 * Re-triggers the prediction if a filter is changed and a point exists.
 */
function updateXGPrediction() {
    if (currentPoint) {
        callXGPrediction(currentPoint.x, currentPoint.y);
    }
}

/**
 * Draws the animated dot on the canvas.
 */
function drawDot() {
    if (currentPoint) {
        ctx.fillStyle = POINT_COLOR;
        const p = metersToPixels(currentPoint.x, currentPoint.y);
        const radius_px = POINT_RADIUS_METERS * scale; // Scale the radius
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius_px, 0, 2 * Math.PI);
        ctx.fill();
    }
}

/**
 * Animates the dot's movement from its current position to the target position.
 * @param {DOMHighResTimeStamp} currentTime - The current time provided by requestAnimationFrame.
 */
function animateDot(currentTime) {
    if (!animationStartTime) animationStartTime = currentTime;
    const elapsedTime = currentTime - animationStartTime;
    const progress = Math.min(elapsedTime / ANIMATION_DURATION_MS, 1); // Clamp between 0 and 1

    if (currentPoint === null) {
        currentPoint = { ...targetPoint }; // Initialize if first time
    }

    // Linear interpolation for smooth movement
    currentPoint.x = currentPoint.x + (targetPoint.x - currentPoint.x) * progress;
    currentPoint.y = currentPoint.y + (targetPoint.y - currentPoint.y) * progress;

    drawPitch(); // Redraw the entire pitch with the updated dot position

    if (progress < 1) {
        requestAnimationFrame(animateDot); // Continue animation
    } else {
        animationStartTime = null; // Reset for next animation
        currentPoint = { ...targetPoint }; // Ensure it lands exactly on target
        drawPitch(); // Final draw to ensure correct position
    }
}

/**
 * Animates the xG value smoothly.
 * @param {DOMHighResTimeStamp} currentTime - The current time provided by requestAnimationFrame.
 */
function animateXG(currentTime) {
    if (!xgAnimationStartTime) xgAnimationStartTime = currentTime;
    const elapsedTime = currentTime - xgAnimationStartTime;
    const progress = Math.min(elapsedTime / XG_ANIMATION_DURATION_MS, 1);

    currentXG = currentXG + (targetXG - currentXG) * progress;
    xgValue.textContent = currentXG.toFixed(2);

    if (progress < 1) {
        requestAnimationFrame(animateXG);
    } else {
        xgAnimationStartTime = null;
        currentXG = targetXG; // Ensure it lands exactly on target
        xgValue.textContent = currentXG.toFixed(2);
    }
}

// --- EVENT LISTENERS & INITIALIZATION ---
window.addEventListener('resize', setupCanvas);
canvas.addEventListener('click', handleCanvasClick);
situationSelect.addEventListener('change', handleSituationChange);
shotTypeSelect.addEventListener('change', updateXGPrediction);

// Initial setup on page load
document.addEventListener('DOMContentLoaded', () => {
    setupCanvas();
    // Reset dropdowns and enable shot type on page load
    situationSelect.value = ''; // "All Situations"
    shotTypeSelect.value = ''; // "All Shot Types"
    shotTypeSelect.disabled = false;
    currentXG = 0; // Initialize currentXG
    targetXG = 0; // Initialize targetXG
    xgValue.textContent = 'X.XX'; // Initial display
});