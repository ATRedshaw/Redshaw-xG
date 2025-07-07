// =================================================================================
// SCRIPT START: HEATMAP VISUALIZATION LOGIC
// =================================================================================

// --- DOM Element Selection ---
const canvas = document.getElementById('football-pitch');
const ctx = canvas.getContext('2d');
const pitchContainer = document.getElementById('pitch-container');
const statusDiv = document.getElementById('status');
const loadButton = document.getElementById('load-heatmap');
const situationSelect = document.getElementById('situation-select');
const shotTypeSelect = document.getElementById('shot-type-select');

// --- Pitch Constants (FIFA Standard) ---
// These constants define the dimensions of the pitch in meters.
// They are crucial for ensuring the proportions of the drawing are correct.
const PITCH_LENGTH_METERS = 105;
const PITCH_WIDTH_METERS = 68;
const GOAL_WIDTH_METERS = 7.32;
const GOAL_DEPTH_METERS = 2;
const PADDING_METERS = 3; // Padding around the pitch.
const TOTAL_WIDTH_METERS = PITCH_LENGTH_METERS + 2 * PADDING_METERS;
const TOTAL_HEIGHT_METERS = PITCH_WIDTH_METERS + 2 * PADDING_METERS;
const ASPECT_RATIO = TOTAL_WIDTH_METERS / TOTAL_HEIGHT_METERS;

// --- Pitch Markings Constants (in meters) ---
const GOAL_AREA_LENGTH = 5.5;
const GOAL_AREA_WIDTH = 18.32;
const PENALTY_AREA_LENGTH = 16.5;
const PENALTY_AREA_WIDTH = 40.32;
const CENTER_CIRCLE_RADIUS = 9.15;
const PENALTY_SPOT_DISTANCE = 11;

// --- State and Configuration ---
let scale; // Calculated dynamically based on container size
let heatmapData = null; // Holds the fetched heatmap data
const LINE_COLOR = '#000080'; // Navy blue for pitch lines

// --- Color Mapping for Heatmap ---
// Defines the color gradient for xG values from 0.0 to 1.0.
// Returns an RGBA object for efficient drawing with ImageData.
const colorStops = [
    { value: 0.0,  color: { r: 255, g: 255, b: 255, a: 0 } },   // Transparent
    { value: 0.01, color: { r: 220, g: 230, b: 255, a: 50 } },  // Light Blue
    { value: 0.1,  color: { r: 100, g: 149, b: 237, a: 100 } }, // Cornflower Blue
    { value: 0.3,  color: { r: 0,   g: 128, b: 0,   a: 150 } }, // Green
    { value: 0.5,  color: { r: 255, g: 255, b: 0,   a: 200 } }, // Yellow
    { value: 0.7,  color: { r: 255, g: 140, b: 0,   a: 220 } }, // Dark Orange
    { value: 1.0,  color: { r: 255, g: 0,   b: 0,   a: 255 } }    // Red
];

/**
 * Calculates the interpolated RGBA color for a given xG value.
 * @param {number} xgValue - The xG value (0.0 to 1.0).
 * @returns {{r: number, g: number, b: number, a: number}} The RGBA color object.
 */
function getHeatmapColorRGBA(xgValue) {
    const intensity = Math.min(Math.max(xgValue, 0), 1);

    if (intensity === 0) return colorStops[0].color;

    // Find the two color stops to interpolate between
    let lowerStop = colorStops[0];
    let upperStop = colorStops[colorStops.length - 1];
    for (let i = 0; i < colorStops.length - 1; i++) {
        if (intensity >= colorStops[i].value && intensity <= colorStops[i + 1].value) {
            lowerStop = colorStops[i];
            upperStop = colorStops[i + 1];
            break;
        }
    }

    // Calculate interpolation factor (t)
    const range = upperStop.value - lowerStop.value;
    const t = (range > 0) ? (intensity - lowerStop.value) / range : 0;

    // Interpolate R, G, B, A components
    const r = Math.round(lowerStop.color.r + t * (upperStop.color.r - lowerStop.color.r));
    const g = Math.round(lowerStop.color.g + t * (upperStop.color.g - lowerStop.color.g));
    const b = Math.round(lowerStop.color.b + t * (upperStop.color.b - lowerStop.color.b));
    const a = Math.round(lowerStop.color.a + t * (upperStop.color.a - lowerStop.color.a));

    return { r, g, b, a };
}

// --- Canvas Drawing Functions ---

/**
 * Converts coordinates from meters to canvas pixels.
 * @param {number} metersX - X coordinate in meters.
 * @param {number} metersY - Y coordinate in meters.
 * @returns {{x: number, y: number}} Pixel coordinates.
 */
function metersToPixels(metersX, metersY) {
    const px = (metersX + PADDING_METERS) * scale;
    const py = (PITCH_WIDTH_METERS - metersY + PADDING_METERS) * scale;
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
 * Draws all the lines and markings of the football pitch.
 */
function drawPitchMarkings() {
    ctx.strokeStyle = LINE_COLOR;
    ctx.fillStyle = LINE_COLOR;
    ctx.lineWidth = 2;

    // Goals
    const goalY1 = (PITCH_WIDTH_METERS - GOAL_WIDTH_METERS) / 2;
    const goalY2 = (PITCH_WIDTH_METERS + GOAL_WIDTH_METERS) / 2;
    drawLine(0, goalY1, -GOAL_DEPTH_METERS, goalY1);
    drawLine(0, goalY2, -GOAL_DEPTH_METERS, goalY2);
    drawLine(-GOAL_DEPTH_METERS, goalY1, -GOAL_DEPTH_METERS, goalY2);
    drawLine(PITCH_LENGTH_METERS, goalY1, PITCH_LENGTH_METERS + GOAL_DEPTH_METERS, goalY1);
    drawLine(PITCH_LENGTH_METERS, goalY2, PITCH_LENGTH_METERS + GOAL_DEPTH_METERS, goalY2);
    drawLine(PITCH_LENGTH_METERS + GOAL_DEPTH_METERS, goalY1, PITCH_LENGTH_METERS + GOAL_DEPTH_METERS, goalY2);

    // Pitch boundary and halfway line
    drawLine(0, 0, PITCH_LENGTH_METERS, 0);
    drawLine(0, PITCH_WIDTH_METERS, PITCH_LENGTH_METERS, PITCH_WIDTH_METERS);
    drawLine(0, 0, 0, PITCH_WIDTH_METERS);
    drawLine(PITCH_LENGTH_METERS, 0, PITCH_LENGTH_METERS, PITCH_WIDTH_METERS);
    drawLine(PITCH_LENGTH_METERS / 2, 0, PITCH_LENGTH_METERS / 2, PITCH_WIDTH_METERS);

    // Center circle and spot
    drawCircle(PITCH_LENGTH_METERS / 2, PITCH_WIDTH_METERS / 2, CENTER_CIRCLE_RADIUS);
    drawCircle(PITCH_LENGTH_METERS / 2, PITCH_WIDTH_METERS / 2, 0.3, true);

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
}

/**
 * Draws the heatmap onto the canvas using ImageData for performance and smoothness.
 * Uses bilinear interpolation for a less "blocky" appearance.
 */
function drawHeatmap() {
    if (!heatmapData) return;

    const { x_coords, y_coords } = heatmapData.grid_definition;
    const heatmap = heatmapData.heatmap;
    const cellWidth = x_coords.length > 1 ? x_coords[1] - x_coords[0] : 0.01;
    const cellHeight = y_coords.length > 1 ? y_coords[1] - y_coords[0] : 0.01;

    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;

    for (let py = 0; py < canvas.height; py++) {
        for (let px = 0; px < canvas.width; px++) {
            const meterX = (px / scale) - PADDING_METERS;
            const meterY = PITCH_WIDTH_METERS + PADDING_METERS - (py / scale);

            if (meterX >= 0 && meterX <= PITCH_LENGTH_METERS && meterY >= 0 && meterY <= PITCH_WIDTH_METERS) {
                const xIndex = Math.floor(meterX / cellWidth);
                const yIndex = Math.floor(meterY / cellHeight);

                let xgValue = 0;
                if (xIndex < x_coords.length - 1 && yIndex < y_coords.length - 1 && y_coords.length > 1) {
                    const x1 = x_coords[xIndex], x2 = x_coords[xIndex + 1];
                    const y1 = y_coords[yIndex], y2 = y_coords[yIndex + 1];
                    const xWeight = (meterX - x1) / (x2 - x1);
                    const yWeight = (meterY - y1) / (y2 - y1);

                    const v00 = heatmap[yIndex]?.[xIndex] || 0;
                    const v10 = heatmap[yIndex]?.[xIndex + 1] || 0;
                    const v01 = heatmap[yIndex + 1]?.[xIndex] || 0;
                    const v11 = heatmap[yIndex + 1]?.[xIndex + 1] || 0;

                    const top = v00 * (1 - xWeight) + v10 * xWeight;
                    const bottom = v01 * (1 - xWeight) + v11 * xWeight;
                    xgValue = top * (1 - yWeight) + bottom * yWeight;
                } else {
                    xgValue = heatmap[yIndex]?.[xIndex] || 0; // Fallback to nearest neighbor
                }

                if (xgValue > 0) {
                    const color = getHeatmapColorRGBA(xgValue);
                    const pixelIndex = (py * canvas.width + px) * 4;
                    data[pixelIndex] = color.r;
                    data[pixelIndex + 1] = color.g;
                    data[pixelIndex + 2] = color.b;
                    data[pixelIndex + 3] = color.a;
                }
            }
        }
    }
    ctx.putImageData(imageData, 0, 0);
}

/**
 * Main drawing function. Clears the canvas and redraws everything.
 */
function drawComplete() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    // Draw heatmap first, so it's underneath the pitch lines
    drawHeatmap();
    
    // Draw pitch markings on top with slight transparency
    ctx.globalAlpha = 0.8;
    drawPitchMarkings();
    ctx.globalAlpha = 1.0; // Reset for other operations
}

/**
 * Sets up the canvas dimensions based on its container size and redraws.
 */
function setupCanvas() {
    const containerWidth = pitchContainer.clientWidth;
    canvas.width = containerWidth;
    canvas.height = containerWidth / ASPECT_RATIO;
    scale = canvas.width / TOTAL_WIDTH_METERS;
    drawComplete();
}

// --- UI and Data Handling ---

/**
 * Displays a status message to the user.
 * @param {string} message - The message to display.
 * @param {'loading' | 'error' | 'success'} type - The type of message.
 */
function showStatus(message, type) {
    statusDiv.textContent = message;
    // Reset classes
    statusDiv.classList.remove(
        'bg-yellow-100', 'text-yellow-800', 'border-yellow-300',
        'bg-red-100', 'text-red-800', 'border-red-300',
        'bg-green-100', 'text-green-800', 'border-green-300'
    );
    
    // Apply new classes based on type
    if (type === 'loading') {
        statusDiv.classList.add('bg-yellow-100', 'text-yellow-800', 'border-yellow-300');
    } else if (type === 'error') {
        statusDiv.classList.add('bg-red-100', 'text-red-800', 'border-red-300');
    } else if (type === 'success') {
        statusDiv.classList.add('bg-green-100', 'text-green-800', 'border-green-300');
    }
    statusDiv.style.display = 'block';
}

function hideStatus() {
    statusDiv.style.display = 'none';
}

/**
 * Fetches heatmap data from the backend API based on selected filters.
 */
async function loadHeatmapData() {
    const situation = situationSelect.value || null;
    const shotType = shotTypeSelect.value || null;
    
    loadButton.disabled = true;

    try {
        const params = new URLSearchParams();
        if (situation) params.append('situation', situation);
        if (shotType) params.append('shot_type', shotType);
        params.append('max_length', PITCH_LENGTH_METERS.toString());
        params.append('max_width', PITCH_WIDTH_METERS.toString());

        // IMPORTANT: Replace with your actual backend API endpoint
        const response = await fetch(`https://redshaw-web-apps.onrender.com/api/redshaw-xg/predict/grid?${params}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        heatmapData = await response.json();
        
        drawComplete();
        
    } catch (error) {
        console.error('Error loading heatmap:', error);
        heatmapData = null; // Clear old data on error
        drawComplete(); // Redraw pitch without heatmap
        showStatus('Failed to load data. Please check the backend connection.', 'error');
    } finally {
        loadButton.disabled = false;
    }
}

// --- Event Listeners and Initialization ---

// Redraw canvas on window resize to maintain proportions
window.addEventListener('resize', setupCanvas);

// Load data when the button is clicked
loadButton.addEventListener('click', loadHeatmapData);

// Initial setup when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    setupCanvas();
    loadHeatmapData();
});