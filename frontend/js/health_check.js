const BACKEND_HEALTH_ENDPOINT = 'https://redshaw-web-apps.onrender.com/health';
const HEALTH_CHECK_TIMEOUT = 5000; // 5 seconds for a single health check attempt
const INITIAL_OVERLAY_DELAY = 2000; // 2 seconds before showing the overlay
const RETRY_INTERVAL = 10000; // Retry every 10 seconds if not healthy

let healthCheckIntervalId = null;
let overlayTimeoutId = null;
let countdownIntervalId = null; // For the 60-second countdown
let countdownValue = 60;
let countdownCompleted = false; // Flag to prevent countdown from restarting

async function checkBackendHealth() {
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT);

        const response = await fetch(BACKEND_HEALTH_ENDPOINT, { signal: controller.signal });
        clearTimeout(timeout);

        if (response.ok) {
            const data = await response.json();
            return data.status === 'healthy';
        }
        return false;
    } catch (error) {
        if (error.name === 'AbortError') {
            console.warn('Backend health check timed out.');
        } else {
            console.error('Error checking backend health:', error);
        }
        return false;
    }
}

function showLoadingOverlay(message) {
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = loadingOverlay.querySelector('.loading-message');
    loadingMessage.textContent = message;
    loadingOverlay.classList.remove('hidden');

    // Start the 60-second countdown only if it hasn't completed yet
    if (!countdownIntervalId && !countdownCompleted) {
        countdownValue = 60;
        loadingMessage.textContent = `Waking up the backend server... ${countdownValue}s`;
        countdownIntervalId = setInterval(() => {
            countdownValue--;
            if (countdownValue > 0) {
                loadingMessage.textContent = `Waking up the backend server... ${countdownValue}s`;
            } else {
                loadingMessage.textContent = 'The server will be ready very soon...';
                clearInterval(countdownIntervalId);
                countdownIntervalId = null;
                countdownCompleted = true; // Mark countdown as completed
            }
        }, 1000);
    }
}

function hideLoadingOverlay() {
    const loadingOverlay = document.getElementById('loading-overlay');
    loadingOverlay.classList.add('hidden');
    if (countdownIntervalId) {
        clearInterval(countdownIntervalId);
        countdownIntervalId = null;
    }
    countdownCompleted = false; // Reset for the next time it's needed
}

async function monitorBackendHealth() {
    const isHealthy = await checkBackendHealth();

    if (isHealthy) {
        console.log('Backend is healthy!');
        if (healthCheckIntervalId) {
            clearInterval(healthCheckIntervalId);
            healthCheckIntervalId = null;
        }
        if (overlayTimeoutId) {
            clearTimeout(overlayTimeoutId);
            overlayTimeoutId = null;
        }
        hideLoadingOverlay();
        window.dispatchEvent(new Event('backendHealthy'));
    } else {
        console.warn('Backend is not healthy. Retrying...');
        // Only show overlay if it hasn't been shown yet or if it was hidden
        if (!document.getElementById('loading-overlay').classList.contains('hidden')) {
            showLoadingOverlay('Waking up the backend server...');
        }
        if (!healthCheckIntervalId) {
            healthCheckIntervalId = setInterval(monitorBackendHealth, RETRY_INTERVAL);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Initially hide the overlay
    document.getElementById('loading-overlay').classList.add('hidden');

    // Set a timeout to show the overlay if the health check takes too long
    overlayTimeoutId = setTimeout(() => {
        showLoadingOverlay('Waking up the backend server...');
    }, INITIAL_OVERLAY_DELAY);

    // Start monitoring health immediately
    monitorBackendHealth();
});