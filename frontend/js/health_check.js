/**
 * health_check.js
 *
 * With the migration to fully client-side ONNX inference, there is no longer
 * an external backend to poll. The loading overlay is hidden immediately and
 * the 'backendHealthy' event is dispatched so that all page listeners
 * initialise without delay.
 */
document.addEventListener('DOMContentLoaded', () => {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }
    window.dispatchEvent(new Event('backendHealthy'));
});