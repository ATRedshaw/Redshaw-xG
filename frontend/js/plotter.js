document.addEventListener('DOMContentLoaded', () => {
    // --- DOM ELEMENT SELECTION ---
    const canvas = document.getElementById('football-pitch');
    const ctx = canvas.getContext('2d');
    const pitchContainer = document.getElementById('pitch-container');
    const loadMatchSelect = document.getElementById('load-match');
    const matchDateInput = document.getElementById('match-date');
    const homeTeamNameInput = document.getElementById('home-team-name');
    const awayTeamNameInput = document.getElementById('away-team-name');
    const homeTeamColorInput = document.getElementById('home-team-color');
    const awayTeamColorInput = document.getElementById('away-team-color');
    const saveMatchDetailsButton = document.getElementById('save-match-details');
    const deleteMatchButton = document.getElementById('delete-match-button');
    const matchDetailsModal = document.getElementById('match-details-modal');
    const matchDetailsModalButton = document.getElementById('match-details-modal-button');
    const matchTitleDisplay = document.getElementById('match-title');
    const plotForHomeRadio = document.getElementById('plot-for-home');
    const plotForAwayRadio = document.getElementById('plot-for-away');
    const homeTeamLabel = document.getElementById('plot-for-home-label');
    const awayTeamLabel = document.getElementById('plot-for-away-label');
    const situationSelect = document.getElementById('situation-select');
    const shotTypeSelect = document.getElementById('shot-type-select');
    const homeTeamXgLabel = document.getElementById('home-team-xg-label');
    const awayTeamXgLabel = document.getElementById('away-team-xg-label');
    const homeTeamXgDisplay = document.getElementById('home-team-xg');
    const awayTeamXgDisplay = document.getElementById('away-team-xg');
    const selectedShotXgDisplay = document.getElementById('selected-shot-xg');
    const homeShotList = document.getElementById('home-shot-list');
    const awayShotList = document.getElementById('away-shot-list');
    const homeShotListHeader = document.getElementById('home-shot-list-header');
    const awayShotListHeader = document.getElementById('away-shot-list-header');
    const editShotModal = document.getElementById('edit-shot-modal');
    const editShotButton = document.getElementById('edit-shot-button');
    const deleteShotButton = document.getElementById('delete-shot-button');
    const saveShotChangesButton = document.getElementById('save-shot-changes-button');
    const editSituationSelect = document.getElementById('edit-situation-select');
    const editShotTypeSelect = document.getElementById('edit-shot-type-select');


    // --- PITCH CONSTANTS (FIFA Standard) ---
    const PITCH_LENGTH_METERS = 105;
    const PITCH_WIDTH_METERS = 68;
    const GOAL_WIDTH_METERS = 7.32;
    const GOAL_DEPTH_METERS = 2;
    const PADDING_METERS = 3;
    const TOTAL_WIDTH_METERS = PITCH_LENGTH_METERS + 2 * PADDING_METERS;
    const TOTAL_HEIGHT_METERS = PITCH_WIDTH_METERS + 2 * PADDING_METERS;
    const ASPECT_RATIO = TOTAL_WIDTH_METERS / TOTAL_HEIGHT_METERS;

    // --- PITCH MARKING CONSTANTS (in meters) ---
    const GOAL_AREA_LENGTH = 5.5;
    const GOAL_AREA_WIDTH = 18.32;
    const PENALTY_AREA_LENGTH = 16.5;
    const PENALTY_AREA_WIDTH = 40.32;
    const CENTER_CIRCLE_RADIUS = 9.15;
    const PENALTY_SPOT_DISTANCE = 11;

    // --- STATE & CONFIGURATION ---
    let scale;
    let db;
    let currentMatch = null;
    let selectedShot = null;
    const dbName = 'xgPlotterDB';
    const dbVersion = 1;
    const POINT_RADIUS_METERS = 0.5;
    const LINE_COLOR = '#000080';

    // --- DATABASE ---
    function initDB() {
        const request = indexedDB.open(dbName, dbVersion);

        request.onerror = (event) => console.error('Database error:', event.target.errorCode);

        request.onsuccess = (event) => {
            db = event.target.result;
            console.log('Database initialised');
            loadMatchesIntoSelect();
            setupEventListeners();
            setupCanvas();
        };

        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains('matches')) {
                const objectStore = db.createObjectStore('matches', { keyPath: 'id', autoIncrement: true });
                objectStore.createIndex('name', 'name', { unique: false });
            }
        };
    }

    // --- PITCH DRAWING ---
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

    function drawPitch() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
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

        // Markings
        drawLine(0, 0, PITCH_LENGTH_METERS, 0);
        drawLine(0, PITCH_WIDTH_METERS, PITCH_LENGTH_METERS, PITCH_WIDTH_METERS);
        drawLine(0, 0, 0, PITCH_WIDTH_METERS);
        drawLine(PITCH_LENGTH_METERS, 0, PITCH_LENGTH_METERS, PITCH_WIDTH_METERS);
        drawLine(PITCH_LENGTH_METERS / 2, 0, PITCH_LENGTH_METERS / 2, PITCH_WIDTH_METERS);
        drawCircle(PITCH_LENGTH_METERS / 2, PITCH_WIDTH_METERS / 2, CENTER_CIRCLE_RADIUS);
        drawCircle(PITCH_LENGTH_METERS / 2, PITCH_WIDTH_METERS / 2, 0.3, true);

        // Left areas
        drawLine(0, (PITCH_WIDTH_METERS - PENALTY_AREA_WIDTH) / 2, PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS - PENALTY_AREA_WIDTH) / 2);
        drawLine(0, (PITCH_WIDTH_METERS + PENALTY_AREA_WIDTH) / 2, PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS + PENALTY_AREA_WIDTH) / 2);
        drawLine(PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS - PENALTY_AREA_WIDTH) / 2, PENALTY_AREA_LENGTH, (PITCH_WIDTH_METERS + PENALTY_AREA_WIDTH) / 2);
        drawLine(0, (PITCH_WIDTH_METERS - GOAL_AREA_WIDTH) / 2, GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS - GOAL_AREA_WIDTH) / 2);
        drawLine(0, (PITCH_WIDTH_METERS + GOAL_AREA_WIDTH) / 2, GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS + GOAL_AREA_WIDTH) / 2);
        drawLine(GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS - GOAL_AREA_WIDTH) / 2, GOAL_AREA_LENGTH, (PITCH_WIDTH_METERS + GOAL_AREA_WIDTH) / 2);
        drawCircle(PENALTY_SPOT_DISTANCE, PITCH_WIDTH_METERS / 2, 0.3, true);
        const leftArcAngle = Math.acos((PENALTY_AREA_LENGTH - PENALTY_SPOT_DISTANCE) / CENTER_CIRCLE_RADIUS);
        drawCircle(PENALTY_SPOT_DISTANCE, PITCH_WIDTH_METERS / 2, CENTER_CIRCLE_RADIUS, false, -leftArcAngle, leftArcAngle);

        // Right areas
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

        drawShots();
    }

    function setupCanvas() {
        const containerWidth = pitchContainer.clientWidth;
        canvas.width = containerWidth;
        canvas.height = containerWidth / ASPECT_RATIO;
        scale = canvas.width / TOTAL_WIDTH_METERS;
        drawPitch();
    }

    function drawShots() {
        if (!currentMatch || !currentMatch.shots) return;

        currentMatch.shots.forEach((shot, index) => {
            ctx.fillStyle = shot.team === 'home' ? currentMatch.homeColor : currentMatch.awayColor;
            const p = metersToPixels(shot.x, shot.y);
            const radius_px = POINT_RADIUS_METERS * scale;
            ctx.beginPath();
            ctx.arc(p.x, p.y, radius_px, 0, 2 * Math.PI);
            ctx.fill();

            if (selectedShot && selectedShot.index === index) {
                ctx.strokeStyle = '#000000';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        });
    }

    // --- EVENT LISTENERS & LOGIC ---
    function setupEventListeners() {
        window.addEventListener('resize', setupCanvas);
        canvas.addEventListener('click', handleCanvasClick);
        saveMatchDetailsButton.addEventListener('click', saveMatchDetails);
        deleteMatchButton.addEventListener('click', deleteCurrentMatch);
        loadMatchSelect.addEventListener('change', handleLoadMatch);
        homeTeamNameInput.addEventListener('input', () => updateTeamLabels());
        awayTeamNameInput.addEventListener('input', () => updateTeamLabels());
        matchDateInput.addEventListener('input', () => updateTeamLabels());
        matchDetailsModalButton.addEventListener('click', openModal);

        // Close modal if clicking outside of it
        window.addEventListener('click', (event) => {
            if (event.target === matchDetailsModal) {
                closeModal();
            }
            if (event.target === editShotModal) {
                closeEditShotModal();
            }
        });

        editShotButton.addEventListener('click', openEditShotModal);
        deleteShotButton.addEventListener('click', deleteSelectedShot);
        saveShotChangesButton.addEventListener('click', saveShotChanges);
    }

    function openModal() {
        matchDetailsModal.classList.remove('hidden');
        updateDeleteMatchButton();
    }

    function closeModal() {
        matchDetailsModal.classList.add('hidden');
    }

    function openEditShotModal() {
        if (!selectedShot) return;
        editSituationSelect.value = selectedShot.situation;
        editShotTypeSelect.value = selectedShot.shotType;
        editShotModal.classList.remove('hidden');
    }

    function closeEditShotModal() {
        editShotModal.classList.add('hidden');
    }

    function updateTeamLabels() {
        const homeName = homeTeamNameInput.value || 'Home Team';
        const awayName = awayTeamNameInput.value || 'Away Team';
        const matchDate = matchDateInput.value;

        homeTeamLabel.textContent = homeName;
        awayTeamLabel.textContent = awayName;
        homeTeamXgLabel.textContent = `${homeName} xG`;
        awayTeamXgLabel.textContent = `${awayName} xG`;
        homeShotListHeader.textContent = `${homeName} Shots`;
        awayShotListHeader.textContent = `${awayName} Shots`;

        if (homeTeamNameInput.value && awayTeamNameInput.value && matchDate) {
            matchTitleDisplay.textContent = `${homeTeamNameInput.value} vs ${awayTeamNameInput.value} - ${matchDate}`;
        } else {
            matchTitleDisplay.textContent = 'xG match plotter';
        }
    }

    async function handleCanvasClick(event) {
        if (!currentMatch) {
            alert('Please save match details before plotting shots.');
            return;
        }

        const rect = canvas.getBoundingClientRect();
        const pixelX = event.clientX - rect.left;
        const pixelY = event.clientY - rect.top;

        let metersX = (pixelX / scale) - PADDING_METERS;
        let metersY = PITCH_WIDTH_METERS + PADDING_METERS - (pixelY / scale);

        if (metersX < 0 || metersX > PITCH_LENGTH_METERS || metersY < 0 || metersY > PITCH_WIDTH_METERS) {
            return; // Click outside the pitch
        }

        if (selectedShot) {
            // Reposition existing shot
            const attackingLeft = (selectedShot.team === 'home' && currentMatch.homeAttacking === 'left') || (selectedShot.team === 'away' && currentMatch.awayAttacking === 'left');
            let apiX = metersX;
            if (attackingLeft) {
                apiX = PITCH_LENGTH_METERS - metersX;
            }
            const xg = await callXGPrediction(apiX, metersY, selectedShot.situation, selectedShot.shotType);
            if (xg === null) return;

            currentMatch.shots[selectedShot.index].x = metersX;
            currentMatch.shots[selectedShot.index].y = metersY;
            currentMatch.shots[selectedShot.index].xg = xg;
            
            selectedShot.x = metersX;
            selectedShot.y = metersY;
            selectedShot.xg = xg;

            updateMatchInDB();
            updateXgDisplay();
            updateShotLists();
            drawPitch();

        } else {
            // Add new shot
            const plottingFor = plotForHomeRadio.checked ? 'home' : 'away';
            const attackingLeft = (plottingFor === 'home' && currentMatch.homeAttacking === 'left') || (plottingFor === 'away' && currentMatch.awayAttacking === 'left');

            let apiX = metersX;
            if (attackingLeft) {
                apiX = PITCH_LENGTH_METERS - metersX;
            }

            const xg = await callXGPrediction(apiX, metersY, situationSelect.value, shotTypeSelect.value);

            if (xg === null) return;

            const newShot = {
                x: metersX,
                y: metersY,
                team: plottingFor,
                situation: situationSelect.value,
                shotType: shotTypeSelect.value,
                xg: xg
            };

            newShot.index = currentMatch.shots.length;
            currentMatch.shots.push(newShot);
            updateMatchInDB();
            drawPitch();
            updateShotLists();
            updateXgDisplay();
            updateMatchTitleDisplay();
        }
    }

    function saveMatchDetails() {
        const homeTeamName = homeTeamNameInput.value;
        const awayTeamName = awayTeamNameInput.value;
        const matchDate = matchDateInput.value;
        const homeTeamColor = homeTeamColorInput.value;
        const awayTeamColor = awayTeamColorInput.value;

        if (!homeTeamName || !awayTeamName || !matchDate) {
            alert('Please fill in all match details.');
            return;
        }

        const matchTitle = `${homeTeamName} vs ${awayTeamName} - ${matchDate}`;

        if (currentMatch) {
            // Update existing match
            currentMatch.name = matchTitle;
            currentMatch.date = matchDate;
            currentMatch.homeTeam = homeTeamName;
            currentMatch.awayTeam = awayTeamName;
            currentMatch.homeColor = homeTeamColor;
            currentMatch.awayColor = awayTeamColor;
            updateMatchInDB();
            updateTeamLabels();
            closeModal();
        } else {
            // Create new match
            currentMatch = {
                name: matchTitle,
                date: matchDate,
                homeTeam: homeTeamName,
                awayTeam: awayTeamName,
                homeColor: homeTeamColor,
                awayColor: awayTeamColor,
                homeAttacking: 'right', // Default
                awayAttacking: 'left', // Default
                shots: []
            };
            addMatchToDB(currentMatch);
        }
    }

    function addMatchToDB(match) {
        const transaction = db.transaction(['matches'], 'readwrite');
        const objectStore = transaction.objectStore('matches');
        const request = objectStore.add(match);

        request.onsuccess = (event) => {
            currentMatch.id = event.target.result;
            alert('Match saved!');
            loadMatchesIntoSelect();
            updateTeamLabels();
            closeModal();
        };
        request.onerror = (event) => console.error('Error saving match:', event.target.errorCode);
    }

    function updateMatchInDB() {
        if (!currentMatch || !currentMatch.id) return;
        const transaction = db.transaction(['matches'], 'readwrite');
        const objectStore = transaction.objectStore('matches');
        const request = objectStore.put(currentMatch);

        request.onsuccess = () => {
            console.log('Match updated successfully.');
            updateXgDisplay();
            updateShotLists();
        }
        request.onerror = (event) => console.error('Error updating match:', event.target.errorCode);
    }

    function loadMatchesIntoSelect() {
        const transaction = db.transaction(['matches'], 'readonly');
        const objectStore = transaction.objectStore('matches');
        const request = objectStore.getAll();

        request.onsuccess = () => {
            const matches = request.result;
            loadMatchSelect.innerHTML = '<option value="">New Match</option>';
            matches.forEach(match => {
                const option = document.createElement('option');
                option.value = match.id;
                option.textContent = match.name;
                if (currentMatch && currentMatch.id === match.id) {
                    option.selected = true;
                }
                loadMatchSelect.appendChild(option);
            });
        };
    }

    function handleLoadMatch() {
        const matchId = parseInt(loadMatchSelect.value, 10);
        if (!matchId) {
            resetToNewMatch();
            return;
        }

        const transaction = db.transaction(['matches'], 'readonly');
        const objectStore = transaction.objectStore('matches');
        const request = objectStore.get(matchId);

        request.onsuccess = () => {
            currentMatch = request.result;
            if (currentMatch) {
                matchDateInput.value = currentMatch.date;
                homeTeamNameInput.value = currentMatch.homeTeam;
                awayTeamNameInput.value = currentMatch.awayTeam;
                homeTeamColorInput.value = currentMatch.homeColor;
                awayTeamColorInput.value = currentMatch.awayColor;
                updateTeamLabels();
                drawPitch();
                updateXgDisplay();
                updateShotLists();
                updateTeamLabels();
            }
        };
    }

    function resetToNewMatch() {
        currentMatch = null;
        selectedShot = null;
        matchDateInput.value = '';
        homeTeamNameInput.value = '';
        awayTeamNameInput.value = '';
        homeTeamColorInput.value = '#ff0000';
        awayTeamColorInput.value = '#0000ff';
        situationSelect.value = ''; // Set to "All Situations"
        shotTypeSelect.value = ''; // Set to "All Shot Types"
        updateTeamLabels();
        drawPitch();
        updateXgDisplay();
        updateShotLists();
        updateEditDeleteButtons();
        updateTeamLabels();
    }

    function updateXgDisplay() {
        let homeXg = 0;
        let awayXg = 0;
        if (currentMatch && currentMatch.shots) {
            currentMatch.shots.forEach(shot => {
                if (shot.team === 'home') {
                    homeXg += shot.xg;
                } else {
                    awayXg += shot.xg;
                }
            });
        }
        homeTeamXgDisplay.textContent = homeXg.toFixed(2);
        awayTeamXgDisplay.textContent = awayXg.toFixed(2);

        if (selectedShot) {
            selectedShotXgDisplay.textContent = selectedShot.xg.toFixed(2);
        } else {
            selectedShotXgDisplay.textContent = '-';
        }
    }

    function updateShotLists() {
        homeShotList.innerHTML = '';
        awayShotList.innerHTML = '';

        if (!currentMatch || !currentMatch.shots) return;

        currentMatch.shots.forEach((shot, index) => {
            const shotItem = document.createElement('li');
            shotItem.className = 'p-2 mb-1 rounded cursor-pointer hover:bg-slate-200';
            const displayShotType = shot.shotType || 'N/A';
            const displaySituation = shot.situation || 'N/A';
            shotItem.textContent = `Shot ${index + 1}: ${shot.xg.toFixed(2)} xG (${displayShotType}, ${displaySituation})`;
            shotItem.dataset.shotIndex = index;

            if (selectedShot && selectedShot.index === index) {
                shotItem.classList.add('bg-blue-200');
            }

            shotItem.addEventListener('click', () => {
                handleShotSelection(index);
            });

            if (shot.team === 'home') {
                homeShotList.appendChild(shotItem);
            } else {
                awayShotList.appendChild(shotItem);
            }
        });
    }

    function handleShotSelection(index) {
        if (selectedShot && selectedShot.index === index) {
            // Deselect if clicking the same shot again
            selectedShot = null;
        } else {
            selectedShot = { ...currentMatch.shots[index], index: index };
        }
        updateXgDisplay();
        updateShotLists();
        drawPitch();
        updateEditDeleteButtons();
    }

    function updateEditDeleteButtons() {
        if (selectedShot) {
            editShotButton.disabled = false;
            deleteShotButton.disabled = false;
        } else {
            editShotButton.disabled = true;
            deleteShotButton.disabled = true;
        }
    }

    function saveShotChanges() {
        if (!selectedShot) return;
    
        const newSituation = editSituationSelect.value;
        const newShotType = editShotTypeSelect.value;
    
        // Recalculate xG using the new values
        const shot = currentMatch.shots[selectedShot.index];
        const attackingLeft = (shot.team === 'home' && currentMatch.homeAttacking === 'left') || (shot.team === 'away' && currentMatch.awayAttacking === 'left');
        let apiX = shot.x;
        if (attackingLeft) {
            apiX = PITCH_LENGTH_METERS - shot.x;
        }
        callXGPrediction(apiX, shot.y, newSituation, newShotType).then(xg => {
            if (xg !== null) {
                // On successful prediction, update both the master list and the selected shot object
                currentMatch.shots[selectedShot.index].situation = newSituation;
                currentMatch.shots[selectedShot.index].shotType = newShotType;
                currentMatch.shots[selectedShot.index].xg = xg;
    
                // This ensures that if the user re-opens the edit modal, it shows the latest data
                selectedShot.situation = newSituation;
                selectedShot.shotType = newShotType;
                selectedShot.xg = xg;
    
                updateMatchInDB();
                updateXgDisplay();
                updateShotLists();
            }
        });
    
        closeEditShotModal();
    }

    function deleteSelectedShot() {
        if (!selectedShot) return;

        currentMatch.shots.splice(selectedShot.index, 1);
        selectedShot = null;
        
        // Re-index shots
        currentMatch.shots.forEach((shot, index) => {
            shot.index = index;
        });

        updateMatchInDB();
        updateXgDisplay();
        updateShotLists();
        drawPitch();
        updateEditDeleteButtons();
    }
 
    function updateDeleteMatchButton() {
        deleteMatchButton.disabled = !currentMatch;
    }
 
    function deleteCurrentMatch() {
        if (!currentMatch || !currentMatch.id) return;
 
        if (!confirm(`Are you sure you want to delete the match "${currentMatch.name}"? This action cannot be undone.`)) {
            return;
        }
 
        const transaction = db.transaction(['matches'], 'readwrite');
        const objectStore = transaction.objectStore('matches');
        const request = objectStore.delete(currentMatch.id);
 
        request.onsuccess = () => {
            alert('Match deleted successfully!');
            resetToNewMatch();
            loadMatchesIntoSelect();
            closeModal();
        };
        request.onerror = (event) => console.error('Error deleting match:', event.target.errorCode);
    }
 
    async function callXGPrediction(x, y, situation, shot_type) {
        const fetchBody = {
            x: x,
            y: y,
            situation: situation,
            shot_type: shot_type,
            normalisation: { is_normalised: false, max_pitch_width: 68, max_pitch_length: 105 }
        };

        try {
            const response = await fetch('http://127.0.0.1:5000/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(fetchBody)
            });
            const data = await response.json();
            if (data.xG !== undefined) {
                return data.xG;
            } else {
                throw new Error(data.error || 'Unexpected response format');
            }
        } catch (err) {
            console.error('Fetch error:', err);
            alert('Error getting xG prediction.');
            return null;
        }
    }

    // --- INITIALIZATION ---
    initDB();

    // Set default values for situation and shot type on initial load
    situationSelect.value = '';
    shotTypeSelect.value = '';
    updateTeamLabels(); // Set initial match title
});