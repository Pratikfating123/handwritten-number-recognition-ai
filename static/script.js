// ============================================================
// HANDWRITTEN NUMBER RECOGNITION AI
// FRONTEND JAVASCRIPT
// ============================================================

const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");

const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");

const predictionResult =
    document.getElementById("predictionResult");

const confidenceSection =
    document.getElementById("confidenceSection");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceBar =
    document.getElementById("confidenceBar");

const totalPredictions =
    document.getElementById("totalPredictions");

const averageConfidence =
    document.getElementById("averageConfidence");

const historyContainer =
    document.getElementById("historyContainer");

const clearHistoryBtn =
    document.getElementById("clearHistoryBtn");


// ============================================================
// CANVAS CONFIGURATION
// ============================================================

let isDrawing = false;

const CANVAS_BACKGROUND = "#ffffff";
const DRAW_COLOR = "#000000";


// ============================================================
// INITIALIZE CANVAS
// ============================================================

function setupCanvas() {

    ctx.fillStyle = CANVAS_BACKGROUND;

    ctx.fillRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.strokeStyle = DRAW_COLOR;

    ctx.lineWidth = 18;

    ctx.lineCap = "round";

    ctx.lineJoin = "round";
}


setupCanvas();


// ============================================================
// GET CANVAS POSITION
// ============================================================

function getPosition(event) {

    const rect =
        canvas.getBoundingClientRect();

    let clientX;
    let clientY;


    if (
        event.touches &&
        event.touches.length > 0
    ) {

        clientX =
            event.touches[0].clientX;

        clientY =
            event.touches[0].clientY;

    } else {

        clientX =
            event.clientX;

        clientY =
            event.clientY;
    }


    return {

        x:
            (clientX - rect.left) *
            (canvas.width / rect.width),

        y:
            (clientY - rect.top) *
            (canvas.height / rect.height)

    };
}


// ============================================================
// START DRAWING
// ============================================================

function startDrawing(event) {

    event.preventDefault();

    isDrawing = true;


    const position =
        getPosition(event);


    ctx.beginPath();

    ctx.moveTo(
        position.x,
        position.y
    );
}


// ============================================================
// DRAW
// ============================================================

function draw(event) {

    if (!isDrawing) {
        return;
    }


    event.preventDefault();


    const position =
        getPosition(event);


    ctx.lineTo(
        position.x,
        position.y
    );


    ctx.stroke();
}


// ============================================================
// STOP DRAWING
// ============================================================

function stopDrawing() {

    if (!isDrawing) {
        return;
    }


    isDrawing = false;

    ctx.closePath();
}


// ============================================================
// MOUSE EVENTS
// ============================================================

canvas.addEventListener(
    "mousedown",
    startDrawing
);

canvas.addEventListener(
    "mousemove",
    draw
);

canvas.addEventListener(
    "mouseup",
    stopDrawing
);

canvas.addEventListener(
    "mouseleave",
    stopDrawing
);


// ============================================================
// TOUCH EVENTS
// ============================================================

canvas.addEventListener(
    "touchstart",
    startDrawing,
    {
        passive: false
    }
);


canvas.addEventListener(
    "touchmove",
    draw,
    {
        passive: false
    }
);


canvas.addEventListener(
    "touchend",
    stopDrawing
);


// ============================================================
// CLEAR CANVAS
// ============================================================

clearBtn.addEventListener(
    "click",
    function () {

        setupCanvas();

        resetPrediction();

    }
);


// ============================================================
// RESET PREDICTION
// ============================================================

function resetPrediction() {

    predictionResult.innerHTML = `

        <div class="placeholder-icon">
            ?
        </div>

        <h3>
            Draw one or more digits
        </h3>

        <p>
            Try 7, 27, 583, or 2026
        </p>

    `;


    confidenceSection.classList.add(
        "hidden"
    );


    confidenceBar.style.width =
        "0%";


    confidenceValue.textContent =
        "0%";


    resetProbabilityPanel();
}


// ============================================================
// PREDICT NUMBER
// ============================================================

predictBtn.addEventListener(
    "click",
    async function () {

        predictBtn.disabled = true;

        predictBtn.innerHTML =
            "Analyzing...";


        predictionResult.innerHTML = `

            <div class="placeholder-icon">
                ...
            </div>

            <h3>
                Analyzing handwriting
            </h3>

            <p>
                Detecting and recognizing digits...
            </p>

        `;


        try {

            const blob =
                await new Promise(
                    function (resolve) {

                        canvas.toBlob(
                            resolve,
                            "image/png"
                        );

                    }
                );


            if (!blob) {

                throw new Error(
                    "Could not capture canvas."
                );

            }


            const formData =
                new FormData();


            formData.append(
                "image",
                blob,
                "handwritten-number.png"
            );


            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            console.log(
                "Prediction response:",
                data
            );


            if (!data.success) {

                showError(
                    data.error,
                    data.type
                );

                return;
            }


            showPrediction(
                data
            );


            await loadHistory();

            await loadAnalytics();

        }

        catch (error) {

            console.error(
                "Prediction error:",
                error
            );


            showError(
                error.message,
                "server"
            );

        }

        finally {

            predictBtn.disabled =
                false;


            predictBtn.innerHTML =
                "<span>✦</span> Predict Number";

        }

    }
);


// ============================================================
// SHOW PREDICTION
// ============================================================

function showPrediction(data) {

    predictionResult.innerHTML = `

        <div class="predicted-digit">
            ${escapeHtml(data.number)}
        </div>

        <h3>
            ${escapeHtml(data.number)} detected
        </h3>

        <p>
            ${data.digit_count} digit(s) recognized
        </p>

    `;


    confidenceSection.classList.remove(
        "hidden"
    );


    confidenceValue.textContent =
        `${data.overall_confidence}%`;


    confidenceBar.style.width =
        `${Math.min(
            data.overall_confidence,
            100
        )}%`;


    renderPerDigitProbabilities(
        data.digits
    );
}


// ============================================================
// PER-DIGIT PROBABILITY VISUALIZATION
// ============================================================

function renderPerDigitProbabilities(
    digits
) {

    const container =
        document.getElementById(
            "probabilityList"
        );


    if (!container) {
        return;
    }


    if (
        !digits ||
        digits.length === 0
    ) {

        container.innerHTML = `
            <p>No probability data available.</p>
        `;

        return;
    }


    let html = "";


    digits.forEach(
        function (item, index) {

            const predictedDigit =
                item.digit;


            const confidence =
                Number(
                    item.confidence
                );


            const probabilities =
                item.probabilities || {};


            html += `

                <div class="digit-probability-card">

                    <div class="digit-probability-header">

                        <div class="digit-position">

                            <span>
                                Digit ${index + 1}
                            </span>

                            <strong>
                                ${predictedDigit}
                            </strong>

                        </div>

                        <div class="digit-confidence">

                            ${confidence.toFixed(2)}%

                        </div>

                    </div>


                    <div class="digit-main-confidence">

                        <div class="digit-main-track">

                            <div
                                class="digit-main-fill"
                                style="width:${Math.min(
                                    confidence,
                                    100
                                )}%">
                            </div>

                        </div>

                    </div>


                    <div class="probability-grid">

                        ${createProbabilityRows(
                            probabilities,
                            predictedDigit
                        )}

                    </div>

                </div>

            `;

        }
    );


    container.innerHTML =
        html;
}


// ============================================================
// CREATE 10-CLASS PROBABILITY ROWS
// ============================================================

function createProbabilityRows(
    probabilities,
    predictedDigit
) {

    let html = "";


    for (
        let digit = 0;
        digit <= 9;
        digit++
    ) {

        const value =
            Number(
                probabilities[
                    String(digit)
                ] || 0
            );


        const active =
            digit === predictedDigit;


        html += `

            <div
                class="probability-item ${
                    active
                        ? "probability-active"
                        : ""
                }">

                <span class="probability-digit">
                    ${digit}
                </span>


                <div class="probability-track">

                    <div
                        class="probability-fill"
                        style="width:${Math.min(
                            value,
                            100
                        )}%">
                    </div>

                </div>


                <span class="probability-value">
                    ${value.toFixed(2)}%
                </span>

            </div>

        `;

    }


    return html;
}


// ============================================================
// RESET PROBABILITY PANEL
// ============================================================

function resetProbabilityPanel() {

    const container =
        document.getElementById(
            "probabilityList"
        );


    if (!container) {
        return;
    }


    container.innerHTML = `

        <div class="probability-empty">

            <span>
                Draw a number to see
                per-digit probabilities.
            </span>

        </div>

    `;
}


// ============================================================
// ERROR MESSAGE
// ============================================================

function showError(
    message,
    type
) {

    confidenceSection.classList.add(
        "hidden"
    );


    confidenceBar.style.width =
        "0%";


    confidenceValue.textContent =
        "0%";


    let icon = "!";


    if (type === "empty") {

        icon = "✎";

    }


    if (type === "invalid") {

        icon = "?";

    }


    predictionResult.innerHTML = `

        <div class="placeholder-icon error-icon">
            ${icon}
        </div>

        <h3>
            Please try again
        </h3>

        <p>
            ${escapeHtml(
                message ||
                "Unable to recognize the drawing."
            )}
        </p>

    `;


    resetProbabilityPanel();
}


// ============================================================
// LOAD HISTORY
// ============================================================

async function loadHistory() {

    try {

        const response =
            await fetch(
                "/history",
                {
                    method: "GET",
                    cache: "no-store"
                }
            );


        const data =
            await response.json();


        if (!data.success) {
            return;
        }


        updateStatistics(
            data
        );


        renderHistory(
            data.history
        );

    }

    catch (error) {

        console.error(
            "History error:",
            error
        );

    }
}


// ============================================================
// UPDATE STATISTICS
// ============================================================

function updateStatistics(
    data
) {

    if (totalPredictions) {

        totalPredictions.textContent =
            data.total;

    }


    if (averageConfidence) {

        averageConfidence.textContent =
            `${data.average_confidence}%`;

    }
}


// ============================================================
// RENDER HISTORY
// ============================================================

function renderHistory(
    history
) {

    if (!history || history.length === 0) {

        historyContainer.innerHTML = `

            <div class="empty-history">

                <div class="history-empty-icon">
                    ⌁
                </div>

                <h3>
                    No predictions yet
                </h3>

                <p>
                    Your prediction history
                    will appear here.
                </p>

            </div>

        `;

        return;
    }


    let html = "";


    history.forEach(
        function (item, index) {

            html += `

                <div class="history-row">

                    <div class="history-number">
                        ${index + 1}
                    </div>

                    <div class="history-digit">
                        ${item.digit}
                    </div>

                    <div class="history-info">

                        <strong>
                            Digit ${item.digit}
                        </strong>

                        <span>
                            CNN Prediction
                        </span>

                    </div>

                    <div class="history-confidence">

                        <strong>
                            ${Number(
                                item.confidence
                            ).toFixed(2)}%
                        </strong>

                        <span>
                            Confidence
                        </span>

                    </div>

                    <div class="history-time">
                        ${item.time}
                    </div>

                </div>

            `;

        }
    );


    historyContainer.innerHTML =
        html;
}


// ============================================================
// CLEAR HISTORY
// ============================================================

clearHistoryBtn.addEventListener(
    "click",
    async function () {

        try {

            const response =
                await fetch(
                    "/history/clear",
                    {
                        method: "POST"
                    }
                );


            const data =
                await response.json();


            if (data.success) {

                await loadHistory();

                await loadAnalytics();

            }

        }

        catch (error) {

            console.error(
                "Clear history error:",
                error
            );

        }

    }
);


// ============================================================
// LOAD ANALYTICS
// ============================================================

async function loadAnalytics() {

    try {

        const response =
            await fetch(
                "/analytics",
                {
                    method: "GET",
                    cache: "no-store"
                }
            );


        const data =
            await response.json();


        if (!data.success) {
            return;
        }


        // ----------------------------------------------------
        // MOST COMMON
        // ----------------------------------------------------

        const mostCommon =
            document.getElementById(
                "mostCommonDigit"
            );


        const mostCommonCount =
            document.getElementById(
                "mostCommonCount"
            );


        if (
            mostCommon &&
            data.most_common_digit !== null
        ) {

            mostCommon.textContent =
                data.most_common_digit;


            mostCommonCount.textContent =
                `${data.most_common_count} predictions`;

        }


        // ----------------------------------------------------
        // HIGHEST CONFIDENCE
        // ----------------------------------------------------

        const highest =
            document.getElementById(
                "highestConfidence"
            );


        const highestDigit =
            document.getElementById(
                "highestConfidenceDigit"
            );


        if (highest) {

            highest.textContent =
                `${data.highest_confidence}%`;

        }


        if (
            highestDigit &&
            data.highest_confidence_digit !== null
        ) {

            highestDigit.textContent =
                `Digit ${data.highest_confidence_digit}`;

        }


        // ----------------------------------------------------
        // LATEST
        // ----------------------------------------------------

        const latest =
            document.getElementById(
                "latestDigit"
            );


        const latestTime =
            document.getElementById(
                "latestPredictionTime"
            );


        if (
            latest &&
            data.latest_digit !== null
        ) {

            latest.textContent =
                data.latest_digit;

        }


        if (
            latestTime &&
            data.latest_time
        ) {

            latestTime.textContent =
                `Predicted at ${data.latest_time}`;

        }


        // ----------------------------------------------------
        // CHART
        // ----------------------------------------------------

        renderDigitChart(
            data.digit_counts
        );

    }

    catch (error) {

        console.error(
            "Analytics error:",
            error
        );

    }
}


// ============================================================
// DIGIT DISTRIBUTION CHART
// ============================================================

function renderDigitChart(
    digitCounts
) {

    const chart =
        document.getElementById(
            "digitChart"
        );


    if (!chart) {
        return;
    }


    const values =
        Object.values(
            digitCounts
        );


    const maximum =
        Math.max(
            ...values,
            1
        );


    let html = "";


    for (
        let digit = 0;
        digit <= 9;
        digit++
    ) {

        const count =
            digitCounts[
                String(digit)
            ] || 0;


        const height =
            count === 0
                ? 4
                : Math.max(
                    10,
                    (
                        count /
                        maximum
                    ) * 100
                );


        html += `

            <div class="chart-column">

                <div class="chart-count">
                    ${count}
                </div>

                <div class="chart-bar-container">

                    <div
                        class="chart-bar"
                        style="height:${height}%">
                    </div>

                </div>

                <div class="chart-digit">
                    ${digit}
                </div>

            </div>

        `;

    }


    chart.innerHTML =
        html;
}


// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHtml(
    value
) {

    const div =
        document.createElement(
            "div"
        );


    div.textContent =
        value;


    return div.innerHTML;
}


// ============================================================
// INITIAL LOAD
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        resetProbabilityPanel();

        loadHistory();

        loadAnalytics();

    }
);