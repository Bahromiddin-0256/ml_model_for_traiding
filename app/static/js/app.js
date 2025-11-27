/**
 * Crypto Predictor - Frontend Application
 */

// Sample data for testing
const SAMPLE_DATA = {
    Open: 45123.45,
    High: 46789.12,
    Low: 44567.89,
    Close: 46234.56,
    Volume: 1523456789,
    SMA_7: 45500.00,
    SMA_25: 44800.00,
    SMA_99: 43200.00,
    EMA_7: 45600.00,
    EMA_25: 44900.00,
    EMA_99: 43400.00,
    MACD: 245.67,
    MACD_Signal: 198.34,
    MACD_Hist: 47.33,
    RSI: 58.5,
    BB_Upper: 48000.00,
    BB_Middle: 45500.00,
    BB_Lower: 43000.00,
    ATR: 1234.56,
    OBV: 987654321,
    Stoch_K: 65.5,
    Stoch_D: 62.3,
    ADX: 28.5,
    CCI: 85.4,
    Williams_R: -35.2,
    ROC: 2.45,
    MFI: 55.8,
    CMF: 0.15,
    VWAP: 45800.00,
    Parabolic_SAR: 44500.00,
    Ichimoku_Tenkan: 45300.00,
    Ichimoku_Kijun: 44800.00,
    Ichimoku_SenkouA: 45050.00,
    Ichimoku_SenkouB: 44200.00,
    Ichimoku_Chikou: 46234.56
};

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const resultContainer = document.getElementById('resultContainer');
const fillSampleBtn = document.getElementById('fillSampleBtn');

/**
 * Fill form with sample data
 */
function fillSampleData() {
    Object.entries(SAMPLE_DATA).forEach(([key, value]) => {
        const input = document.getElementById(key);
        if (input) {
            input.value = value;
        }
    });
}

/**
 * Get form data as object
 */
function getFormData(form) {
    const formData = new FormData(form);
    const data = {};
    
    for (const [key, value] of formData.entries()) {
        if (value !== '') {
            data[key] = parseFloat(value);
        }
    }
    
    return data;
}

/**
 * Display prediction result
 */
function displayResult(result) {
    const isUp = result.prediction === 1;
    const direction = isUp ? '↑' : '↓';
    const directionClass = isUp ? 'up' : 'down';
    const label = result.prediction_label;
    
    const probUp = (result.probability_up * 100).toFixed(1);
    const probDown = (result.probability_down * 100).toFixed(1);
    const confidence = (result.confidence * 100).toFixed(1);
    
    resultContainer.innerHTML = `
        <div class="prediction-result">
            <div class="prediction-direction ${directionClass}">${direction}</div>
            <div class="prediction-label">${label}</div>
            
            <div class="probability-bars">
                <div class="prob-item">
                    <span class="prob-label">UP</span>
                    <div class="prob-bar">
                        <div class="prob-fill up" style="width: ${probUp}%"></div>
                    </div>
                    <span class="prob-value">${probUp}%</span>
                </div>
                <div class="prob-item">
                    <span class="prob-label">DOWN</span>
                    <div class="prob-bar">
                        <div class="prob-fill down" style="width: ${probDown}%"></div>
                    </div>
                    <span class="prob-value">${probDown}%</span>
                </div>
            </div>
            
            <div class="confidence-badge">
                Confidence: ${confidence}%
            </div>
        </div>
    `;
}

/**
 * Display error message
 */
function displayError(message) {
    resultContainer.innerHTML = `
        <div class="result-error">
            <strong>Error</strong>
            <p>${message}</p>
        </div>
    `;
}

/**
 * Display loading state
 */
function displayLoading() {
    resultContainer.innerHTML = `
        <div class="result-placeholder">
            <div class="placeholder-icon">◈</div>
            <p class="loading">Analyzing indicators...</p>
        </div>
    `;
}

/**
 * Make prediction API call
 */
async function makePrediction(data) {
    const response = await fetch('/api/v1/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Prediction failed');
    }
    
    return response.json();
}

// Event Listeners
if (fillSampleBtn) {
    fillSampleBtn.addEventListener('click', fillSampleData);
}

if (predictionForm) {
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const data = getFormData(predictionForm);
        
        // Validate required fields
        const requiredFields = ['Open', 'High', 'Low', 'Close', 'Volume'];
        const missingFields = requiredFields.filter(field => !(field in data));
        
        if (missingFields.length > 0) {
            displayError(`Missing required fields: ${missingFields.join(', ')}`);
            return;
        }
        
        displayLoading();
        
        try {
            const result = await makePrediction(data);
            displayResult(result);
        } catch (error) {
            displayError(error.message);
        }
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Crypto Predictor initialized');
});