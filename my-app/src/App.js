import React, { useState } from 'react';
import { Upload, X, Loader2, Zap, AlertTriangle, CheckCircle, Image } from 'lucide-react'; 

// IMPORTANT: This is the URL for your running Python server (server.py)
const ML_API_ENDPOINT = 'http://127.0.0.1:5000/predict'; 

// Helper component for displaying the outcome (Success/Error)
const MessageCard = ({ type, message, details }) => {
  const isSuccess = type === 'success';
  const icon = isSuccess ? CheckCircle : AlertTriangle;
  const colorClass = isSuccess ? 'message-success' : 'message-error';
  const title = isSuccess ? "Prediction Successful!" : "Prediction Failed";

  return (
    <div className={`message-card ${colorClass}`}>
      <div className="message-content">
        {React.createElement(icon, { className: 'message-icon' })}
        <div>
          <h3 className="message-title">{title}</h3>
          <p className="message-text">{message}</p>
          {details && <p className="message-details">Details: {details}</p>}
        </div>
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null); 

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImageFile(file);
      setImagePreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setMessage(null);
    }
  };

  const handleClear = () => {
    setImageFile(null);
    setImagePreviewUrl('');
    setPrediction(null);
    setMessage(null);
    document.getElementById('file-upload').value = null;
  };

  const handlePrediction = async () => {
    if (!imageFile) {
      setMessage({ type: 'error', message: "Please upload an image file first." });
      return;
    }

    setLoading(true);
    setPrediction(null);
    setMessage(null);

    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const response = await fetch(ML_API_ENDPOINT, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server returned HTTP status ${response.status}: ${errorText.substring(0, 100)}...`);
      }

      const data = await response.json();

      if (data.error) {
         setMessage({ 
           type: 'error', 
           message: "Prediction failed on the server side.", 
           details: data.error 
         });
      } else {
        setPrediction(data);
        setMessage({ 
          type: 'success', 
          message: `The predicted face shape is ${data.predicted_shape} with ${Math.round(parseFloat(data.confidence) * 100)}% confidence.`, 
        });
      }
    } catch (error) {
      console.error("Prediction Error:", error);
      setMessage({ 
        type: 'error', 
        message: "Failed to connect to the ML API.", 
        details: `Is server.py running? Error: ${error.message.substring(0, 80)}...` 
      });
    } finally {
      setLoading(false);
    }
  };
  
  const getConfidenceColorClass = (confidence) => {
    const normalized = parseFloat(confidence);
    if (normalized >= 0.8) return 'bar-green';
    if (normalized >= 0.6) return 'bar-yellow';
    return 'bar-red';
  };
  
  const getConfidenceTextClass = (confidence) => {
    const normalized = parseFloat(confidence);
    if (normalized >= 0.8) return 'text-green';
    if (normalized >= 0.6) return 'text-yellow';
    return 'text-red';
  };

  return (
    <>
      <style>{`
        /* Reset and Base Styles */
        body { margin: 0; padding: 0; font-family: 'Inter', sans-serif; background-color: #f9fafb; }
        .app-container {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 32px 16px;
        }

        /* Header */
        header {
          width: 100%;
          max-width: 900px;
          text-align: center;
          margin-bottom: 40px;
        }
        .header-title {
          font-size: 40px;
          font-weight: 800;
          color: #1f2937;
          letter-spacing: -0.025em;
        }
        .header-subtitle {
          margin-top: 12px;
          font-size: 20px;
          color: #6b7280;
        }

        /* Main Card Layout */
        .main-card {
          width: 100%;
          max-width: 900px;
          background-color: white;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
          border-radius: 16px;
          padding: 40px;
          border: 1px solid #f3f4f6;
        }
        .grid-container {
          display: grid;
          grid-template-columns: 1fr;
          gap: 32px;
        }
        @media (min-width: 1024px) {
          .grid-container {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        /* Column Styles */
        .column {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .column-header {
          font-size: 24px;
          font-weight: 600;
          color: #374151;
          border-bottom: 1px solid #e5e7eb;
          padding-bottom: 8px;
          margin-bottom: 8px;
          display: flex;
          align-items: center;
        }
        .column-icon-indigo { color: #4f46e5; margin-right: 8px; width: 24px; height: 24px; }
        .column-icon-emerald { color: #10b981; margin-right: 8px; width: 24px; height: 24px; }


        /* File Input Style */
        .file-input-label { display: block; }
        .file-input {
          display: block;
          width: 100%;
          color: #6b7280;
          padding: 8px 0;
        }
        .file-input::-webkit-file-upload-button {
          margin-right: 16px;
          padding: 8px 16px;
          border-radius: 9999px;
          border: 0;
          font-size: 14px;
          font-weight: 600;
          background-color: #eef2ff;
          color: #4f46e5;
          cursor: pointer;
          transition: background-color 0.15s;
        }
        .file-input::-webkit-file-upload-button:hover {
          background-color: #e0e7ff;
        }
        

        /* Image Preview Area */
        .preview-area {
          position: relative;
          width: 100%;
          height: 320px;
          background-color: #f3f4f6;
          border: 2px dashed #d1d5db;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          overflow: hidden;
        }
        .preview-img {
          width: 100%;
          height: 100%;
          object-fit: contain;
        }
        .preview-placeholder {
          text-align: center;
          color: #9ca3af;
        }
        .preview-placeholder-icon {
          width: 40px;
          height: 40px;
          margin: 0 auto 8px;
        }
        .clear-button {
          position: absolute;
          top: 8px;
          right: 8px;
          padding: 4px;
          background-color: #ef4444;
          border-radius: 9999px;
          color: white;
          cursor: pointer;
          transition: background-color 0.15s, transform 0.15s;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .clear-button:hover {
          background-color: #dc2626;
          transform: scale(1.05);
        }
        .clear-button-icon { width: 20px; height: 20px; }


        /* Prediction Button */
        .predict-button {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 12px 24px;
          border-radius: 12px;
          font-size: 18px;
          font-weight: 700;
          transition: background-color 0.3s, transform 0.3s, box-shadow 0.3s;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
          border: none;
        }
        .predict-button:not(:disabled) {
          background-color: #4f46e5;
          color: white;
          cursor: pointer;
        }
        .predict-button:not(:disabled):hover {
          background-color: #4338ca;
          transform: scale(1.01);
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        }
        .predict-button:disabled {
          background-color: #9ca3af;
          color: #f3f4f6;
          cursor: not-allowed;
        }
        .predict-button-icon {
          width: 24px;
          height: 24px;
          margin-right: 8px;
        }
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        /* Result Area */
        .result-area {
          padding: 24px;
          border-radius: 12px;
          height: 128px;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.3s;
        }
        .result-empty {
          border: 4px solid #d1d5db;
          background-color: #f3f4f6;
        }
        .result-predicted {
          border: 4px solid #818cf8;
          background-color: #eef2ff;
          box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        }
        .result-text {
          text-align: center;
        }
        .result-label {
          font-size: 18px;
          color: #4b5563;
          font-weight: 600;
          margin-bottom: 4px;
        }
        .result-shape {
          font-size: 48px;
          font-weight: 800;
          color: #4f46e5;
          letter-spacing: 0.05em;
        }
        .result-placeholder {
          font-size: 20px;
          color: #9ca3af;
        }
        
        /* Message Card */
        .message-card {
          padding: 16px;
          border-radius: 8px;
          border-left: 4px solid;
          margin-top: 16px;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .message-success { border-color: #34d399; background-color: #ecfdf5; }
        .message-error { border-color: #f87171; background-color: #fef2f2; }
        .message-content { display: flex; align-items: center; }
        .message-icon { width: 24px; height: 24px; margin-right: 12px; flex-shrink: 0; }
        .message-success .message-icon { color: #059669; }
        .message-error .message-icon { color: #ef4444; }
        .message-title { font-size: 18px; font-weight: 600; color: #1f2937; }
        .message-text { font-size: 14px; font-weight: 500; color: #374151; }
        .message-details { font-size: 12px; margin-top: 4px; color: #4b5563; font-style: italic; }

        /* Confidence Breakdown */
        .confidence-box {
          margin-top: 16px;
          padding: 16px;
          background-color: white;
          border-radius: 12px;
          box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
          border: 1px solid #e5e7eb;
        }
        .confidence-title {
          font-size: 18px;
          font-weight: 700;
          margin-bottom: 12px;
          color: #1f2937;
        }
        .confidence-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 14px;
          margin-bottom: 8px;
        }
        .confidence-label {
          font-weight: 500;
          color: #4b5563;
        }
        .confidence-bar-container {
          width: 75%;
          background-color: #e5eeeb;
          border-radius: 9999px;
          height: 10px;
        }
        .confidence-bar {
          height: 10px;
          border-radius: 9999px;
          transition: width 0.5s;
        }
        .bar-green { background-color: #10b981; }
        .bar-yellow { background-color: #f59e0b; }
        .bar-red { background-color: #ef4444; }
        
        .confidence-percentage {
          font-size: 12px;
          font-weight: 600;
          padding: 2px 8px;
          border-radius: 9999px;
        }
        .text-green { background-color: #d1fae5; color: #065f46; }
        .text-yellow { background-color: #fffac2; color: #92400e; }
        .text-red { background-color: #fee2e2; color: #991b1b; }


        /* Footer */
        footer {
          margin-top: 32px;
          text-align: center;
          color: #6b7280;
          font-size: 14px;
          width: 100%;
          max-width: 900px;
          padding-top: 16px;
          border-top: 1px solid #e5e7eb;
        }
        .api-note code {
          background-color: #e5e7eb;
          padding: 2px 4px;
          border-radius: 4px;
          font-size: 12px;
        }
      `}</style>

      <div className="app-container">
        
        {/* Header */}
        <header>
          <h1 className="header-title">
            Face Shape Prediction App
          </h1>
          <p className="header-subtitle">
            Upload a face image to predict the shape using VGG16 features and a Decision Tree.
          </p>
        </header>

        {/* Main Content Card */}
        <main className="main-card">
          
          <div className="grid-container">
            
            {/* Left Column: Image Upload & Preview */}
            <div className="column">
              <h2 className="column-header">
                <Upload className="column-icon-indigo" />
                Upload Image
              </h2>
              
              {/* File Input */}
              <label className="file-input-label">
                <span style={{ display: 'none' }}>Choose profile photo</span>
                <input 
                  id="file-upload"
                  type="file" 
                  accept="image/*"
                  onChange={handleFileChange} 
                  className="file-input"
                />
              </label>

              {/* Image Preview Area */}
              <div className="preview-area">
                {imagePreviewUrl ? (
                  <>
                    <img src={imagePreviewUrl} alt="Preview" className="preview-img" />
                    <button 
                      onClick={handleClear} 
                      className="clear-button"
                      title="Clear Image"
                    >
                      <X className="clear-button-icon" />
                    </button>
                  </>
                ) : (
                  <div className="preview-placeholder">
                    <Image className="preview-placeholder-icon" />
                    <p>No Image Selected</p>
                  </div>
                )}
              </div>

              {/* Prediction Button */}
              <button
                onClick={handlePrediction}
                disabled={loading || !imageFile}
                className="predict-button"
              >
                {loading ? (
                  <Loader2 className="predict-button-icon animate-spin" />
                ) : (
                  <Zap className="predict-button-icon" />
                )}
                {loading ? 'Analyzing...' : 'Run Prediction'}
              </button>
            </div>

            {/* Right Column: Results & Status */}
            <div className="column">
              <h2 className="column-header">
                <CheckCircle className="column-icon-emerald" />
                Prediction Results
              </h2>

              {/* Status Message Area */}
              {message && <MessageCard {...message} />}
              
              {/* Main Prediction Output */}
              <div className={`result-area ${prediction ? 'result-predicted' : 'result-empty'}`}>
                {prediction ? (
                  <div className="result-text">
                    <p className="result-label">Predicted Face Shape:</p>
                    <p className="result-shape">
                      {prediction.predicted_shape}
                    </p>
                  </div>
                ) : (
                  <p className="result-placeholder">Prediction results will appear here.</p>
                )}
              </div>
              
              {/* Detailed Probabilities */}
              {prediction && prediction.full_probabilities && (
                <div className="confidence-box">
                  <h3 className="confidence-title">Confidence Breakdown</h3>
                  <div className="space-y-2">
                    {Object.entries(prediction.full_probabilities).map(([shape, confidence]) => (
                      <div key={shape} className="confidence-item">
                        <span className="confidence-label">{shape}</span>
                        <div className="confidence-bar-container">
                          <div 
                            className={`confidence-bar ${getConfidenceColorClass(confidence)}`} 
                            style={{ width: `${parseFloat(confidence) * 100}%` }}
                          ></div>
                        </div>
                        <span className={`confidence-percentage ${getConfidenceTextClass(confidence)}`}>
                          {Math.round(parseFloat(confidence) * 100)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>
        
        {/* Footer / API Status Note */}
        <footer>
            <p className="api-note">
              <span style={{ fontWeight: 700 }}>API Endpoint:</span> {ML_API_ENDPOINT} 
              (Requires <code>server.py</code> running)
            </p>
        </footer>
      </div>
    </>
  );
};

export default App;
