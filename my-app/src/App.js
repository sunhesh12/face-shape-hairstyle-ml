
import React, { useState } from 'react';
import { Upload, X, Loader2, Zap, AlertTriangle, CheckCircle, Image, Scissors, Sparkles } from 'lucide-react'; 

const ML_API_ENDPOINT = 'http://127.0.0.1:5000/predict'; 

// Message Card
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
        throw new Error(`Server returned HTTP ${response.status}: ${errorText}`);
      }

      const data = await response.json();

      if (data.error) {
        setMessage({
          type: 'error',
          message: "Prediction failed on the server.",
          details: data.error
        });
      } else {
        setPrediction(data);
        setMessage({
          type: 'success',
          message: `Predicted face shape: ${data.predicted_shape} (${Math.round(parseFloat(data.confidence) * 100)}% confidence)`
        });
      }
    } catch (error) {
      setMessage({
        type: 'error',
        message: "Failed to connect to ML API.",
        details: `Is server.py running? Error: ${error.message}`
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style>{`
        body {
          margin: 0;
          font-family: 'Inter', sans-serif;
          background: linear-gradient(135deg, #eef2ff, #fdf2f8);
          color: #1f2937;
        }
        .app-container {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 32px 16px;
        }
        header {
          width: 100%;
          max-width: 900px;
          text-align: center;
          margin-bottom: 40px;
        }
        .header-title {
          font-size: 42px;
          font-weight: 800;
          color: #312e81;
        }
        .header-subtitle {
          margin-top: 12px;
          font-size: 18px;
          color: #6b7280;
        }
        .main-card {
          width: 100%;
          max-width: 950px;
          background-color: rgba(255,255,255,0.8);
          backdrop-filter: blur(12px);
          border-radius: 24px;
          padding: 40px;
          box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        }
        .grid-container {
          display: grid;
          gap: 32px;
          grid-template-columns: 1fr;
        }
        @media (min-width: 1024px) {
          .grid-container {
            grid-template-columns: repeat(2, 1fr);
          }
        }
        .column {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .column-header {
          font-size: 22px;
          font-weight: 600;
          color: #374151;
          border-bottom: 1px solid #e5e7eb;
          padding-bottom: 8px;
          display: flex;
          align-items: center;
        }
        .column-icon-indigo {
          color: #4f46e5;
          margin-right: 8px;
        }
        .column-icon-emerald {
          color: #10b981;
          margin-right: 8px;
        }
        .file-input {
          width: 100%;
          color: #6b7280;
        }
        .file-input::-webkit-file-upload-button {
          padding: 8px 16px;
          border-radius: 9999px;
          border: 0;
          background-color: #eef2ff;
          color: #4f46e5;
          font-weight: 600;
          cursor: pointer;
        }
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
        .clear-button {
          position: absolute;
          top: 8px;
          right: 8px;
          background-color: #ef4444;
          border-radius: 9999px;
          color: white;
          padding: 4px;
          cursor: pointer;
        }
        .predict-button {
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 12px 24px;
          border-radius: 12px;
          font-size: 18px;
          font-weight: 700;
          border: none;
          background: linear-gradient(90deg, #6366f1, #8b5cf6);
          color: white;
          cursor: pointer;
          transition: 0.3s;
        }
        .predict-button:hover {
          transform: translateY(-2px);
          background: linear-gradient(90deg, #4f46e5, #7c3aed);
        }

        /* Highlight Hairstyle Section */
        .hairstyle-highlight {
          background: linear-gradient(135deg, #f0f9ff, #e0f2fe, #fdf2f8);
          padding: 32px;
          border-radius: 20px;
          text-align: center;
          box-shadow: 0 10px 25px rgba(79,70,229,0.1);
          border: 2px solid #c7d2fe;
          margin-top: 16px;
        }
        .hairstyle-highlight h3 {
          font-size: 28px;
          font-weight: 800;
          color: #312e81;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          margin-bottom: 20px;
        }
        .hairstyle-item {
          background-color: white;
          border-radius: 12px;
          padding: 12px 20px;
          margin: 8px auto;
          width: 80%;
          box-shadow: 0 4px 10px rgba(0,0,0,0.05);
          font-size: 17px;
          font-weight: 500;
          color: #374151;
          transition: all 0.2s ease-in-out;
        }
        .hairstyle-item:hover {
          transform: scale(1.02);
          background-color: #eef2ff;
          color: #4338ca;
        }

        .result-area {
          margin-top: 24px;
          padding: 20px;
          border-radius: 16px;
          background-color: #eef2ff;
          text-align: center;
          border: 2px solid #c7d2fe;
        }
        .result-shape {
          font-size: 40px;
          font-weight: 800;
          color: #4f46e5;
        }
        .message-card {
          padding: 16px;
          border-radius: 8px;
          border-left: 4px solid;
          margin-top: 16px;
        }
        .message-success {
          border-color: #34d399;
          background-color: #ecfdf5;
        }
        .message-error {
          border-color: #f87171;
          background-color: #fef2f2;
        }
        footer {
          margin-top: 40px;
          text-align: center;
          color: #6b7280;
          font-size: 14px;
          border-top: 1px solid #e5e7eb;
          padding-top: 16px;
        }
      `}</style>

      <div className="app-container">
        <header>
          <h1 className="header-title">Smart Face Shape & Hairstyle Recommender 💇‍♀️</h1>
          <p className="header-subtitle">
            Upload your face image to discover your face shape and get personalized hairstyle recommendations.
          </p>
        </header>

        <main className="main-card">
          <div className="grid-container">
            {/* Left Column */}
            <div className="column">
              <h2 className="column-header"><Upload className="column-icon-indigo" /> Upload Image</h2>
              <input id="file-upload" type="file" accept="image/*" onChange={handleFileChange} className="file-input" />
              <div className="preview-area">
                {imagePreviewUrl ? (
                  <>
                    <img src={imagePreviewUrl} alt="Preview" className="preview-img" />
                    <button onClick={handleClear} className="clear-button"><X /></button>
                  </>
                ) : (
                  <div className="preview-placeholder">
                    <Image size={40} /><p>No Image Selected</p>
                  </div>
                )}
              </div>
              <button onClick={handlePrediction} disabled={loading || !imageFile} className="predict-button">
                {loading ? <Loader2 className="animate-spin" /> : <Zap />} {loading ? 'Analyzing...' : 'Run Prediction'}
              </button>
            </div>

            {/* Right Column */}
            <div className="column">
              <h2 className="column-header"><CheckCircle className="column-icon-emerald" /> Prediction Results</h2>
              {message && <MessageCard {...message} />}

              {prediction && (
                <>
                  {/* Highlight Hairstyle Section */}
                  <div className="hairstyle-highlight">
                    <h3><Scissors /> Recommended Hairstyles <Sparkles color="#f59e0b" /></h3>
                    {prediction.recommended_hairstyles.map((style, i) => (
                      <div className="hairstyle-item" key={i}>{style}</div>
                    ))}
                  </div>

                  {/* Face Shape Section */}
                  <div className="result-area">
                    <p className="result-shape">{prediction.predicted_shape}</p>
                    <p>Confidence: {(parseFloat(prediction.confidence) * 100).toFixed(1)}%</p>
                  </div>
                </>
              )}
            </div>
          </div>
        </main>

        <footer>
          <p>API Endpoint: {ML_API_ENDPOINT} (Requires <code>server.py</code> running)</p>
        </footer>
      </div>
    </>
  );
};

export default App;
