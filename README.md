# Used Car Price Predictor

A web application that predicts used car prices using machine learning.

## Features

- **AI-Powered Predictions**: Uses RandomForestRegressor to predict car prices
- **Interactive Web Interface**: Modern, responsive design with animations
- **Real-time Validation**: Client-side form validation
- **Admin Dashboard**: Model management and data insights
- **REST API**: JSON endpoints for predictions

## Tech Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **ML**: scikit-learn, pandas, numpy
- **Styling**: Custom CSS with animations

## Installation

1. **Clone or download** the project files

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data files exist**:
   - `CAR DETAILS FROM CAR DEKHO.csv` (training data)
   - `model.pkl` (trained model)
   - `scaler.pkl` (feature scaler)

## Usage

1. **Activate virtual environment** (if using one):
   ```bash
   # Windows
   car\Scripts\activate

   # Linux/Mac
   source car/bin/activate
   ```

2. **Run the application**:
   ```bash
   python main.py
   ```

3. **Open browser** and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Price prediction API
- `GET /health` - Health check

### Prediction API

**Request:**
```json
POST /predict
Content-Type: application/json

{
  "year": 2020,
  "km_driven": 50000,
  "seller_type": 0,
  "transmission": 0,
  "owner": 0
}
```

**Response:**
```json
{
  "price": 450000,
  "range": {
    "min": 420000,
    "max": 480000
  },
  "confidence": 75,
  "insights": [...]
}
```

## Project Structure

```
used_car/
├── main.py                 # Flask application
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Main web interface
├── CAR DETAILS FROM CAR DEKHO.csv  # Training data
├── model.pkl              # Trained ML model
├── scaler.pkl             # Feature scaler
└── used_car_cleaned.csv   # Processed data
```

## Model Training

The model is trained on car features including:
- Year of manufacture
- Kilometers driven
- Seller type (Individual/Dealer/Trustmark Dealer)
- Transmission type (Manual/Automatic)
- Owner type (First/Second/Third/Fourth/Test Drive)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.