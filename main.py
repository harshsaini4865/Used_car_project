from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
feature_columns = ['year', 'km_driven', 'seller_type', 'transmission', 'owner']

# Stable mapping dictionaries for training and prediction
SELLER_TYPE_MAP = {
    'Individual': 0,
    'Dealer': 1,
    'Trustmark Dealer': 2
}
TRANSMISSION_MAP = {
    'Manual': 0,
    'Automatic': 1
}
OWNER_MAP = {
    'First Owner': 0,
    'Second Owner': 1,
    'Third Owner': 2,
    'Fourth & Above Owner': 3,
    'Test Drive Car': 4
}

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    
    try:
        # Check if model files exist
        if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Model and scaler loaded successfully")
        else:
            logger.warning("Model files not found. Training new model...")
            train_model()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        train_model()

def train_model():
    """Train a new model using the available data"""
    global model, scaler
    
    try:
        # Load training data
        if os.path.exists('CAR DETAILS FROM CAR DEKHO.csv'):
            df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
            logger.info(f"Training data loaded: {df.shape}")
            
            # Basic preprocessing
            df = preprocess_data(df)
            
            # Prepare features and target
            X = df[feature_columns]
            y = df['selling_price']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_scaled, y)
            
            # Save model and scaler
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
                
            logger.info("Model trained and saved successfully")
            
        else:
            logger.warning("Training data not found. Using mock model.")
            create_mock_model()
            
    except Exception as e:
        logger.error(f"Error training model: {e}")
        create_mock_model()

def create_mock_model():
    """Create a mock model for demonstration"""
    global model, scaler
    
    # Create a simple mock model
    class MockModel:
        def predict(self, X):
            # Simple prediction logic based on input features
            predictions = []
            for row in X:
                year, km_driven, seller_type, transmission, owner = row
                base_price = 1000000  # Base price in INR
                
                # Age factor
                age = max(0, 2024 - year)
                age_factor = max(0.3, 1 - (age * 0.08))
                
                # Mileage factor
                mileage_factor = max(0.4, 1 - (km_driven / 200000))
                
                # Owner factor
                owner_factor = max(0.5, 1 - (owner * 0.1))
                
                # Seller type adjustment
                seller_adjustment = 1.0
                if seller_type == 1:  # Dealer
                    seller_adjustment = 1.05
                elif seller_type == 2:  # Trustmark
                    seller_adjustment = 1.08
                
                # Transmission adjustment
                transmission_adjustment = 1.03 if transmission == 1 else 1.0
                
                final_price = base_price * age_factor * mileage_factor * owner_factor * seller_adjustment * transmission_adjustment
                predictions.append(max(50000, final_price))  # Minimum price of 50k
            
            return np.array(predictions)
    
    model = MockModel()
    scaler = StandardScaler()
    # Fit scaler with dummy data for mock model
    scaler.fit(np.array([[2020, 50000, 1, 1, 0]]))
    
    logger.info("Mock model created for demonstration")

def preprocess_data(df):
    """Preprocess the training data"""
    from pandas.api.types import is_string_dtype, is_object_dtype

    try:
        # Remove duplicates and handle missing values
        df = df.drop_duplicates()
        df = df.dropna()

        # Convert known categorical columns using stable mappings
        if 'seller_type' in df.columns and (is_string_dtype(df['seller_type']) or is_object_dtype(df['seller_type'])):
            df['seller_type'] = df['seller_type'].map(SELLER_TYPE_MAP).fillna(-1).astype(int)

        if 'transmission' in df.columns and (is_string_dtype(df['transmission']) or is_object_dtype(df['transmission'])):
            df['transmission'] = df['transmission'].map(TRANSMISSION_MAP).fillna(-1).astype(int)

        if 'owner' in df.columns and (is_string_dtype(df['owner']) or is_object_dtype(df['owner'])):
            df['owner'] = df['owner'].map(OWNER_MAP).fillna(-1).astype(int)

        # If any column is still non-numeric, attempt category codes fallback
        for col in ['seller_type', 'transmission', 'owner']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.Categorical(df[col]).codes

        # Ensure all required columns exist
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in data. Using default values.")
                df[col] = 0

        return df
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return df

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from request
        data = request.json
        
        # Validate input
        validation_result = validate_input(data)
        if not validation_result['valid']:
            return jsonify({
                'error': validation_result['message']
            }), 400
        
        # Prepare features
        features = np.array([[
            data['year'],
            data['km_driven'],
            data['seller_type'],
            data['transmission'],
            data['owner']
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Calculate confidence and range
        confidence = calculate_confidence(data)
        price_range = calculate_price_range(prediction, confidence)
        
        # Generate insights
        insights = generate_insights(data, prediction)
        
        response = {
            'price': int(prediction),
            'range': price_range,
            'confidence': confidence,
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: ₹{prediction:,.0f} for data: {data}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Internal server error. Please try again later.'
        }), 500

def validate_input(data):
    """Validate input data"""
    required_fields = ['year', 'km_driven', 'seller_type', 'transmission', 'owner']
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            return {'valid': False, 'message': f'Missing required field: {field}'}
    
    # Validate ranges
    current_year = datetime.now().year
    
    if not (1990 <= data['year'] <= current_year):
        return {'valid': False, 'message': f'Year must be between 1990 and {current_year}'}
    
    if data['km_driven'] < 0 or data['km_driven'] > 1000000:
        return {'valid': False, 'message': 'Kilometers driven must be between 0 and 1,000,000'}
    
    if data['seller_type'] not in [0, 1, 2]:
        return {'valid': False, 'message': 'Invalid seller type'}
    
    if data['transmission'] not in [0, 1]:
        return {'valid': False, 'message': 'Invalid transmission type'}
    
    if data['owner'] not in [0, 1, 2, 3, 4]:
        return {'valid': False, 'message': 'Invalid owner type'}
    
    return {'valid': True}

def calculate_confidence(data):
    """Calculate prediction confidence"""
    confidence = 70  # Base confidence
    
    # Increase confidence for recent cars with low mileage
    current_year = datetime.now().year
    age = current_year - data['year']
    
    if age <= 5 and data['km_driven'] <= 50000:
        confidence += 15
    
    if data['seller_type'] == 2:  # Trustmark dealer
        confidence += 10
    
    if data['owner'] == 0:  # First owner
        confidence += 5
    
    return min(confidence, 95)

def calculate_price_range(prediction, confidence):
    """Calculate price range based on confidence"""
    # Wider range for lower confidence
    range_factor = (100 - confidence) / 200 + 0.1  # 10% to 35% range

    min_price = int(prediction * (1 - range_factor))
    max_price = int(prediction * (1 + range_factor))

    return {
        'min': max(50000, min_price),  # Minimum 50k
        'max': max_price
    }


def generate_insights(data, prediction):
    """Generate market insights"""
    insights = []
    current_year = datetime.now().year
    age = current_year - data['year']
    
    # Age-based insights
    if age <= 3:
        insights.append({
            'icon': 'fas fa-star',
            'text': 'Recent model year - High resale value',
            'type': 'positive'
        })
    elif age > 8:
        insights.append({
            'icon': 'fas fa-calendar-times',
            'text': 'Older vehicle - Consider market demand',
            'type': 'warning'
        })
    
    # Mileage insights
    if data['km_driven'] < 30000:
        insights.append({
            'icon': 'fas fa-road',
            'text': 'Low mileage - Premium pricing',
            'type': 'positive'
        })
    elif data['km_driven'] > 100000:
        insights.append({
            'icon': 'fas fa-tachometer-alt',
            'text': 'High mileage - Price adjustment needed',
            'type': 'warning'
        })
    
    # Owner insights
    if data['owner'] == 0:
        insights.append({
            'icon': 'fas fa-user-check',
            'text': 'First owner - Better market appeal',
            'type': 'positive'
        })
    
    # Transmission insights
    if data['transmission'] == 1:
        insights.append({
            'icon': 'fas fa-cogs',
            'text': 'Automatic transmission - Higher demand',
            'type': 'positive'
        })
    
    # Seller insights
    if data['seller_type'] == 2:
        insights.append({
            'icon': 'fas fa-shield-alt',
            'text': 'Trustmark dealer - Verified quality',
            'type': 'positive'
        })
    
    # Market trend
    market_trends = ['Growing demand', 'Stable market', 'Increasing prices', 'High liquidity']
    insights.append({
        'icon': 'fas fa-chart-line',
        'text': f'Market trend: {market_trends[hash(str(data)) % len(market_trends)]}',
        'type': 'info'
    })

    return insights

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()

    # Run the application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'

    app.run(host='0.0.0.0', port=port, debug=debug)