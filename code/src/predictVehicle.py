import os
import pickle
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataCleanup import make_map

def predict_vehicle_price(make: str, model: str, vehicle_age: float, odometer: float, condition: float, state: str, output_dir: str = "../../output"):
    """
    Predict the selling price of a vehicle using the trained OLS model.
    
    Parameters:
    - make (str): The make of the vehicle
    - model (str): The model of the vehicle
    - vehicle_age (float): The age of the vehicle in years
    - odometer (float): The odometer reading of the vehicle (miles)
    - condition (float): The condition score of the vehicle (between 1 and 50)
    - state (str): The state abbreviation where the vehicle is sold (i.e. "CA", "TX", etc.)
    - output_dir (str): Directory containing the serialized model and metadata
    
    Returns:
    - dict: A dictionary containing the predicted log price, median price, and mean (expected) price.
    """
    # 1. Load model and metadata
    model_path = os.path.join(output_dir, 'car_price_model.pkl')
    metadata_path = os.path.join(output_dir, 'model_metadata.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Model or metadata not found in {output_dir}. Please run trendAnalysis.py first to train and save the model."
        )
        
    fit_log = sm.load(model_path)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        
    common_models = metadata['common_models']
    train_columns = metadata['train_columns']
    residual_variance = metadata['residual_variance']
    
    # 2. Preprocess input parameters
    # Canonicalize make (lowercase, strip, replace aliases)
    clean_make = str(make).strip().lower()
    clean_make = make_map.get(clean_make, clean_make)
    
    # Perform case-insensitive lookup against common_models
    # common_models is a set of (make_lowercase, model_original_case)
    model_lookup = { (m_make, m_model.lower()): m_model for m_make, m_model in common_models }
    clean_model_lower = str(model).strip().lower()
    
    # Map to canonical model name if it's common, otherwise collapse to make_Other
    if (clean_make, clean_model_lower) in model_lookup:
        collapsed_model = model_lookup[(clean_make, clean_model_lower)]
    else:
        collapsed_model = f"{clean_make}_Other"
        
    # Scale odometer to 10k miles units
    odometer_10k = odometer / 10000.0
    
    # Canonicalize state (lowercase, strip)
    clean_state = str(state).strip().lower()
    
    # 3. Construct the single-row DataFrame for encoding
    input_df = pd.DataFrame([{
        'vehicle_age': float(vehicle_age),
        'odometer_10k': float(odometer_10k),
        'condition': float(condition),
        'model': collapsed_model,
        'state': clean_state
    }])
    
    # One-hot encode the categorical columns
    input_encoded = pd.get_dummies(input_df, columns=['model', 'state'])
    
    # 4. Align with training columns (fill missing columns with 0, drop extra columns)
    X_new = input_encoded.reindex(columns=train_columns, fill_value=0)
    
    # Set the statsmodels constant term to 1.0
    if 'const' in X_new.columns:
        X_new['const'] = 1.0
        
    # Ensure columns are floats
    X_new = X_new.astype(float)
    
    # 5. Predict log selling price
    predicted_log_price = fit_log.predict(X_new)[0]
    
    # 6. Revert log transformation
    # Naive exponentiation yields the median prediction (since E[log(Y)] matches the median for log-normal)
    predicted_median_price = np.exp(predicted_log_price)
    
    # Correcting for log-normal expectation bias yields the mean (expected value) prediction:
    # E[Y] = exp(mu + sigma^2 / 2)
    predicted_mean_price = np.exp(predicted_log_price + (residual_variance / 2.0))
    
    return {
        "raw_inputs": {
            "make": make,
            "model": model,
            "vehicle_age": vehicle_age,
            "odometer": odometer,
            "condition": condition,
            "state": state
        },
        "processed_inputs": {
            "canonical_make": clean_make,
            "collapsed_model": collapsed_model,
            "odometer_10k": odometer_10k,
            "clean_state": clean_state
        },
        "predicted_log_price": predicted_log_price,
        "predicted_median_price": predicted_median_price,
        "predicted_expected_price": predicted_mean_price
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict vehicle price using the trained OLS model.")
    parser.add_argument("--make", type=str, required=True, help="Make of the vehicle (e.g. Ford)")
    parser.add_argument("--model", type=str, required=True, help="Model of the vehicle (e.g. Focus)")
    parser.add_argument("--age", type=float, required=True, help="Age of the vehicle in years")
    parser.add_argument("--odometer", type=float, required=True, help="Odometer reading in miles")
    parser.add_argument("--condition", type=float, required=True, help="Condition rating (1.0 - 5.0)")
    parser.add_argument("--state", type=str, required=True, help="State abbreviation where the vehicle is sold (e.g. CA)")
    parser.add_argument("--dir", type=str, default="../../output", help="Directory where model artifacts are saved")
    
    args = parser.parse_args()
    
    try:
        results = predict_vehicle_price(
            make=args.make,
            model=args.model,
            vehicle_age=args.age,
            odometer=args.odometer,
            condition=args.condition,
            state=args.state,
            output_dir=args.dir
        )
        
        print("\n=== Vehicle Valuation Prediction ===")
        print(f"Input Vehicle: {results['raw_inputs']['make'].title()} {results['raw_inputs']['model'].title()}")
        print(f"State: {results['raw_inputs']['state'].upper()} | Age: {results['raw_inputs']['vehicle_age']} years | Odometer: {results['raw_inputs']['odometer']:,} miles | Condition: {results['raw_inputs']['condition']}")
        print(f"Canonical Model Representation: {results['processed_inputs']['collapsed_model']}")
        print("-" * 36)
        print(f"Predicted Log Selling Price:     {results['predicted_log_price']:.4f}")
        print(f"Predicted Median Value (naive):  ${results['predicted_median_price']:,.2f}")
        print(f"Predicted Expected Value (mean): ${results['predicted_expected_price']:,.2f}")
        print(f"Predicted for sale in: {results['processed_inputs']['clean_state'].upper()}")
        print(f"Trade-in offer for 10-15% profit: ${results['predicted_expected_price'] * 0.85:,.2f} to ${results['predicted_expected_price'] * 0.90:,.2f}")
        print("====================================\n")
        
    except Exception as e:
        print(f"Error: {e}")
