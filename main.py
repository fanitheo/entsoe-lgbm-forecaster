import pandas as pd
import lightgbm as lgb
from entsoe import EntsoePandasClient
import datetime
import os
import joblib

# --- Ρυθμίσεις ---
API_KEY = "355022e3-373d-4720-a33a-01ac1c423819"
COUNTRY = "NL"
COUNTRY_DE = "DE_LU"
TZ = "Europe/Brussels"

# Διαδρομές αρχείων στο Docker
HISTORICAL_CSV = "/app/data/nl_entsoe_day_ahead_15min_oct2025_to_now (1).csv"
OUTPUT_CSV = "/app/output/realtime_forecast_2days.csv"
MODEL_FILE = "/app/data/trained_lgbm_model.pkl"

def prepare_features(df):
    df_feat = df.copy()
    df_feat['hour'] = df_feat.index.hour
    df_feat['minute'] = df_feat.index.minute
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)
    return df_feat

def smart_impute(df, features, stage_name=""):
    print(f"  -> Έλεγχος δεδομένων {stage_name}...")
    
    missing_counts = df[features].isna().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"     [!] Εντοπίστηκαν {total_missing} κενές τιμές συνολικά:")
        for feat, count in missing_counts.items():
            if count > 0:
                print(f"         - {feat}: {count} κενά (αντιστοιχεί σε {count/4} ώρες)")
    else:
        print("     [OK] Όλες οι τιμές είναι πλήρεις, δεν απαιτείται αναπλήρωση.")

    df_imp = df.copy()
    
    # Βήμα 1: Γραμμική παρεμβολή ΜΟΝΟ για κενά έως 1 ώρα (4 δεκαπεντάλεπτα)
    df_imp[features] = df_imp[features].interpolate(method='linear', limit=4)
    
    # Βήμα 2: Ιστορική αναπλήρωση για μεγαλύτερα κενά (1, 2, 3 ημέρες πίσω)
    for feat in features:
        for days_back in [1, 2, 3]:
            if df_imp[feat].isna().any():
                shift_steps = days_back * 96
                shifted_values = df_imp[feat].shift(shift_steps)
                df_imp[feat] = df_imp[feat].fillna(shifted_values)
                
    return df_imp

def train_model():
    print("[1] Φόρτωση ιστορικών δεδομένων και εκπαίδευση μοντέλου...")
    df = pd.read_csv(HISTORICAL_CSV, index_col='measurementDate')
    df.index = pd.to_datetime(df.index, utc=True)
    
    desired_features = [
        'load_forecast', 'solar_fc', 'wind_off_fc', 'wind_on_fc',
        'solar_fc_de', 'wind_on_fc_de', 'wind_off_fc_de', 'price_de_lu'
    ]
    
    base_features = []
    for col in desired_features:
        if col in df.columns and not df[col].isna().all():
            base_features.append(col)
        else:
            print(f"  [ΠΡΟΕΙΔΟΠΟΙΗΣΗ] Το feature '{col}' λείπει ή είναι κενό στο CSV.")
            
    df = smart_impute(df, base_features, stage_name="(Ιστορικά Εκπαίδευσης)")
    df = prepare_features(df)
    
    all_features = base_features + ['hour', 'minute', 'dayofweek', 'month', 'is_weekend']
    target = 'value'
    
    df_clean = df.dropna(subset=[target] + all_features)
    X = df_clean[all_features]
    y = df_clean[target]
    
    model = lgb.LGBMRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    print("Η εκπαίδευση ολοκληρώθηκε.")
    
    return model, all_features

def fetch_future_data(trained_features):
    print("[2] Άντληση δεδομένων σε πραγματικό χρόνο από ENTSO-E (Ενιαίο Μπλοκ)...")
    client = EntsoePandasClient(api_key=API_KEY)
    
    now = pd.Timestamp.now(tz=TZ)
    start_model = now.floor('15min')
    
    start_api = now.normalize()
    end_api = start_api + pd.Timedelta(days=2) 
    
    df_future = pd.DataFrame(index=pd.date_range(start=start_api, end=end_api, freq="15min", tz=TZ))
    
    try:
        print(f"  -> Άντληση Ολλανδίας (NL)...")
        load_nl = client.query_load_forecast(COUNTRY, start=start_api, end=end_api)
        ws_nl = client.query_wind_and_solar_forecast(COUNTRY, start=start_api, end=end_api)
        
        if isinstance(load_nl, pd.DataFrame): load_nl = load_nl.sum(axis=1)
        df_future['load_forecast'] = load_nl
        
        ws_nl = ws_nl.rename(columns={"Solar": "solar_fc", "Wind Offshore": "wind_off_fc", "Wind Onshore": "wind_on_fc"})
        df_future = df_future.join(ws_nl)
    except Exception as e:
        print(f"     [!] Σφάλμα άντλησης NL: {e}")

    try:
        print(f"  -> Άντληση Γερμανίας (DE_LU)...")
        ws_de = client.query_wind_and_solar_forecast(COUNTRY_DE, start=start_api, end=end_api)
        ws_de = ws_de.rename(columns={"Solar": "solar_fc_de", "Wind Offshore": "wind_off_fc_de", "Wind Onshore": "wind_on_fc_de"})
        df_future = df_future.join(ws_de)
        
        price_de = client.query_day_ahead_prices(COUNTRY_DE, start=start_api, end=end_api).to_frame(name="price_de_lu")
        price_de = price_de.resample("15min").ffill()
        df_future = df_future.join(price_de)
    except Exception as e:
        print(f"     [!] Σφάλμα άντλησης DE_LU: {e}")

    df_future = df_future[~df_future.index.duplicated(keep='first')]
    df_future.index = df_future.index.tz_convert(TZ)
    df_future = df_future[df_future.index >= start_model]
    
    actual_end = df_future.index.max()
    if pd.isna(actual_end) or actual_end < start_model:
        print("  [ΠΡΟΕΙΔΟΠΟΙΗΣΗ] Τα δεδομένα που επέστρεψε το API είναι κενά.")
        exit(1)
        
    full_idx = pd.date_range(start_model, actual_end, freq="15min", tz=TZ)
    df_future = df_future.reindex(full_idx)
    
    time_cols = ['hour', 'minute', 'dayofweek', 'month', 'is_weekend']
    cols_to_check = [col for col in trained_features if col not in time_cols]
    
    for col in cols_to_check:
        if col not in df_future.columns:
            df_future[col] = pd.NA
            
    df_future = smart_impute(df_future, cols_to_check, stage_name="(Πραγματικού Χρόνου)")
    
    df_future.index = df_future.index.tz_convert('UTC')
    return prepare_features(df_future)

def main():
    if os.path.exists(MODEL_FILE):
        print(f"[1] Φόρτωση έτοιμου εκπαιδευμένου μοντέλου από: {MODEL_FILE}")
        model, feature_cols = joblib.load(MODEL_FILE)
    else:
        model, feature_cols = train_model()
        joblib.dump((model, feature_cols), MODEL_FILE)
        print(f"[1.5] Το μοντέλο αποθηκεύτηκε επιτυχώς στο: {MODEL_FILE}")
    
    df_future = fetch_future_data(feature_cols)
    X_future = df_future[feature_cols]
    
    print("[3] Παραγωγή προβλέψεων...")
    predictions = model.predict(X_future)
    
    results_df = pd.DataFrame({
        'timestamp': X_future.index,
        'predicted_price': predictions
    })
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[4] Επιτυχία! Οι προβλέψεις αποθηκεύτηκαν στο: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()