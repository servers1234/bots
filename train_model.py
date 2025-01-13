import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
import pandas_ta as ta
import joblib
import os
import glob
import warnings

warnings.filterwarnings('ignore')

# Dizinleri oluştur
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

DATA_DIR = "data"

def clean_infinite_values(df):
    """Sonsuz değerleri ve aşırı büyük/küçük değerleri temizle"""
    # Sonsuz değerleri NaN ile değiştir
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Her sütun için aşırı değerleri temizle
    for column in df.select_dtypes(include=[np.number]).columns:
        # Sütunun çeyreklik değerlerini hesapla
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Aşırı değerleri belirle ve NaN ile değiştir
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan
    
    return df

def load_and_prepare_data():
    """Tüm sembol verilerini yükle ve birleştir"""
    all_data = []
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_data.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR} directory")
    
    print(f"Found {len(csv_files)} CSV files")
    
    for file in csv_files:
        try:
            symbol = os.path.basename(file).replace("_data.csv", "")
            df = pd.read_csv(file)
            
            # Veri tiplerini düzelt
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Temel özellikleri hesapla
            df['Price_Change'] = df['close'].pct_change()
            df['Volume_Change'] = df['volume'].pct_change()
            df['Daily_Return'] = (df['close'] - df['open']) / df['open']
            
            # Moving averages
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Volatilite
            df['Volatility'] = df['close'].rolling(window=20).std()
            
            # RSI
            df['RSI'] = ta.rsi(df['close'], length=14)
            
            # MACD basitleştirilmiş
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            
            df['symbol'] = symbol
            all_data.append(df)
            print(f"Loaded data for {symbol}")
            
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data could be loaded from CSV files")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
    combined_data = combined_data.sort_values('timestamp')
    
    return combined_data

try:
    print("Loading data...")
    data = load_and_prepare_data()
    
    print("Preparing features...")
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'Price_Change', 'Volume_Change', 'Daily_Return',
        'SMA_20', 'EMA_20', 'Volatility', 'RSI', 'MACD'
    ]
    
    # Özellikleri hazırla ve temizle
    features = data[feature_columns].copy()
    
    # Sonsuz ve aşırı değerleri temizle
    features = clean_infinite_values(features)
    
    # NaN değerleri doldur
    features = features.fillna(method='ffill')  # İleri doğru doldur
    features = features.fillna(method='bfill')  # Geri doğru doldur
    features = features.fillna(0)  # Kalan NaN değerleri 0 ile doldur
    
    # Veri ölçekleme için RobustScaler kullan (aykırı değerlere karşı daha dayanıklı)
    print("Scaling features...")
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Etiketleri oluştur
    labels = (data.groupby('symbol')['close'].shift(-1) > data['close']).astype(int)
    labels = labels[:-1]  # Son satırı kaldır
    scaled_features = scaled_features[:-1]  # Etiketlerle eşleştir
    
    # Eğitim ve test setlerine ayır
    print("Splitting data...")
    train_size = int(len(scaled_features) * 0.8)
    X_train = scaled_features[:train_size]
    X_test = scaled_features[train_size:]
    y_train = labels[:train_size]
    y_test = labels[train_size:]
    
    # Model eğitimi
    print("Training model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Model performansını değerlendir
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Özellik önemliliği
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    # Model ve scaler'ı kaydet
    print("\nSaving model and scaler...")
    joblib.dump(model, 'models/ml_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Model and scaler successfully saved.")

except Exception as e:
    print(f"Error during training: {str(e)}")
    raise