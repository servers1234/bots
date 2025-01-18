import asyncio
import json
import logging
from datetime import datetime, time
import time as time_module
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta
from binance.um_futures import UMFutures
from binance.error import ClientError
from telegram import Bot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Sonsuz ve aşırı değerleri temizle"""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df



class BinanceFuturesBot:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.client = UMFutures(
            key=self.config['api_key'],
            secret=self.config['api_secret']
        )
        self.telegram = Bot(token=self.config['telegram_token'])
        self.chat_id = self.config['telegram_chat_id']
        self.positions = {}
        self.last_api_call = 0
        self.rate_limit_delay = 0.1
        self.model = self._load_ml_model()
        self.scaler = self._load_scaler()
        self.daily_trades = 0
        self.daily_stats = {
            'trades': 0,
            'profit': 0.0,
            'losses': 0.0
        }
        self.last_daily_reset = datetime.now().date()

    def _load_config(self, config_path: str) -> dict:
        """Config dosyasını yükle"""
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
            self._validate_config(config)
            return config
        except Exception as e:
            logging.error(f"Config yükleme hatası: {e}")
            raise

    def _validate_config(self, config: dict) -> None:
        """Config dosyasını doğrula"""
        required_fields = [
            'api_key', 'api_secret', 'telegram_token', 'telegram_chat_id',
            'symbols', 'risk_management', 'trading_hours', 'timeframes',
            'ml_model_path', 'scaler_path'
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Eksik config alanı: {field}")

    def _load_ml_model(self) -> GradientBoostingClassifier:
        """Makine öğrenimi modelini yükle"""
        try:
            model = joblib.load(self.config['ml_model_path'])
            return model
        except Exception as e:
            logging.error(f"ML model yükleme hatası: {e}")
            raise

    def _load_scaler(self) -> StandardScaler:
        """Ölçekleyiciyi yükle"""
        try:
            scaler = joblib.load(self.config['scaler_path'])
            return scaler
        except Exception as e:
            logging.error(f"Scaler yükleme hatası: {e}")
            raise

    async def send_telegram(self, message: str) -> None:
        """Telegram mesajı gönder"""
        if self.config['notifications']['trade_updates']:
            try:
                await self.telegram.send_message(
                    chat_id=self.chat_id,
                    text=message
                )
            except Exception as e:
                logging.error(f"Telegram mesaj hatası: {e}")

    def validate_symbol(self, symbol: str) -> bool:
        """Validate the symbol using Binance API"""
        try:
            exchange_info = self.client.exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols']]
            return symbol in symbols
        except Exception as e:
            logging.error(f"Symbol validation error: {e}")
            return False

    def get_klines(self, symbol: str) -> pd.DataFrame:
        """Mum verilerini al"""
        if not self.validate_symbol(symbol):
            logging.error(f"Invalid symbol: {symbol}")
            return pd.DataFrame()

        try:
            timeframe = self.config['timeframes']['default']
            klines = self.client.klines(
                symbol=symbol,
                interval=timeframe,
                limit=100
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df

        except Exception as e:
            logging.error(f"Kline veri alma hatası: {e}")
            return pd.DataFrame()

    def calculate_bearish_engulfing(self, df: pd.DataFrame) -> pd.Series:
            """Yutan Ayı (Bearish Engulfing) formasyonunu hesapla"""
            try:
                prev_body = df['close'].shift(1) - df['open'].shift(1)
                curr_body = df['close'] - df['open']

                return ((prev_body > 0) & 
                        (curr_body < 0) & 
                        (df['open'] > df['close'].shift(1)) & 
                        (df['close'] < df['open'].shift(1))).astype(int)
            except Exception as e:
                logging.error(f"Bearish Engulfing hesaplama hatası: {str(e)}")
                return pd.Series(0, index=df.index)
    def calculate_morning_star(self, df: pd.DataFrame) -> pd.Series:
           """Sabah Yıldızı (Morning Star) formasyonunu hesapla"""
           try:
               result = pd.Series(0, index=df.index)
               for i in range(2, len(df)):
                   if (df['close'].iloc[i - 2] < df['open'].iloc[i - 2] and
                       abs(df['close'].iloc[i - 1] - df['open'].iloc[i - 1]) < (df['high'].iloc[i - 1] - df['low'].iloc[i - 1]) * 0.1 and
                       df['close'].iloc[i] > df['open'].iloc[i] and
                       df['close'].iloc[i] > df['open'].iloc[i - 2]):
                       result.iloc[i] = 1
               return result
           except Exception as e:
               logging.error(f"Morning Star hesaplama hatası: {str(e)}")
               return pd.Series(0, index=df.index)
    def calculate_evening_star(self, df: pd.DataFrame) -> pd.Series:
        """Akşam Yıldızı (Evening Star) formasyonunu hesapla"""
        try:
            result = pd.Series(0, index=df.index)
            for i in range(2, len(df)):
                if (df['close'].iloc[i - 2] > df['open'].iloc[i - 2] and
                    abs(df['close'].iloc[i - 1] - df['open'].iloc[i - 1]) < (df['high'].iloc[i - 1] - df['low'].iloc[i - 1]) * 0.1 and
                    df['close'].iloc[i] < df['open'].iloc[i] and
                    df['close'].iloc[i] < df['open'].iloc[i - 2]):
                    result.iloc[i] = 1
            return result
        except Exception as e:
            logging.error(f"Evening Star hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)
    def calculate_three_white_soldiers(self, df: pd.DataFrame) -> pd.Series:
        """Üç Beyaz Asker (Three White Soldiers) formasyonunu hesapla"""
        try:
            result = pd.Series(0, index=df.index)
            for i in range(2, len(df)):
                if (df['close'].iloc[i] > df['open'].iloc[i] and
                    df['close'].iloc[i - 1] > df['open'].iloc[i - 1] and
                    df['close'].iloc[i - 2] > df['open'].iloc[i - 2] and
                    df['close'].iloc[i] > df['close'].iloc[i - 1] > df['close'].iloc[i - 2]):
                    result.iloc[i] = 1
            return result
        except Exception as e:
            logging.error(f"Three White Soldiers hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)
    def calculate_three_black_crows(self, df: pd.DataFrame) -> pd.Series:
        """Üç Siyah Karga (Three Black Crows) formasyonunu hesapla"""
        try:
            result = pd.Series(0, index=df.index)
            for i in range(2, len(df)):
                if (df['close'].iloc[i] < df['open'].iloc[i] and
                    df['close'].iloc[i - 1] < df['open'].iloc[i - 1] and
                    df['close'].iloc[i - 2] < df['open'].iloc[i - 2] and
                    df['close'].iloc[i] < df['close'].iloc[i - 1] < df['close'].iloc[i - 2]):
                    result.iloc[i] = 1
            return result
        except Exception as e:
            logging.error(f"Three Black Crows hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)



    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Temel ve gelişmiş teknik indikatörleri hesapla"""
        try:
            logging.info("Calculating technical indicators...")

            # Gerekli sütunları kontrol et
            required_columns = ['high', 'low', 'close', 'open']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                logging.error(f"Missing required columns: {missing}")
                return df

            # Minimum veri uzunluğu kontrolü
            if len(df) < 52:  # Ichimoku için minimum 52 periyot gerekli
                logging.warning("Not enough data for calculations")
                return df

            # ---- TEMEL İNDİKATÖRLER ----
            # RSI hesaplama
            df['RSI'] = ta.rsi(df['close'], length=14)

            # MACD hesaplama
            macd_data = ta.macd(df['close'])
            df['MACD'] = macd_data['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd_data['MACDs_12_26_9']
            df['MACD_HIST'] = macd_data['MACDh_12_26_9']

            # Bollinger Bands hesaplama
            bollinger = ta.bbands(df['close'], length=20, std=2)
            df['BB_UPPER'] = bollinger['BBU_20_2.0']
            df['BB_MIDDLE'] = bollinger['BBM_20_2.0']
            df['BB_LOWER'] = bollinger['BBL_20_2.0']

            # Moving Averages
            df['SMA_20'] = ta.sma(df['close'], length=20)
            df['EMA_20'] = ta.ema(df['close'], length=20)
            df['EMA_50'] = ta.ema(df['close'], length=50)
            df['EMA_200'] = ta.ema(df['close'], length=200)

            # StochRSI hesaplama
            stochrsi = ta.stochrsi(df['close'], length=14)
            df['StochRSI_K'] = stochrsi['STOCHRSIk_14_14_3_3']
            df['StochRSI_D'] = stochrsi['STOCHRSId_14_14_3_3']
            df['StochRSI'] = df['StochRSI_K']

            # ---- GELİŞMİŞ İNDİKATÖRLER ----
            # ADX (Average Directional Index)
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            df['ADX'] = adx['ADX_14']
            df['DI_plus'] = adx['DMP_14']
            df['DI_minus'] = adx['DMN_14']

            # ATR (Average True Range)
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            # momentum (Average True Range)
            df = self.calculate_momentum_indicators(df)
            # Parabolic SAR
            psar = ta.psar(df['high'], df['low'], df['close'])
            df['SAR'] = psar['PSARl_0.02_0.2']  # Use appropriate key for PSAR

            # ---- MUM FORMASYONLARI ----
            df['DOJI'] = self.calculate_doji(df)
            df['HAMMER'] = self.calculate_hammer(df)
            df['BULLISH_ENGULFING'] = self.calculate_bullish_engulfing(df)
            df['BEARISH_ENGULFING'] = self.calculate_bearish_engulfing(df)
            df['MORNING_STAR'] = self.calculate_morning_star(df)
            df['EVENING_STAR'] = self.calculate_evening_star(df)
            df['THREE_WHITE_SOLDIERS'] = self.calculate_three_white_soldiers(df)
            df['THREE_BLACK_CROWS'] = self.calculate_three_black_crows(df)

            # NaN değerleri temizle
            df = df.ffill().bfill()

            # Hesaplanan göstergeleri kontrol et
            required_indicators = [
                'RSI', 'MACD', 'MACD_SIGNAL', 'BB_UPPER', 'BB_LOWER',
                'SMA_20', 'EMA_20', 'EMA_50', 'EMA_200', 'StochRSI_K', 'StochRSI_D',
                'ADX', 'ATR', 'SAR'
            ]

            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]

            if missing_indicators:
                logging.warning(f"Missing indicators after calculation: {missing_indicators}")
            else:
                logging.info("All required indicators calculated successfully")

            return df

        except Exception as e:
            logging.error(f"İndikatör hesaplama hatası: {str(e)}", exc_info=True)
            return df
    def calculate_doji(self, df: pd.DataFrame) -> pd.Series:
        """Doji mum formasyonunu hesapla"""
        try:
            body = abs(df['close'] - df['open'])
            wick = df['high'] - df['low']
            return (body <= (wick * 0.1)).astype(int)
        except Exception as e:
            logging.error(f"Doji hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum göstergelerini hesapla"""
        try:
            logging.info("Momentum göstergeleri hesaplanıyor...")
    
            # Manuel MFI hesaplama
            def calculate_mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
                try:
                    # Önce tüm değerleri float64'e çevir
                    high = df['high'].astype('float64')
                    low = df['low'].astype('float64')
                    close = df['close'].astype('float64')
                    volume = df['volume'].astype('float64')
    
                    # Typical price hesapla
                    typical_price = (high + low + close) / 3.0
                    
                    # Money flow hesapla
                    raw_money_flow = typical_price * volume
                    
                    # Money flow'u pandas Series'e çevir
                    money_flow = pd.Series(raw_money_flow, index=df.index, dtype='float64')
                    
                    # Fiyat değişimi hesapla
                    price_diff = typical_price.diff()
    
                    # Pozitif ve negatif flow'ları başlangıçta sıfır olarak ayarla
                    positive_flow = pd.Series(0.0, index=df.index, dtype='float64')
                    negative_flow = pd.Series(0.0, index=df.index, dtype='float64')
    
                    # İndeks bazlı atama yerine loc kullan
                    positive_flow.loc[price_diff > 0] = money_flow[price_diff > 0]
                    negative_flow.loc[price_diff < 0] = money_flow[price_diff < 0]
    
                    # Hareketli ortalamalar
                    positive_mf = positive_flow.rolling(window=length, min_periods=1).sum()
                    negative_mf = negative_flow.rolling(window=length, min_periods=1).sum()
    
                    # Sıfıra bölünmeyi önle
                    mfi = np.where(
                        negative_mf != 0,
                        100 - (100 / (1 + (positive_mf / negative_mf))),
                        100
                    )
    
                    return pd.Series(mfi, index=df.index).fillna(50)
    
                except Exception as e:
                    logging.error(f"Manuel MFI hesaplama hatası: {e}")
                    return pd.Series(50, index=df.index, dtype='float64')
    
            # Veri tiplerini başlangıçta düzelt
            for col in ['high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    
            # MFI hesapla
            df['MFI'] = calculate_mfi(df)
    
            # Diğer göstergeler...
            indicators = {
                'CMF': lambda: ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20),
                'ADL': lambda: ta.ad(df['high'], df['low'], df['close'], df['volume']),
                'OBV': lambda: ta.obv(df['close'], df['volume']),
                'MOM': lambda: ta.mom(df['close'], length=14),
                'ROC': lambda: ta.roc(df['close'], length=9)
            }
    
            for indicator_name, calc_func in indicators.items():
                try:
                    df[indicator_name] = calc_func().astype('float64')
                except Exception as e:
                    logging.error(f"{indicator_name} hesaplama hatası: {e}")
                    df[indicator_name] = 0.0
    
            # OBV EMA hesapla
            try:
                df['OBV_EMA'] = ta.ema(df['OBV'], length=20).astype('float64')
            except Exception as e:
                logging.error(f"OBV_EMA hesaplama hatası: {e}")
                df['OBV_EMA'] = 0.0
    
            # TSI hesapla
            try:
                close_diff = df['close'].diff().astype('float64')
                double_smoothed = ta.ema(ta.ema(close_diff, length=25), length=13)
                double_smoothed_abs = ta.ema(ta.ema(close_diff.abs(), length=25), length=13)
                
                df['TSI'] = np.where(
                    double_smoothed_abs != 0,
                    100 * (double_smoothed / double_smoothed_abs),
                    0
                ).astype('float64')
            except Exception as e:
                logging.error(f"TSI hesaplama hatası: {e}")
                df['TSI'] = 0.0
    
            # NaN değerleri temizle
            for col in df.columns:
                if df[col].dtype.kind in 'fc':  # float veya complex tipler için
                    df[col] = df[col].ffill().bfill().fillna(0).astype('float64')
    
            logging.info("Momentum göstergeleri başarıyla hesaplandı")
            return df
    
        except Exception as e:
            logging.error(f"Momentum göstergeleri hesaplama hatası: {e}")
            default_columns = ['MFI', 'CMF', 'ADL', 'OBV', 'OBV_EMA', 'TSI', 'MOM', 'ROC']
            for col in default_columns:
                df[col] = 0.0
            return df
    def calculate_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Çekiç formasyonunu hesapla"""
        try:
            body = abs(df['close'] - df['open'])
            lower_wick = df[['open', 'close']].min(axis=1) - df['low']
            upper_wick = df['high'] - df[['open', 'close']].max(axis=1)

            return ((lower_wick > (body * 2)) & (upper_wick <= (body * 0.1))).astype(int)
        except Exception as e:
            logging.error(f"Hammer hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)
        

    def analyze_trend_strength(self, df: pd.DataFrame) -> dict:
        """Trend gücünü analiz et"""
        try:
            logging.info("Trend gücü analizi yapılıyor...")

            last_row = df.iloc[-1]
            last_rows = df.tail(2)

            trend_strength = {
                'trend': 'NEUTRAL',
                'strength': 0,
                'confidence': 0,
                'momentum_confirmation': False,
                'volume_confirmation': False
            }

            # EMA Trend Analizi
            ema_trend = (last_row['EMA_20'] > last_row['EMA_50']) and (last_row['EMA_50'] > last_row['EMA_200'])
            ema_trend_reverse = (last_row['EMA_20'] < last_row['EMA_50']) and (last_row['EMA_50'] < last_row['EMA_200'])

            # ADX Trend Gücü
            adx_strength = last_row['ADX'] > 25

            # Momentum Göstergeleri Kontrolü
            momentum_signals = [
                last_row['RSI'] > 50,  # RSI trend
                last_row['MACD'] > last_row['MACD_SIGNAL'],  # MACD trend
                last_row['MFI'] > 50,  # MFI trend
                last_row['ROC'] > 0,  # ROC trend
                last_row['TSI'] > 0,  # TSI trend
                last_row['OBV'] > last_row['OBV_EMA']  # OBV trend
            ]

            # Volume Trend
            volume_trend = (
                last_row['volume'] > df['volume'].rolling(window=20).mean().iloc[-1] and
                last_row['CMF'] > 0
            )

            # Toplam bullish sinyalleri hesapla
            bullish_signals = sum([
                ema_trend,
                adx_strength,
                *momentum_signals,
                volume_trend
            ])

            total_signals = 8  # Toplam sinyal sayısı

            # Trend yönü ve gücünü belirle
            if ema_trend and bullish_signals >= 6:
                trend_strength['trend'] = 'STRONG_BULLISH'
                trend_strength['strength'] = 0.9
            elif ema_trend and bullish_signals >= 5:
                trend_strength['trend'] = 'BULLISH'
                trend_strength['strength'] = 0.7
            elif ema_trend_reverse and bullish_signals <= 2:
                trend_strength['trend'] = 'STRONG_BEARISH'
                trend_strength['strength'] = 0.9
            elif ema_trend_reverse and bullish_signals <= 3:
                trend_strength['trend'] = 'BEARISH'
                trend_strength['strength'] = 0.7
            else:
                trend_strength['trend'] = 'NEUTRAL'
                trend_strength['strength'] = 0.5

            # Güven seviyesi hesaplama
            trend_strength['confidence'] = bullish_signals / total_signals
            trend_strength['momentum_confirmation'] = sum(momentum_signals) / len(momentum_signals) > 0.6
            trend_strength['volume_confirmation'] = volume_trend

            logging.info(f"Trend analizi sonucu: {trend_strength}")
            return trend_strength

        except Exception as e:
            logging.error(f"Trend gücü analizi hatası: {e}")
            return {
                'trend': 'NEUTRAL',
                'strength': 0,
                'confidence': 0,
                'momentum_confirmation': False,
                'volume_confirmation': False
            }

    def check_volatility_conditions(self, df: pd.DataFrame) -> dict:
        """Volatilite koşullarını kontrol et"""
        try:
            logging.info("Volatilite analizi yapılıyor...")

            result = {
                'is_valid': False,
                'atr_ratio': 0,
                'bb_width_ratio': 0,
                'volatility_state': 'HIGH'
            }

            # ATR bazlı volatilite kontrolü
            current_atr = df['ATR'].iloc[-1]
            avg_atr = df['ATR'].rolling(window=20).mean().iloc[-1]
            atr_ratio = current_atr / avg_atr

            # Bollinger Bands genişliği
            bb_width = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
            current_bb_width = bb_width.iloc[-1]
            avg_bb_width = bb_width.rolling(window=20).mean().iloc[-1]
            bb_width_ratio = current_bb_width / avg_bb_width

            # Sonuçları kaydet
            result['atr_ratio'] = atr_ratio
            result['bb_width_ratio'] = bb_width_ratio

            # Volatilite durumunu belirle
            if 0.8 <= atr_ratio <= 1.5 and 0.7 <= bb_width_ratio <= 1.3:
                result['is_valid'] = True
                result['volatility_state'] = 'NORMAL'
            elif atr_ratio < 0.8 and bb_width_ratio < 0.7:
                result['volatility_state'] = 'LOW'
            else:
                result['volatility_state'] = 'HIGH'

            logging.info(f"Volatilite analizi sonucu: {result}")
            return result

        except Exception as e:
            logging.error(f"Volatilite kontrolü hatası: {e}")
            return {
                'is_valid': False,
                'atr_ratio': 0,
                'bb_width_ratio': 0,
                'volatility_state': 'ERROR'
            }
    def calculate_bullish_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Yutan boğa formasyonunu hesapla"""
        try:
            prev_body = df['close'].shift(1) - df['open'].shift(1)
            curr_body = df['close'] - df['open']

            return ((prev_body < 0) & 
                    (curr_body > 0) & 
                    (df['open'] < df['close'].shift(1)) & 
                    (df['close'] > df['open'].shift(1))).astype(int)
        except Exception as e:
            logging.error(f"Bullish Engulfing hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """VWAP (Volume Weighted Average Price) hesapla"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        except Exception as e:
            logging.error(f"VWAP hesaplama hatası: {str(e)}")
            return pd.Series(0, index=df.index)

    def verify_indicators(self, df: pd.DataFrame) -> None:
        """İndikatörlerin varlığını ve geçerliliğini kontrol et"""
        required_indicators = ['ICHIMOKU_CONVERSION', 'ICHIMOKU_BASE']

        for indicator in required_indicators:
            if indicator not in df.columns:
                logging.error(f"Missing indicator: {indicator}")
            elif df[indicator].isnull().any():
                logging.warning(f"NaN values found in {indicator}")
            else:
                logging.info(f"{indicator} calculated successfully")

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """İleri seviye indikatörleri hesapla"""
        try:
            if df.empty:
                logging.error("DataFrame is empty. Cannot calculate advanced indicators.")
                return df

            # Ichimoku hesaplaması
            try:
                ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])

                # Debug için mevcut sütunları logla
                if isinstance(ichimoku, pd.DataFrame):
                    logging.debug(f"Available Ichimoku columns: {ichimoku.columns.tolist()}")

                    # Güncel pandas_ta sütun isimleri
                    column_mapping = {
                        'TENKAN_9': 'ICHIMOKU_CONVERSION',
                        'KIJUN_26': 'ICHIMOKU_BASE',
                        'SENKOU_A_26': 'ICHIMOKU_SPAN_A',
                        'SENKOU_B_52': 'ICHIMOKU_SPAN_B',
                        'CHIKOU_26': 'ICHIMOKU_CHIKOU'
                    }

                    # Eğer yeni sütun isimleri çalışmazsa manuel hesaplama yap
                    if not any(col in ichimoku.columns for col in column_mapping.keys()):
                        # Manuel Ichimoku hesaplama
                        period9_high = df['high'].rolling(window=9).max()
                        period9_low = df['low'].rolling(window=9).min()
                        df['ICHIMOKU_CONVERSION'] = (period9_high + period9_low) / 2

                        period26_high = df['high'].rolling(window=26).max()
                        period26_low = df['low'].rolling(window=26).min()
                        df['ICHIMOKU_BASE'] = (period26_high + period26_low) / 2

                        period52_high = df['high'].rolling(window=52).max()
                        period52_low = df['low'].rolling(window=52).min()
                        df['ICHIMOKU_SPAN_B'] = (period52_high + period52_low) / 2

                        df['ICHIMOKU_SPAN_A'] = (df['ICHIMOKU_CONVERSION'] + df['ICHIMOKU_BASE']) / 2
                        df['ICHIMOKU_CHIKOU'] = df['close'].shift(-26)
                    else:
                        # pandas_ta sütunlarını eşle
                        for old_col, new_col in column_mapping.items():
                            if old_col in ichimoku.columns:
                                df[new_col] = ichimoku[old_col]

                    # Kontrol et
                    required_cols = ['ICHIMOKU_CONVERSION', 'ICHIMOKU_BASE']
                    if all(col in df.columns for col in required_cols):
                        logging.info("Ichimoku indicators calculated successfully")
                    else:
                        logging.warning("Some Ichimoku indicators are missing")

                else:
                    # Eğer DataFrame dönmezse manuel hesapla
                    period9_high = df['high'].rolling(window=9).max()
                    period9_low = df['low'].rolling(window=9).min()
                    df['ICHIMOKU_CONVERSION'] = (period9_high + period9_low) / 2

                    period26_high = df['high'].rolling(window=26).max()
                    period26_low = df['low'].rolling(window=26).min()
                    df['ICHIMOKU_BASE'] = (period26_high + period26_low) / 2

                    logging.info("Ichimoku indicators calculated manually")

            except Exception as ichimoku_error:
                logging.error(f"Ichimoku calculation error: {ichimoku_error}")
                # Hata durumunda manuel hesaplama
                period9_high = df['high'].rolling(window=9).max()
                period9_low = df['low'].rolling(window=9).min()
                df['ICHIMOKU_CONVERSION'] = (period9_high + period9_low) / 2

                period26_high = df['high'].rolling(window=26).max()
                period26_low = df['low'].rolling(window=26).min()
                df['ICHIMOKU_BASE'] = (period26_high + period26_low) / 2

            # ADX hesaplaması (mevcut kod)
            try:
                adx = ta.adx(df['high'], df['low'], df['close'])
                if isinstance(adx, pd.DataFrame):
                    if 'ADX_14' in adx.columns:
                        df['ADX'] = adx['ADX_14']
                    elif 'ADX' in adx.columns:
                        df['ADX'] = adx['ADX']
                    logging.info("ADX calculated successfully")

            except Exception as adx_error:
                logging.error(f"ADX calculation error: {adx_error}")

            # NaN değerleri temizle
            df = df.ffill().bfill()

            return df

        except Exception as e:
            logging.error(f"İleri seviye indikatör hesaplama hatası: {str(e)}")
            self.verify_indicators(df)
            return df
            
    def _calculate_atr(self, symbol: str) -> float:
        """ATR hesapla"""
        try:
            df = self.get_klines(symbol)
            atr = ta.atr(df['high'], df['low'], df['close'], length=14)
            return atr.iloc[-1]
        except Exception as e:
            logging.error(f"ATR hesaplama hatası: {e}")
            return 0.0

    def _calculate_dynamic_stop_loss(self, price: float, atr: float, trade_type: str, symbol: str) -> tuple:
        """
        İyileştirilmiş dinamik stop loss ve take profit hesaplama
        - ATR bazlı dinamik hesaplama
        - Volatiliteye göre ayarlanan çarpanlar  
        - Trend yönüne göre optimizasyon
        """
        try:
            # Volatilite bazlı çarpan hesaplama
            volatility = self._calculate_volatility(symbol)
            base_multiplier = self.config['risk_management'].get('base_atr_multiplier', 2.0)

            # Volatiliteye göre çarpanı ayarla
            if volatility > 0.05:  # Yüksek volatilite
                sl_multiplier = base_multiplier * 1.5
                tp_multiplier = base_multiplier * 2
            else:  # Normal volatilite
                sl_multiplier = base_multiplier
                tp_multiplier = base_multiplier * 1.5

            # Trend analizi
            trend = self._analyze_trend(symbol)

            if trade_type == 'BUY':
                # Trend yönüne göre stop loss ve take profit ayarla
                if trend == 'BULLISH':
                    sl_price = price - (atr * sl_multiplier * 0.8)  # Daha yakın SL
                    tp_price = price + (atr * tp_multiplier * 1.2)  # Daha uzak TP
                elif trend == 'BEARISH':
                    sl_price = price - (atr * sl_multiplier * 1.2)  # Daha uzak SL
                    tp_price = price + (atr * tp_multiplier * 0.8)  # Daha yakın TP
                else:  # SIDEWAYS
                    sl_price = price - (atr * sl_multiplier)
                    tp_price = price + (atr * tp_multiplier)

            elif trade_type == 'SELL':
                if trend == 'BEARISH':
                    sl_price = price + (atr * sl_multiplier * 0.8)
                    tp_price = price - (atr * tp_multiplier * 1.2)
                elif trend == 'BULLISH':
                    sl_price = price + (atr * sl_multiplier * 1.2)
                    tp_price = price - (atr * tp_multiplier * 0.8)
                else:
                    sl_price = price + (atr * sl_multiplier)
                    tp_price = price - (atr * tp_multiplier)

            # Risk/Ödül oranı kontrolü
            rr_ratio = abs(tp_price - price) / abs(sl_price - price)
            min_rr_ratio = self.config['risk_management'].get('min_risk_reward_ratio', 1.5)

            if rr_ratio < min_rr_ratio:
                logging.warning(f"Risk/Ödül oranı çok düşük: {rr_ratio:.2f}")
                return None, None

            return sl_price, tp_price

        except Exception as e:
            logging.error(f"Stop loss/Take profit hesaplama hatası: {e}")
            return None, None


    def _calculate_volatility(self, symbol: str) -> float:
        """
        Sembol için volatilite hesapla
        20 günlük standart sapma kullanılıyor
        """
        try:
            df = self.get_klines(symbol)
            if not df.empty:
                returns = df['close'].pct_change()
                volatility = returns.std() * (252 ** 0.5)  # Yıllık volatilite
                return volatility
            return 0
        except Exception as e:
            logging.error(f"Volatilite hesaplama hatası: {e}")
            return 0

    def _analyze_trend(self, symbol: str) -> str:
        """
        Trend analizi yap
        EMA50 ve EMA200 kullanılıyor
        """
        try:
            df = self.get_klines(symbol)
            if not df.empty:
                df['EMA50'] = ta.ema(df['close'], length=50)
                df['EMA200'] = ta.ema(df['close'], length=200)

                last_row = df.iloc[-1]

                if last_row['EMA50'] > last_row['EMA200'] * 1.02:
                    return 'BULLISH'
                elif last_row['EMA50'] < last_row['EMA200'] * 0.98:
                    return 'BEARISH'
                else:
                    return 'SIDEWAYS'
            return 'SIDEWAYS'
        except Exception as e:
            logging.error(f"Trend analizi hatası: {e}")
            return 'SIDEWAYS'

    def _calculate_dynamic_take_profit(self, price: float, atr: float, trade_type: str, multiplier: float) -> float:
        """Dinamik take profit hesapla"""
        if trade_type == 'BUY':
            return price + (atr * multiplier)
        elif trade_type == 'SELL':
            return price - (atr * multiplier)

    async def _place_orders(self, symbol: str, trade_type: str, position_size: float, stop_loss: float, take_profit: float):
        """Order'ları yerleştir"""
        try:
            if trade_type == 'BUY':
                order = self.client.new_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=position_size
                )
            elif trade_type == 'SELL':
                order = self.client.new_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=position_size
                )
            # Add stop loss and take profit orders
            sl_order = self.client.new_order(
                symbol=symbol,
                side='SELL' if trade_type == 'BUY' else 'BUY',
                type='STOP_MARKET',
                stopPrice=stop_loss,
                quantity=position_size
            )
            tp_order = self.client.new_order(
                symbol=symbol,
                side='SELL' if trade_type == 'BUY' else 'BUY',
                type='TAKE_PROFIT_MARKET',
                stopPrice=take_profit,
                quantity=position_size
            )
            return order
        except Exception as e:
            logging.error(f"Order yerleştirme hatası: {e}")
            return None

    
    def rsi_strategy(self, df: pd.DataFrame) -> str:
        """RSI Stratejisi"""
        if df['RSI'].iloc[-1] < 30:
            return "BUY"
        elif df['RSI'].iloc[-1] > 70:
            return "SELL"
        return "HOLD"

    def ema_strategy(self, df: pd.DataFrame) -> str:
        """EMA Kesişim Stratejisi"""
        if df['EMA_20'].iloc[-1] > df['SMA_20'].iloc[-1]:
            return "BUY"
        elif df['EMA_20'].iloc[-1] < df['SMA_20'].iloc[-1]:
            return "SELL"
        return "HOLD"

    def bollinger_strategy(self, df: pd.DataFrame) -> str:
        """Bollinger Bands Stratejisi"""
        if df['close'].iloc[-1] < df['BB_LOWER'].iloc[-1]:
            return "BUY"
        elif df['close'].iloc[-1] > df['BB_UPPER'].iloc[-1]:
            return "SELL"
        return "HOLD"

    def hammer_pattern(self, df: pd.DataFrame) -> str:
        """Çekiç (Hammer) formasyonu"""
        for i in range(1, len(df)):
            body = abs(df['open'].iloc[i] - df['close'].iloc[i])
            lower_shadow = df['low'].iloc[i] - min(df['open'].iloc[i], df['close'].iloc[i])
            upper_shadow = max(df['open'].iloc[i], df['close'].iloc[i]) - df['high'].iloc[i]
            if lower_shadow > 2 * body and upper_shadow < body and df['close'].iloc[i] > df['open'].iloc[i]:
                return "BUY"
        return "HOLD"

    def dark_cloud_cover(self, df: pd.DataFrame) -> str:
        """Kara Bulut Örtüsü (Dark Cloud Cover) formasyonu"""
        for i in range(1, len(df)):
            if (df['open'].iloc[i] > df['close'].iloc[i - 1] and
                df['close'].iloc[i] < (df['open'].iloc[i - 1] + df['close'].iloc[i - 1]) / 2 and
                df['close'].iloc[i] < df['open'].iloc[i]):
                return "SELL"
        return "HOLD"

    def inverted_hammer(self, df: pd.DataFrame) -> str:
        """Ters Çekiç (Inverted Hammer) formasyonu"""
        for i in range(1, len(df)):
            body = abs(df['open'].iloc[i] - df['close'].iloc[i])
            upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
            lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
            if upper_shadow > 2 * body and lower_shadow < body and df['close'].iloc[i] > df['open'].iloc[i]:
                return "BUY"
        return "HOLD"

    def bullish_engulfing(self, df: pd.DataFrame) -> str:
        """Yutan Boğa (Bullish Engulfing) formasyonu"""
        for i in range(1, len(df)):
            if (df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i - 1] < df['open'].iloc[i - 1] and
                df['close'].iloc[i] > df['open'].iloc[i - 1] and
                df['open'].iloc[i] < df['close'].iloc[i - 1]):
                return "BUY"
        return "HOLD"

    def bearish_engulfing(self, df: pd.DataFrame) -> str:
        """Yutan Ayı (Bearish Engulfing) formasyonu"""
        for i in range(1, len(df)):
            if (df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i - 1] > df['open'].iloc[i - 1] and
                df['close'].iloc[i] < df['open'].iloc[i - 1] and
                df['open'].iloc[i] > df['close'].iloc[i - 1]):
                return "SELL"
        return "HOLD"

    def doji_pattern(self, df: pd.DataFrame) -> str:
        """Doji formasyonu"""
        for i in range(len(df)):
            body = abs(df['open'].iloc[i] - df['close'].iloc[i])
            if body < (df['high'].iloc[i] - df['low'].iloc[i]) * 0.1:
                return "CAUTION"
        return "HOLD"

    def morning_star(self, df: pd.DataFrame) -> str:
        """Sabah Yıldızı (Morning Star) formasyonu"""
        for i in range(2, len(df)):
            if (df['close'].iloc[i - 2] < df['open'].iloc[i - 2] and
                abs(df['close'].iloc[i - 1] - df['open'].iloc[i - 1]) < (df['high'].iloc[i - 1] - df['low'].iloc[i - 1]) * 0.1 and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i] > df['open'].iloc[i - 2]):
                return "BUY"
        return "HOLD"

    def three_white_soldiers(self, df: pd.DataFrame) -> str:
        """Üç Beyaz Asker (Three White Soldiers) formasyonu"""
        for i in range(2, len(df)):
            if (df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i - 1] > df['open'].iloc[i - 1] and
                df['close'].iloc[i - 2] > df['open'].iloc[i - 2] and
                df['close'].iloc[i] > df['close'].iloc[i - 1] > df['close'].iloc[i - 2]):
                return "BUY"
        return "HOLD"
   
    def generate_ml_signals(self, df: pd.DataFrame) -> dict:
        """ML sinyalleri üret"""
        try:
            # DataFrame'i kopyala
            df_features = df.copy()
    
            # Temel özellikleri hesapla
            df_features['Price_Change'] = df_features['close'].pct_change()
            df_features['Volume_Change'] = df_features['volume'].pct_change()
            df_features['Daily_Return'] = (df_features['close'] - df_features['open']) / df_features['open']
    
            # Moving averages
            df_features['SMA_20'] = df_features['close'].rolling(window=20).mean()
            df_features['EMA_20'] = df_features['close'].ewm(span=20, adjust=False).mean()
    
            # Volatilite
            df_features['Volatility'] = df_features['close'].rolling(window=20).std()
    
            # RSI
            df_features['RSI'] = ta.rsi(df_features['close'], length=14)
    
            # MACD basitleştirilmiş
            ema12 = df_features['close'].ewm(span=12, adjust=False).mean()
            ema26 = df_features['close'].ewm(span=26, adjust=False).mean()
            df_features['MACD'] = ema12 - ema26
    
            # Özellik seçimi - train_model.py ile aynı sıra ve isimde olmalı
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'Price_Change', 'Volume_Change', 'Daily_Return',
                'SMA_20', 'EMA_20', 'Volatility', 'RSI', 'MACD'
            ]
    
            # Son satırı al ve özellikleri hazırla
            features = df_features[feature_columns].iloc[-1].to_frame().T
    
            # Debug için özellikleri logla
            logging.info(f"Features before cleaning: {features.to_dict('records')}")
    
            # Sonsuz ve aşırı değerleri temizle
            features = clean_infinite_values(features)
    
            # NaN değerleri doldur
            features = features.ffill()
            features = features.bfill()
            features = features.fillna(0)
    
            # Debug için özellikleri logla
            logging.info(f"Features after cleaning: {features.to_dict('records')}")
    
            # Ölçeklendirme işlemi
            scaled_features = self.scaler.transform(features)
    
            # Debug için ölçeklendirilmiş özellikleri logla
            logging.info(f"Scaled features: {scaled_features}")
    
            # Tahmin
            prediction = self.model.predict(scaled_features)
            probabilities = self.model.predict_proba(scaled_features)
            probability = probabilities[0][prediction[0]]
    
            # Debug için tahmin ve olasılıkları logla
            logging.info(f"Prediction: {prediction}, Probabilities: {probabilities}")
    
            return {
                'type': 'BUY' if prediction[0] == 1 else 'SELL',
                'probability': float(probability)
            }
    
        except Exception as e:
            logging.error(f"ML sinyal üretim hatası: {e}")
            return {'type': 'NONE', 'probability': 0}
    

        
    def generate_signals(self, df: pd.DataFrame) -> dict:
        """Geliştirilmiş teknik analiz sinyalleri üretimi"""
        try:
            required_columns = [
                'RSI', 'MACD', 'MACD_SIGNAL', 'BB_UPPER', 'BB_LOWER',
                'StochRSI_K', 'StochRSI_D', 'StochRSI', 'ADX', 'DI_plus', 'DI_minus', 
                'ATR', 'SAR', 'MFI', 'CMF', 'OBV', 'OBV_EMA', 'ROC', 'volume', 'Volatility', 'Volume_Change'
            ]
    
            # Gerekli sütunların kontrolü ve eksik sütunları hesaplama
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                for col in missing_columns:
                    if col == 'Volatility':
                        df['Volatility'] = df['close'].rolling(window=20).std()
                    elif col == 'Volume_Change':
                        df['Volume_Change'] = df['volume'].pct_change()
                    # Diğer eksik sütunların hesaplanması burada yapılabilir
    
            # Pattern sinyallerini başlangıçta hesapla
            hammer_pattern = self.hammer_pattern(df)
            doji_pattern = self.doji_pattern(df)
    
            last_row = df.iloc[-1]
            signal_strength = 0
            total_weight = 0
            buy_score = 0
            sell_score = 0
    
            # Genişletilmiş ağırlıklar ve skorlar
            weights = {
                'TECHNICAL': {
                    'RSI': 10,
                    'MACD': 10,
                    'BB': 10,
                    'STOCH': 10,
                    'ADX': 10,
                    'ATR': 10,
                    'SAR': 10,
                    'MFI': 8,
                    'CMF': 8,
                    'OBV': 8,
                    'ROC': 8
                },
                'PATTERN': {
                    'HAMMER': 10,
                    'DOJI': 5,
                    'ENGULFING': 10,
                    'MORNING_STAR': 8,
                    'EVENING_STAR': 8,
                    'THREE_WHITE_SOLDIERS': 8,
                    'THREE_BLACK_CROWS': 8
                },
                'TREND': {
                    'EMA_TREND': 12,
                    'VOLUME_TREND': 10,
                    'MOMENTUM': 10
                }
            }
    
            # RSI Analizi
            if 'RSI' in df.columns:
                weight = weights['TECHNICAL']['RSI']
                total_weight += weight
                rsi = last_row['RSI']
                if rsi < 30:
                    buy_score += weight * ((30 - rsi) / 30)
                elif rsi > 70:
                    sell_score += weight * ((rsi - 70) / 30)
    
            # MACD Analizi
            if all(col in df.columns for col in ['MACD', 'MACD_SIGNAL']):
                weight = weights['TECHNICAL']['MACD']
                total_weight += weight
                # Small epsilon value to prevent division by zero
                epsilon = 1e-10
                if abs(last_row['MACD']) > epsilon:
                    if last_row['MACD'] > last_row['MACD_SIGNAL']:
                        buy_score += weight * (abs(last_row['MACD'] - last_row['MACD_SIGNAL']) / abs(last_row['MACD']))
                    else:
                        sell_score += weight * (abs(last_row['MACD'] - last_row['MACD_SIGNAL']) / abs(last_row['MACD']))
                else:
                    if last_row['MACD'] > last_row['MACD_SIGNAL']:
                        buy_score += weight * abs(last_row['MACD'] - last_row['MACD_SIGNAL'])
                    else:
                        sell_score += weight * abs(last_row['MACD'] - last_row['MACD_SIGNAL'])
    
            # Bollinger Bands Analizi
            if all(col in df.columns for col in ['BB_UPPER', 'BB_LOWER']):
                weight = weights['TECHNICAL']['BB']
                total_weight += weight
                bb_range = last_row['BB_UPPER'] - last_row['BB_LOWER']
                if last_row['close'] < last_row['BB_LOWER']:
                    distance = (last_row['BB_LOWER'] - last_row['close']) / bb_range
                    buy_score += weight * min(distance, 1.0)
                elif last_row['close'] > last_row['BB_UPPER']:
                    distance = (last_row['close'] - last_row['BB_UPPER']) / bb_range
                    sell_score += weight * min(distance, 1.0)
    
            # StochRSI Analizi
            if all(col in df.columns for col in ['StochRSI_K', 'StochRSI_D']):
                weight = weights['TECHNICAL']['STOCH']
                total_weight += weight
                if last_row['StochRSI_K'] < 20 and last_row['StochRSI_D'] < 20:
                    buy_score += weight * (1 - max(last_row['StochRSI_K'], last_row['StochRSI_D']) / 20)
                elif last_row['StochRSI_K'] > 80 and last_row['StochRSI_D'] > 80:
                    sell_score += weight * (min(last_row['StochRSI_K'], last_row['StochRSI_D']) - 80) / 20
    
            # ADX Analizi
            if all(col in df.columns for col in ['ADX', 'DI_plus', 'DI_minus']):
                weight = weights['TECHNICAL']['ADX']
                total_weight += weight
                adx = last_row['ADX']
                if adx > 25 and last_row['DI_plus'] > last_row['DI_minus']:
                    buy_score += weight * ((adx - 25) / 25)
                elif adx > 25 and last_row['DI_minus'] > last_row['DI_plus']:
                    sell_score += weight * ((adx - 25) / 25)
    
            # ATR Analizi
            atr_ratio = 1.0
            if 'ATR' in df.columns:
                weight = weights['TECHNICAL']['ATR']
                total_weight += weight
                atr = last_row['ATR']
                atr_ratio = atr / df['ATR'].mean()
                if atr < df['ATR'].mean():
                    buy_score += weight * (df['ATR'].mean() / atr)
                elif atr > df['ATR'].mean():
                    sell_score += weight * (atr / df['ATR'].mean())
    
            # SAR Analizi
            if 'SAR' in df.columns:
                weight = weights['TECHNICAL']['SAR']
                total_weight += weight
                if last_row['SAR'] < last_row['close']:
                    buy_score += weight
                elif last_row['SAR'] > last_row['close']:
                    sell_score += weight
    
            # MFI Analizi
            if 'MFI' in df.columns:
                weight = weights['TECHNICAL']['MFI']
                total_weight += weight
                if last_row['MFI'] < 20:
                    buy_score += weight * ((20 - last_row['MFI']) / 20)
                elif last_row['MFI'] > 80:
                    sell_score += weight * ((last_row['MFI'] - 80) / 20)
    
            # CMF Analizi
            if 'CMF' in df.columns:
                weight = weights['TECHNICAL']['CMF']
                total_weight += weight
                if last_row['CMF'] > 0:
                    buy_score += weight * min(abs(last_row['CMF']), 1.0)
                else:
                    sell_score += weight * min(abs(last_row['CMF']), 1.0)
    
            # OBV Trend Analizi
            if all(col in df.columns for col in ['OBV', 'OBV_EMA']):
                weight = weights['TECHNICAL']['OBV']
                total_weight += weight
                if last_row['OBV'] > last_row['OBV_EMA']:
                    buy_score += weight
                else:
                    sell_score += weight
    
            # ROC Analizi
            if 'ROC' in df.columns:
                weight = weights['TECHNICAL']['ROC']
                total_weight += weight
                if last_row['ROC'] > 0:
                    buy_score += weight * min(last_row['ROC'] / 2, 1.0)
                else:
                    sell_score += weight * min(abs(last_row['ROC']) / 2, 1.0)
    
            # Formasyon Analizleri
            # Hammer Pattern
            if hammer_pattern == "BUY":
                weight = weights['PATTERN']['HAMMER']
                total_weight += weight
                buy_score += weight
    
            # Doji Pattern
            if doji_pattern == "CAUTION":
                weight = weights['PATTERN']['DOJI']
                total_weight += weight
                if buy_score > sell_score:
                    sell_score += weight
                else:
                    buy_score += weight
    
            # Engulfing Patterns
            if df['BULLISH_ENGULFING'].iloc[-1]:
                weight = weights['PATTERN']['ENGULFING']
                total_weight += weight
                buy_score += weight
            elif df['BEARISH_ENGULFING'].iloc[-1]:
                weight = weights['PATTERN']['ENGULFING']
                total_weight += weight
                sell_score += weight
    
            # Star Patterns
            if df['MORNING_STAR'].iloc[-1]:
                weight = weights['PATTERN']['MORNING_STAR']
                total_weight += weight
                buy_score += weight
            elif df['EVENING_STAR'].iloc[-1]:
                weight = weights['PATTERN']['EVENING_STAR']
                total_weight += weight
                sell_score += weight
    
            # Soldier/Crow Patterns
            if df['THREE_WHITE_SOLDIERS'].iloc[-1]:
                weight = weights['PATTERN']['THREE_WHITE_SOLDIERS']
                total_weight += weight
                buy_score += weight
            elif df['THREE_BLACK_CROWS'].iloc[-1]:
                weight = weights['PATTERN']['THREE_BLACK_CROWS']
                total_weight += weight
                sell_score += weight
    
            # Trend Analysis
            if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
                weight = weights['TREND']['EMA_TREND']
                total_weight += weight
                if last_row['EMA_20'] > last_row['EMA_50']:
                    buy_score += weight
                else:
                    sell_score += weight
    
            # Volume Trend Analysis
            if 'volume' in df.columns:
                weight = weights['TREND']['VOLUME_TREND']
                total_weight += weight
                avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
                if last_row['volume'] > avg_volume * 1.5:
                    if last_row['close'] > last_row['open']:
                        buy_score += weight
                    else:
                        sell_score += weight
    
            # Hacim düşüşü ve düşük volatilite kontrolü
            volume_change = last_row['Volume_Change']
            volatility = last_row['Volatility']
            if volume_change < -0.1:  # Hacim %10'dan fazla düşmüşse
                logging.info("Hacimde belirgin düşüş gözlemlendi, işlem yapılmayacak.")
                return {'type': 'NONE', 'reason': 'volume_drop'}
            if volatility < 0.0001:  # Volatilite belirli bir eşikten düşükse
                logging.info("Volatilite çok düşük, işlem yapılmayacak.")
                return {'type': 'NONE', 'reason': 'low_volatility'}
    
            # Sonuçları hesapla
            if total_weight > 0:
                buy_strength = buy_score / total_weight
                sell_strength = sell_score / total_weight
    
                # Trend gücünü hesapla
                trend_strength = min(last_row['ADX'] / 50.0, 1.0) if 'ADX' in df.columns else 0
    
                # Volatilite bazlı ayarlama
                volatility_multiplier = 1.0
                if atr_ratio > 1.5:
                    volatility_multiplier = 0.8
                elif atr_ratio < 0.5:
                    volatility_multiplier = 1.2
    
                # Güçleri ayarla
                buy_strength = buy_strength * volatility_multiplier * (1 + trend_strength * 0.5)
                sell_strength = sell_strength * volatility_multiplier * (1 + trend_strength * 0.5)
    
                # Sinyal türünü ve gücünü belirle
                if buy_strength > sell_strength and buy_strength > 0.3:  # Eşik değeri düşürüldü
                    signal_type = 'BUY'
                    signal_strength = buy_strength
                elif sell_strength > buy_strength and sell_strength > 0.3:  # Eşik değeri düşürüldü
                    signal_type = 'SELL'
                    signal_strength = sell_strength
                else:
                    signal_type = 'HOLD'
                    signal_strength = max(buy_strength, sell_strength)  # HOLD durumunda en yüksek gücü al
    
                # Güven seviyesi hesaplama
                confidence = abs(buy_strength - sell_strength)
    
                return {
                    'type': signal_type,
                    'strength': float(signal_strength),  # Artık her zaman bir değer olacak
                    'confidence': float(confidence),
                    'buy_score': float(buy_score),
                    'sell_score': float(sell_score),
                    'total_weight': total_weight,
                    'buy_strength': float(buy_strength),
                    'sell_strength': float(sell_strength),
                    'trend_strength': float(trend_strength),
                    'volatility_state': 'HIGH' if atr_ratio > 1.5 else 'LOW' if atr_ratio < 0.5 else 'NORMAL',
                    'pattern_signals': {
                        'hammer': hammer_pattern,
                        'doji': doji_pattern,
                        'bullish_engulfing': bool(df['BULLISH_ENGULFING'].iloc[-1]),
                        'bearish_engulfing': bool(df['BEARISH_ENGULFING'].iloc[-1]),
                        'morning_star': bool(df['MORNING_STAR'].iloc[-1]),
                        'evening_star': bool(df['EVENING_STAR'].iloc[-1]),
                        'three_white_soldiers': bool(df['THREE_WHITE_SOLDIERS'].iloc[-1]),
                        'three_black_crows': bool(df['THREE_BLACK_CROWS'].iloc[-1])
                    }
                }
    
            return {
                'type': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'buy_score': 0.0,
                'sell_score': 0.0,
                'total_weight': 0,
                'buy_strength': 0.0,
                'sell_strength': 0.0,
                'trend_strength': 0.0,
                'volatility_state': 'NORMAL',
                'pattern_signals': {
                    'hammer': 'HOLD',
                    'doji': 'HOLD',
                    'bullish_engulfing': False,
                    'bearish_engulfing': False,
                    'morning_star': False,
                    'evening_star': False,
                    'three_white_soldiers': False,
                    'three_black_crows': False
                }
            }
    
        except Exception as e:
            logging.error(f"Signal generation error: {str(e)}", exc_info=True)
            return {'type': 'NONE', 'reason': 'error'}
    def _validate_signals(self, ml_signal: dict, technical_signal: dict) -> bool:
        """Sinyalleri doğrula ve sinyal gücünü optimize et"""
        try:
            logging.info(f"ML Sinyal: {ml_signal}")
            logging.info(f"Teknik Sinyal: {technical_signal}")

            # Ana metrikler
            signal_type = technical_signal.get('type', 'NONE')
            buy_strength = technical_signal.get('buy_strength', 0)
            sell_strength = technical_signal.get('sell_strength', 0)
            signal_confidence = technical_signal.get('confidence', 0)
            trend_strength = technical_signal.get('trend_strength', 0)
            ml_probability = float(ml_signal.get('probability', 0))

            # Sinyal gücünü yeniden hesapla
            if signal_type == 'BUY':
                signal_strength = buy_strength * (1 + trend_strength * 0.5)
            elif signal_type == 'SELL':
                signal_strength = sell_strength * (1 + trend_strength * 0.5)
            else:
                signal_strength = max(buy_strength, sell_strength) * (1 + trend_strength * 0.3)

            # Minimum eşik değerleri
            min_strength = 1.01       # Düşürüldü
            min_confidence = 1    # Düşürüldü
            min_ml_prob = 0.55      # ML minimum olasılık

            # Formasyon desteği kontrolü
            pattern_signals = technical_signal.get('pattern_signals', {})
            supporting_patterns = 0

            if signal_type == 'BUY':
                if pattern_signals.get('hammer') == 'BUY':
                    supporting_patterns += 1
                if pattern_signals.get('bullish_engulfing'):
                    supporting_patterns += 1
                if pattern_signals.get('morning_star'):
                    supporting_patterns += 1
                if pattern_signals.get('three_white_soldiers'):
                    supporting_patterns += 1
                if pattern_signals.get('doji') == 'CAUTION' and sell_strength > buy_strength:
                    supporting_patterns += 1

            elif signal_type == 'SELL':
                if pattern_signals.get('evening_star'):
                    supporting_patterns += 1
                if pattern_signals.get('bearish_engulfing'):
                    supporting_patterns += 1
                if pattern_signals.get('three_black_crows'):
                    supporting_patterns += 1
                if pattern_signals.get('doji') == 'CAUTION' and buy_strength > sell_strength:
                    supporting_patterns += 1

            # Aktif formasyon sayısı
            total_patterns = sum(1 for value in pattern_signals.values() if value and value != 'HOLD')

            # Pattern desteği oranı
            pattern_support = supporting_patterns / max(total_patterns, 1) if total_patterns > 0 else 0

            # ML ve Teknik sinyal uyumu kontrolü
            signal_agreement = (
                (signal_type == 'BUY' and ml_signal['type'] == 'BUY') or
                (signal_type == 'SELL' and ml_signal['type'] == 'SELL')
            )

            # Sinyal gücünü güncelle
            technical_signal['strength'] = float(signal_strength)

            # Detaylı sinyal istatistikleri
            signal_details = (
                f"Sinyal İstatistikleri:\n"
                f"Sinyal Türü: {signal_type}\n"
                f"Alış Skoru: {technical_signal.get('buy_score', 0):.2f}\n"
                f"Satış Skoru: {technical_signal.get('sell_score', 0):.2f}\n"
                f"Alış Gücü: {buy_strength:.2f}\n"
                f"Satış Gücü: {sell_strength:.2f}\n"
                f"Toplam Ağırlık: {technical_signal.get('total_weight', 0)}\n"
                f"Sinyal Gücü: {signal_strength:.2f}\n"
                f"Güven Seviyesi: {signal_confidence:.2f}\n"
                f"ML Olasılığı: {ml_probability:.2f}\n"
                f"Trend Gücü: {trend_strength:.2f}\n"
                f"Formasyon Desteği: {pattern_support:.2f}\n"
                f"Formasyon Sinyalleri: {pattern_signals}"
            )

            logging.info(signal_details)

            # Doğrulama koşulları
            conditions = {
                'Sinyal Tipi Eşleşmesi': signal_agreement,
                'Sinyal Gücü Yeterli': signal_strength >= min_strength,
                'Güven Seviyesi Yeterli': signal_confidence >= min_confidence,
                'ML Olasılığı Yeterli': ml_probability >= min_ml_prob
            }

            # Doğrulama sonuçlarını logla
            validation_details = "\nDoğrulama Detayları:"
            for condition_name, condition_met in conditions.items():
                validation_details += f"\n{condition_name}: {condition_met}"

            logging.info(validation_details)

            # Sinyal onaylama
            if signal_type in ['BUY', 'SELL']:
                if signal_agreement:
                    if (signal_strength >= min_strength and 
                        (signal_confidence >= min_confidence or pattern_support > 0) and
                        ml_probability >= min_ml_prob):

                        # Formasyon desteği varsa güven skorunu artır
                        if pattern_support > 0:
                            signal_confidence *= (1 + pattern_support)

                        logging.info(f"✅ Sinyal Onaylandı: {signal_type}\n"
                                   f"Final Güven Skoru: {signal_confidence:.2f}\n"
                                   f"Final Sinyal Gücü: {signal_strength:.2f}")
                        return True

            logging.info("❌ Sinyal Reddedildi")
            return False

        except Exception as e:
            logging.error(f"Sinyal doğrulama hatası: {e}", exc_info=True)
            return False
    def is_trading_allowed(self) -> bool:
        """Trading koşullarını kontrol et"""
        current_hour = datetime.now().hour
        if not (self.config['trading_hours']['start_hour'] <= 
                current_hour < self.config['trading_hours']['end_hour']):
            return False
            
        if self.daily_trades >= self.config['risk_management']['max_trades_per_day']:
            return False
            
        return True

    def calculate_position_size(self, symbol: str, current_price: float) -> float:
        """
        Geliştirişmiş pozisyon büyüklüğü hesaplama:
        - Risk yüzdesi ayarlanabilir hale getirildi
        - Maksimum pozisyon büyüklüğü sınırı eklendi
        - Kelly Criterion uygulandı
        - Dinamik risk ayarı eklendi
        - Bütçenin %70'i ile işlem yapma
        """
        try:
            # Hesap bakiyesini al
            balance = float(self.get_account_balance()) * 0.7  # Bütçenin %70'i
            logging.info(f"Mevcut bakiye (bütçenin %70'i): {balance} USDT")

            # Risk parametreleri
            base_risk_percentage = 0.02  # Temel risk yüzdesi %2
            max_risk_percentage = 0.5   # Maksimum risk yüzdesi %5
            min_trade_value = 5.1        # Minimum işlem değeri USDT
            max_position_value = 1000    # Maksimum pozisyon değeri USDT

            # Son 20 işlemin performansını kontrol et
            win_rate = self.calculate_win_rate(20)
            avg_win_loss_ratio = self.calculate_avg_win_loss_ratio(20)

            # Kelly Criterion hesaplama
            if win_rate > 0 and avg_win_loss_ratio > 0:
                kelly_percentage = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
                kelly_percentage = max(0, min(kelly_percentage, max_risk_percentage))
            else:
                kelly_percentage = base_risk_percentage

            # Volatilite bazlı risk ayarı
            volatility_multiplier = self.calculate_volatility_multiplier(symbol)
            adjusted_risk = kelly_percentage * volatility_multiplier

            # Final risk yüzdesini belirle
            final_risk_percentage = min(adjusted_risk, max_risk_percentage)
            risk_amount = balance * final_risk_percentage

            # Pozisyon büyüklüğünü hesapla
            position_size = risk_amount / current_price

            # Pozisyon sınırlamalarını uygula
            position_value = position_size * current_price
            if position_value > max_position_value:
                position_size = max_position_value / current_price

            if position_value < min_trade_value:
                logging.warning(f"İşlem değeri çok düşük: {position_value} USDT")
                return 0

            # Sembol hassasiyetine göre yuvarla
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                position_size = self.round_to_precision(
                    position_size, 
                    symbol_info['quantityPrecision']
                )

            logging.info(
                f"Pozisyon hesaplama: Risk%={final_risk_percentage*100:.2f}, "
                f"Size={position_size}, Value={position_size*current_price:.2f} USDT"
            )

            return position_size

        except Exception as e:
            logging.error(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0


    def calculate_win_rate(self, lookback: int = 20) -> float:
        """Son n işlemin kazanma oranını hesapla"""
        try:
            trades = self.get_recent_trades(lookback)
            if not trades:
                return 0.5  # Yeterli veri yoksa varsayılan oran

            winning_trades = sum(1 for trade in trades if trade['realizedPnl'] > 0)
            return winning_trades / len(trades)
        except Exception as e:
            logging.error(f"Win rate hesaplama hatası: {e}")
            return 0.5

    def calculate_avg_win_loss_ratio(self, lookback: int = 20) -> float:
        """Son n işlemin ortalama kazanç/kayıp oranını hesapla"""
        try:
            trades = self.get_recent_trades(lookback)
            if not trades:
                return 1.0  # Yeterli veri yoksa varsayılan oran

            wins = [float(t['realizedPnl']) for t in trades if float(t['realizedPnl']) > 0]
            losses = [abs(float(t['realizedPnl'])) for t in trades if float(t['realizedPnl']) < 0]

            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 1

            return avg_win / avg_loss if avg_loss else 1.0
        except Exception as e:
            logging.error(f"Win/Loss ratio hesaplama hatası: {e}")
            return 1.0

    def calculate_volatility_multiplier(self, symbol: str) -> float:
        """Volatilite bazlı risk çarpanı hesapla"""
        try:
            df = self.get_klines(symbol)
            if df.empty:
                return 1.0

            # Son 20 mumdaki volatiliteyi hesapla
            returns = df['close'].pct_change().tail(20)
            volatility = returns.std()

            # Volatilite bazlı çarpan (0.5 ile 1.0 arasında)
            base_volatility = 0.02  # Baz volatilite
            multiplier = 1.0 - min(max(volatility / base_volatility - 1, 0), 0.5)

            return multiplier
        except Exception as e:
            logging.error(f"Volatilite çarpanı hesaplama hatası: {e}")
            return 1.0




        
    def get_symbol_info(self, symbol: str) -> dict:
        """Sembol bilgilerini al"""
        try:
            exchange_info = self.client.exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    return {
                        'pricePrecision': s['pricePrecision'],
                        'quantityPrecision': s['quantityPrecision'],
                        'minQty': float(next(f['minQty'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE')),
                        'maxQty': float(next(f['maxQty'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE')),
                        'stepSize': float(next(f['stepSize'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE'))
                    }
            return None
        except Exception as e:
            logging.error(f"Sembol bilgisi alma hatası: {e}")
            return None

    def round_to_precision(self, value: float, precision: int) -> float:
        """Değeri belirtilen hassasiyete yuvarla"""
        factor = 10 ** precision
        return float(round(value * factor) / factor)

    async def execute_trade_with_risk_management(self, symbol: str, signal_type: str, current_price: float):
        """İşlem yönetimi ve risk kontrolü"""
        try:
            trade_side = signal_type

            # Mevcut pozisyonu kontrol et
            current_position = self.positions.get(symbol)
            if current_position:
                # Eğer mevcut pozisyon varsa ve sinyal ters yöndeyse pozisyonu kapat
                if (current_position['side'] == 'BUY' and signal_type == 'SELL') or (current_position['side'] == 'SELL' and signal_type == 'BUY'):
                    try:
                        close_order = self.client.new_order(
                            symbol=symbol,
                            side='SELL' if current_position['side'] == 'BUY' else 'BUY',
                            type='MARKET',
                            quantity=current_position['quantity']
                        )
                        logging.info(f"Mevcut pozisyon kapatıldı: {symbol} {current_position['side']} {current_position['quantity']}")
                        await self.send_telegram(f"⚠️ Mevcut pozisyon kapatıldı: {symbol} {current_position['side']} {current_position['quantity']}")
                        self.positions.pop(symbol)
                    except Exception as close_order_error:
                        logging.error(f"Mevcut pozisyon kapatma hatası: {close_order_error}")
                        await self.send_telegram(f"⚠️ Mevcut pozisyon kapatma hatası: {symbol} - {str(close_order_error)}")
                        return False

            # Hesap bakiyesini al
            balance = float(self.get_account_balance()) * 0.9  # Bütçenin %90'ı
            logging.info(f"Mevcut bakiye (bütçenin %90'i): {balance} USDT")

            # Check if balance is below 5 USD
            if balance < 0:
                logging.warning(f"Yetersiz bakiye: {balance} USDT. İşlem yapılmayacak.")
                await self.send_telegram(f"⚠️ Yetersiz bakiye: {balance} USDT. İşlem yapılmayacak.")
                return False

            # Kaldıraç ayarı
            try:
                self.client.change_leverage(
                    symbol=symbol,
                    leverage=9
                )
                logging.info(f"Kaldıraç ayarlandı: {symbol} 12x")
            except Exception as e:
                logging.error(f"Kaldıraç ayarlama hatası: {e}")
                return False

            # Sembol bilgilerini al
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                logging.error(f"Sembol bilgisi alınamadı: {symbol}")
                return False

            # Minimum işlem değeri (5.1 USDT) için quantity hesaplama
            min_notional = 5  # Biraz daha yüksek tutalım
            min_quantity = min_notional / current_price

            # Risk bazlı quantity hesaplama
            risk_percentage = 0.95
            risk_based_quantity = (balance * risk_percentage) / current_price

            # İkisinden büyük olanı seç
            quantity = max(min_quantity, risk_based_quantity)

            # Quantity'yi sembol hassasiyetine yuvarla
            quantity = self.round_to_precision(quantity, symbol_info['quantityPrecision'])
            price = self.round_to_precision(current_price, symbol_info['pricePrecision'])

            # Son kontrol
            final_notional = quantity * price
            logging.info(f"Final işlem değeri: {final_notional} USDT")

            if final_notional < min_notional:
                # Quantity'yi tekrar ayarla
                quantity = self.round_to_precision((min_notional / price) * 1.01, symbol_info['quantityPrecision'])
                final_notional = quantity * price
                logging.info(f"Quantity yeniden ayarlandı: {quantity} ({final_notional} USDT)")

            # Market emri oluştur
            try:
                order = self.client.new_order(
                    symbol=symbol,
                    side=trade_side,
                    type='MARKET',
                    quantity=quantity
                )

                # Stop Loss ve Take Profit hesapla
                sl_price = price * (0.98 if trade_side == 'BUY' else 1.02)
                tp_price = price * (1.03 if trade_side == 'BUY' else 0.97)

                # Stop Loss emri
                sl_order = self.client.new_order(
                    symbol=symbol,
                    side='SELL' if trade_side == 'BUY' else 'BUY',
                    type='STOP_MARKET',
                    stopPrice=self.round_to_precision(sl_price, symbol_info['pricePrecision']),
                    closePosition='true'
                )

                # Take Profit emri
                tp_order = self.client.new_order(
                    symbol=symbol,
                    side='SELL' if trade_side == 'BUY' else 'BUY',
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=self.round_to_precision(tp_price, symbol_info['pricePrecision']),
                    closePosition='true'
                )

                # Pozisyonu kaydet
                self.positions[symbol] = {
                    'side': trade_side,
                    'quantity': quantity,
                    'entry_price': price
                }

                message = (
                    f"✅ İşlem Gerçekleşti\n"
                    f"Sembol: {symbol}\n"
                    f"Yön: {trade_side}\n"
                    f"Miktar: {quantity}\n"
                    f"Fiyat: {price}\n"
                    f"İşlem Değeri: {final_notional:.2f} USDT\n"
                    f"Stop Loss: {sl_price}\n"
                    f"Take Profit: {tp_price}\n"
                    f"Kaldıraç: 12x\n"
                    f"Bakiye: {balance} USDT"
                )

                logging.info(f"İşlem başarılı: {symbol} {trade_side} {quantity}")
                await self.send_telegram(message)

            except Exception as order_error:
                logging.error(f"Order yerleştirme hatası: {order_error}")
                await self.send_telegram(f"⚠️ İşlem Hatası: {symbol} - {str(order_error)}")
                return False

            # ROI %10 olduğunda pozisyonu kapatma kontrolü
            while True:
                try:
                    # Mevcut fiyatı al
                    current_price = float(self.client.ticker_price(symbol=symbol)['price'])
                    entry_price = self.positions[symbol]['entry_price']
                    roi = (current_price - entry_price) / entry_price * 100 if trade_side == 'BUY' else (entry_price - current_price) / entry_price * 100

                    # ROI %10'a ulaştıysa pozisyonu kapat
                    if roi >= 15:
                        close_order = self.client.new_order(
                            symbol=symbol,
                            side='SELL' if trade_side == 'BUY' else 'BUY',
                            type='MARKET',
                            quantity=self.positions[symbol]['quantity']
                        )
                        logging.info(f"ROI %15'a ulaştı: {symbol} {trade_side} {self.positions[symbol]['quantity']}")
                        await self.send_telegram(f"🏆 ROI %15'a ulaştı: {symbol} {trade_side} {self.positions[symbol]['quantity']}")
                        self.positions.pop(symbol)
                        break

                    # Bekleme süresi
                    await asyncio.sleep(self.config['check_interval'])

                except Exception as roi_check_error:
                    logging.error(f"ROI kontrol hatası: {roi_check_error}")
                    await self.send_telegram(f"⚠️ ROI kontrol hatası: {symbol} - {str(roi_check_error)}")
                    break

            return True

        except Exception as e:
            logging.error(f"İşlem yönetimi hatası: {e}")
            await self.send_telegram(f"⚠️ İşlem Yönetimi Hatası: {symbol} - {str(e)}")
            return False

    def get_account_balance(self):
        """Hesap bakiyesini al (Vadeli işlemler hesabı)"""
        try:
            account = self.client.futures_account_balance()
            for asset in account:
                if asset['asset'] == 'USDT':
                    return float(asset['balance'])
            return 0.0
        except Exception as e:
            logging.error(f"Bakiye alma hatası: {e}")
            return 0.0
          
    async def _send_trade_notification(self, symbol, signal, price, size, sl, tp):
        """Trade bildirimini gönder"""
        message = (
            f"🤖 Trade Executed\n"
            f"Symbol: {symbol}\n"
            f"Type: {signal['type']}\n"
            f"Price: {price:.8f}\n"
            f"Size: {size:.8f}\n"
            f"Stop Loss: {sl:.8f}\n"
            f"Take Profit: {tp:.8f}\n"
            f"Probability: {signal['probability']:.2f}"
        )
        await self.send_telegram(message)

    def reset_daily_stats(self):
            """Günlük istatistikleri sıfırla"""
            try:
                # Günlük işlem sayısını ve kar/zarar istatistiklerini sıfırla
                self.daily_stats = {
                    'trades': 0,
                    'profit': 0.0,
                    'losses': 0.0
                }
                self.daily_trades = 0
                self.last_daily_reset = datetime.now().date()
                logging.info("Günlük istatistikler sıfırlandı")
            except Exception as e:
                logging.error(f"Günlük istatistikleri sıfırlama hatası: {str(e)}")
    async def run(self):
        """Ana bot döngüsü"""
        try:
            logging.info(f"Bot started by {self.config.get('created_by', 'unknown')}")
            await self.send_telegram("🚀 Trading Bot Activated")
    
            while True:
                try:
                    # Trading saatleri kontrolü
                    if self.is_trading_allowed():
                        for symbol in self.config['symbols']:
                            # Mum verilerini al
                            df = self.get_klines(symbol)
                            if df.empty:
                                logging.warning(f"No data received for {symbol}")
                                continue

                                # Temel göstergeleri hesapla
                            df = self.calculate_indicators(df)
                            logging.info(f"Basic indicators calculated for {symbol}")

                            # İleri seviye göstergeleri hesapla
                            df = self.calculate_advanced_indicators(df)
                            logging.info(f"Advanced indicators calculated for {symbol}")

                            # ML ve teknik sinyalleri üret
                            ml_signal = self.generate_ml_signals(df)
                            technical_signal = self.generate_signals(df)

                            # Sinyalleri doğrula
                            if self._validate_signals(ml_signal, technical_signal):
                                current_price = float(df['close'].iloc[-1])
                                logging.info(f"Sinyal onaylandı: {ml_signal['type']} (Güç: {technical_signal['strength']}, ML Olasılık: {ml_signal['probability']})")
                            
                                # Burada signal_type olarak sadece string gönderiyoruz
                                await self.execute_trade_with_risk_management(
                                    symbol=symbol,
                                    signal_type=ml_signal['type'],  # Sadece 'BUY' veya 'SELL' string'i
                                    current_price=current_price
                                )

                            # Rate limit kontrolü
                            await asyncio.sleep(self.rate_limit_delay)

                    # Günlük istatistikleri sıfırla
                    if datetime.now().date() > self.last_daily_reset:
                        self.reset_daily_stats()

                    # Ana döngü bekleme süresi
                    await asyncio.sleep(self.config['check_interval'])

                except Exception as loop_error:
                    logging.error(f"Loop iteration error: {loop_error}")
                    await self.send_telegram(f"⚠️ Error in main loop: {loop_error}")
                    await asyncio.sleep(60)

        except Exception as e:
            logging.error(f"Critical error in run method: {e}")
            await self.send_telegram("🚨 Bot stopped due to critical error!")
            raise

if __name__ == "__main__":
    # Logging ayarları
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='advanced_trading_bot.log'
    )

    try:
        # Bot instance'ını oluştur
        bot = BinanceFuturesBot()
        
        # Modern asyncio kullanımı
        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Critical error: {e}")