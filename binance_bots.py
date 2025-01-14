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

    def get_klines(self, symbol: str) -> pd.DataFrame:
        """Mum verilerini al"""
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

            # Ichimoku Cloud - Düzeltilmiş versiyon
          

            # ---- MUM FORMASYONLARI ----
            df['DOJI'] = self.calculate_doji(df)
            df['HAMMER'] = self.calculate_hammer(df)
            df['BULLISH_ENGULFING'] = self.calculate_bullish_engulfing(df)
            df['BEARISH_ENGULFING'] = self.calculate_bearish_engulfing(df)
            df['MORNING_STAR'] = self.calculate_morning_star(df)
            df['EVENING_STAR'] = self.calculate_evening_star(df)
            df['THREE_WHITE_SOLDIERS'] = self.calculate_three_white_soldiers(df)
            df['THREE_BLACK_CROWS'] = self.calculate_three_black_crows(df)

            # ---- MOMENTUM İNDİKATÖRLERİ ----
            df['ROC'] = ta.roc(df['close'], length=9)
            df['WILLIAMS_R'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=20)

            # ---- VOLATILITE İNDİKATÖRLERİ ----
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

            # ---- HACİM İNDİKATÖRLERİ ----
            if 'volume' in df.columns:
                df['OBV'] = ta.obv(df['close'], df['volume'])
                df['VWAP'] = self.calculate_vwap(df)

            # NaN değerleri temizle
            df = df.ffill().bfill()

            # Hesaplanan göstergeleri kontrol et
            required_indicators = [
                'RSI', 'MACD', 'MACD_SIGNAL', 'BB_UPPER', 'BB_LOWER',
                'SMA_20', 'EMA_20', 'EMA_50', 'EMA_200', 'StochRSI_K', 'StochRSI_D',
                'ADX'
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

    def _calculate_dynamic_stop_loss(self, price: float, atr: float, trade_type: str, multiplier: float) -> float:
        """Dinamik stop loss hesapla"""
        if trade_type == 'BUY':
            return price - (atr * multiplier)
        elif trade_type == 'SELL':
            return price + (atr * multiplier)

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
        """Teknik analiz sinyalleri üret"""
        try:
            required_columns = [
                'RSI', 'MACD', 'MACD_SIGNAL', 'BB_UPPER', 'BB_LOWER',
                'StochRSI_K', 'StochRSI_D', 'StochRSI'
            ]
    
            # Gerekli sütunların kontrolü
            missing_columns = [col for col in required_columns if col not in df.columns]
            if df.empty or missing_columns:
                logging.warning(f"Missing columns for signal generation: {missing_columns}")
                return {'type': 'NONE', 'reason': 'missing_data'}
    
            last_row = df.iloc[-1]
            signal_strength = 0
            total_weight = 0
            buy_score = 0
            sell_score = 0
    
            # Ağırlıklar ve skorlar
            weights = {
                'TECHNICAL': {
                    'RSI': 15,
                    'MACD': 15,
                    'BB': 15,
                    'STOCH': 15
                },
                'PATTERN': {
                    'HAMMER': 10,
                    'DOJI': 5,
                    'ENGULFING': 10,
                    'MORNING_STAR': 8,
                    'EVENING_STAR': 8
                }
            }
    
            # 1. Teknik İndikatör Sinyalleri
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
                if last_row['MACD'] > last_row['MACD_SIGNAL']:
                    buy_score += weight * (abs(last_row['MACD'] - last_row['MACD_SIGNAL']) / abs(last_row['MACD']))
                else:
                    sell_score += weight * (abs(last_row['MACD'] - last_row['MACD_SIGNAL']) / abs(last_row['MACD']))
    
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
    
            # 2. Formasyon Sinyalleri
            # Çekiç (Hammer)
            hammer = self.hammer_pattern(df)
            if hammer == "BUY":
                weight = weights['PATTERN']['HAMMER']
                total_weight += weight
                buy_score += weight
    
            # Doji
            doji = self.doji_pattern(df)
            if doji == "CAUTION":
                weight = weights['PATTERN']['DOJI']
                total_weight += weight
                # Doji trendin tersine sinyal verir
                if buy_score > sell_score:
                    sell_score += weight
                else:
                    buy_score += weight
    
            # Yutan Formasyonlar
            if df['BULLISH_ENGULFING'].iloc[-1]:
                weight = weights['PATTERN']['ENGULFING']
                total_weight += weight
                buy_score += weight
            elif df['BEARISH_ENGULFING'].iloc[-1]:
                weight = weights['PATTERN']['ENGULFING']
                total_weight += weight
                sell_score += weight
    
            # Morning Star ve Evening Star
            if df['MORNING_STAR'].iloc[-1]:
                weight = weights['PATTERN']['MORNING_STAR']
                total_weight += weight
                buy_score += weight
            elif df['EVENING_STAR'].iloc[-1]:
                weight = weights['PATTERN']['EVENING_STAR']
                total_weight += weight
                sell_score += weight
    
            # Sonuçları hesapla
            if total_weight > 0:
                buy_strength = buy_score / total_weight
                sell_strength = sell_score / total_weight
                
                # Sinyal türünü ve gücünü belirle
                if buy_strength > sell_strength:
                    signal_type = 'BUY'
                    signal_strength = buy_strength
                elif sell_strength > buy_strength:
                    signal_type = 'SELL'
                    signal_strength = sell_strength
                else:
                    signal_type = 'HOLD'
                    signal_strength = 0
    
                # Güven seviyesi hesaplama
                confidence = abs(buy_strength - sell_strength)
    
                return {
                    'type': signal_type,
                    'strength': float(signal_strength),
                    'confidence': float(confidence),
                    'buy_score': float(buy_score),
                    'sell_score': float(sell_score),
                    'total_weight': total_weight,
                    'buy_strength': float(buy_strength),
                    'sell_strength': float(sell_strength),
                    'pattern_signals': {
                        'hammer': hammer,
                        'doji': doji,
                        'bullish_engulfing': bool(df['BULLISH_ENGULFING'].iloc[-1]),
                        'bearish_engulfing': bool(df['BEARISH_ENGULFING'].iloc[-1]),
                        'morning_star': bool(df['MORNING_STAR'].iloc[-1]),
                        'evening_star': bool(df['EVENING_STAR'].iloc[-1])
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
                'pattern_signals': {
                    'hammer': 'HOLD',
                    'doji': 'HOLD',
                    'bullish_engulfing': False,
                    'bearish_engulfing': False,
                    'morning_star': False,
                    'evening_star': False
                }
            }
    
        except Exception as e:
            logging.error(f"Signal generation error: {str(e)}", exc_info=True)
            return {'type': 'NONE', 'reason': 'error'}
        
    def _validate_signals(self, ml_signal: dict, technical_signal: dict) -> bool:
      """Sinyalleri doğrula"""
      try:
          logging.info(f"ML Sinyal: {ml_signal}")
          logging.info(f"Teknik Sinyal: {technical_signal}")
          # Detaylı sinyal istatistikleri
          signal_details = (
              f"Sinyal İstatistikleri:\n"
              f"Sinyal Türü: {technical_signal.get('type', 'NONE')}\n"
              f"Alış Skoru: {technical_signal.get('buy_score', 0):.2f}\n"
              f"Satış Skoru: {technical_signal.get('sell_score', 0):.2f}\n"
              f"Alış Gücü: {technical_signal.get('buy_strength', 0):.2f}\n"
              f"Satış Gücü: {technical_signal.get('sell_strength', 0):.2f}\n"
              f"Toplam Ağırlık: {technical_signal.get('total_weight', 0)}\n"
              f"Sinyal Gücü: {technical_signal.get('strength', 0):.2f}\n"
              f"Güven Seviyesi: {technical_signal.get('confidence', 0):.2f}\n"
              f"ML Olasılığı: {ml_signal.get('probability', 0):.2f}\n"
              f"Formasyon Sinyalleri: {technical_signal.get('pattern_signals', {})}"
          )
          logging.info(signal_details)
          if technical_signal['type'] in ['BUY', 'SELL']:
              # Ana metrikler
              signal_strength = technical_signal.get('strength', 0)
              buy_strength = technical_signal.get('buy_strength', 0)
              sell_strength = technical_signal.get('sell_strength', 0)
              signal_confidence = technical_signal.get('confidence', 0)
              ml_probability = float(ml_signal.get('probability', 0))
              
              # Minimum eşik değerleri
              min_strength = 1.20       # Düşürüldü: 0.60 -> 0.05
              min_confidence = 1.01     # Düşürüldü: 0.40 -> 0.02
              min_ml_prob = 0.62       # Düşürüldü: 0.55 -> 0.51
              
              # Formasyon desteği kontrolü
              pattern_signals = technical_signal.get('pattern_signals', {})
              supporting_patterns = 0
              
              if technical_signal['type'] == 'BUY':
                  # Alış yönünde formasyon kontrolü
                  if pattern_signals.get('hammer') == 'BUY':
                      supporting_patterns += 1
                  if pattern_signals.get('bullish_engulfing'):
                      supporting_patterns += 1
                  if pattern_signals.get('morning_star'):
                      supporting_patterns += 1
                  # Doji kontrolü
                  if pattern_signals.get('doji') == 'CAUTION' and sell_strength > buy_strength:
                      supporting_patterns += 1
              elif technical_signal['type'] == 'SELL':
                  # Satış yönünde formasyon kontrolü
                  if pattern_signals.get('evening_star'):
                      supporting_patterns += 1
                  if pattern_signals.get('bearish_engulfing'):
                      supporting_patterns += 1
                  # Doji kontrolü
                  if pattern_signals.get('doji') == 'CAUTION' and buy_strength > sell_strength:
                      supporting_patterns += 1
              # Aktif formasyon sayısı
              total_patterns = sum(1 for value in pattern_signals.values() if value and value != 'HOLD')
              
              # Pattern desteği oranı
              pattern_support = supporting_patterns / max(total_patterns, 1) if total_patterns > 0 else 0
              # ML ve Teknik sinyal uyumu kontrolü
              signal_agreement = (
                  (technical_signal['type'] == 'BUY' and ml_signal['type'] == 'BUY') or
                  (technical_signal['type'] == 'SELL' and ml_signal['type'] == 'SELL')
              )
              # Doğrulama koşulları
              conditions = {
                  'Sinyal Tipi Eşleşmesi': signal_agreement,
                  'Sinyal Gücü Yeterli': signal_strength >= min_strength,
                  'Güven Seviyesi Yeterli': signal_confidence >= min_confidence,
                  'ML Olasılığı Yeterli': ml_probability >= min_ml_prob
              }
              validation_details = "\nDoğrulama Detayları:"
              for condition_name, condition_met in conditions.items():
                  validation_details += f"\n{condition_name}: {condition_met}"
              
              validation_details += f"\nFormasyon Destek Oranı: {pattern_support:.2f}"
              validation_details += f"\nDestekleyen Formasyon Sayısı: {supporting_patterns}"
              validation_details += f"\nToplam Aktif Formasyon: {total_patterns}"
              
              logging.info(validation_details)
              # Sinyal onaylama koşulları
              if signal_agreement:  # ML ve Teknik sinyal aynı yönde ise
                  if (signal_strength >= min_strength and 
                      (signal_confidence >= min_confidence or pattern_support > 0) and
                      ml_probability >= min_ml_prob):
                      
                      # Formasyon desteği varsa güven skorunu artır
                      if pattern_support > 0:
                          signal_confidence *= (1 + pattern_support)
                          
                      logging.info(f"✅ Sinyal Onaylandı: {technical_signal['type']}\n"
                                 f"Final Güven Skoru: {signal_confidence:.2f}\n"
                                 f"{signal_details}\n{validation_details}")
                      return True
                  
              logging.info(f"❌ Sinyal Reddedildi\n{validation_details}")
          return False
      except Exception as e:
          logging.error(f"Sinyal doğrulama hatası: {e}")
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
        """Pozisyon büyüklüğünü hesapla"""
        try:
            # Bakiyeyi al
            balance = float(self.get_account_balance())
            logging.info(f"Mevcut bakiye: {balance} USDT")
        
            # Minimum işlem miktarı (örnek: 0.001 BTC için yaklaşık 0.05 USDT)
            min_trade_value = 0.05
        
            # Risk miktarını hesapla (bakiyenin %95'i)
            risk_amount = balance * 0.95
        
            # Pozisyon büyüklüğünü hesapla
            position_size = risk_amount / current_price
        
            # Minimum işlem değeri kontrolü
            if position_size * current_price < min_trade_value:
                logging.warning(f"İşlem değeri çok düşük: {position_size * current_price} USDT")
                return 0
            
            return position_size
        
        except Exception as e:
            logging.error(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0
        
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

            # Hesap bakiyesini al
            balance = float(self.get_account_balance())
            logging.info(f"Mevcut bakiye: {balance} USDT")

            # Check if balance is below 5 USD
            if balance < 0:
                logging.warning(f"Yetersiz bakiye: {balance} USDT. İşlem yapılmayacak.")
                await self.send_telegram(f"⚠️ Yetersiz bakiye: {balance} USDT. İşlem yapılmayacak.")
                return False

            # Kaldıraç ayarı
            try:
                self.client.change_leverage(
                    symbol=symbol,
                    leverage=10
                )
                logging.info(f"Kaldıraç ayarlandı: {symbol} 10x")
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

                message = (
                    f"✅ İşlem Gerçekleşti\n"
                    f"Sembol: {symbol}\n"
                    f"Yön: {trade_side}\n"
                    f"Miktar: {quantity}\n"
                    f"Fiyat: {price}\n"
                    f"İşlem Değeri: {final_notional:.2f} USDT\n"
                    f"Stop Loss: {sl_price}\n"
                    f"Take Profit: {tp_price}\n"
                    f"Kaldıraç: 5x\n"
                    f"Bakiye: {balance} USDT"
                )

                logging.info(f"İşlem başarılı: {symbol} {trade_side} {quantity}")
                await self.send_telegram(message)

                return True

            except Exception as order_error:
                logging.error(f"Order yerleştirme hatası: {order_error}")
                await self.send_telegram(f"⚠️ İşlem Hatası: {symbol} - {str(order_error)}")
                return False

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