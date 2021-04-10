import pandas as pd
import logging
log = logging.getLogger('creditcard')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').disabled = True
import yfinance as yf
import time
import datetime
from pygame import mixer  # Load the popular external library
from playsound import playsound


support = 0
resistance = 0
prev_volume = 0
support_found = False
resistance_found = False
trapping_candle_found = False


def check_for_trapping_candle(volume_traded,open_value,close_value,high_value,low_value):
    global trapping_candle_found
    if(volume_traded >= prev_volume):
        # Sellers trapping candle
        if(open_value < close_value):
            
            risk = high_value - low_value
            # if(risk < 5):
            trapping_candle_found = True
            print("***************** Sellers Trapping Candle Found**************** Risk :",high_value,low_value,risk)
            playsound('alert_tone.mp3')

def calculate_support_resistance(high_value,low_value):
    global resistance,support
    if(high_value > resistance):
        resistance = high_value
    if(low_value < support or low_value==0):
        support = low_value

while True :
    # full_data = yf.download(tickers='AAPL', period='1d', interval='5m')
    # for index, data in full_data.iterrows():
    now = datetime.datetime.now()
    # market_opne_time = now.replace(hour=9,minute=16,second=0)
    market_opne_time = now.replace(hour=19,minute=10,second=0) # us 
    if 1:
    # if (now > market_opne_time):
        # for index, data in full_data.iterrows():
        # print("Market opened")
        data = yf.download(tickers='AAPL', period='5m', interval='5m')
        open_value =data['Open'][0]
        high_value = data['High'][0]
        low_value = data['Low'][0]
        close_value = data['Close'][0]
        volume_traded = data['Volume'][0]
        # print(data)
        # open_value =data['Open']
        # high_value = data['High']
        # low_value = data['Low']
        # close_value = data['Close']
        # volume_traded = data['Volume']

        if (prev_volume == 0 ):
            prev_volume = volume_traded

        if(resistance < high_value or support > low_value ):
            support_found = False
            resistance_found = False
            trapping_candle_found = False
            calculate_support_resistance(high_value,low_value)

        if(support_found and resistance_found):
            check_for_trapping_candle(volume_traded,open_value,close_value,high_value,low_value)

        elif(volume_traded > prev_volume):
            if(open_value <= close_value):
                support_found = True
            else:
                resistance_found = True
        
        prev_volume = volume_traded
    else:
        now = datetime.datetime.now()
    time.sleep(1)