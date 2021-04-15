
import pandas as pd
import yfinance as yf
import time
import datetime
from pygame import mixer  # Load the popular external library
from playsound import playsound
import sys

#For linux color output
from termcolor import cprint
#For windows color output
import colorama
colorama.init()

class ScriptInfo:
    def __init__(self):
        self.time = 0
        self.support = 0
        self.resistance = 0
        self.prev_volume = 0
        self.support_found = False
        self.resistance_found = False
        self.buyers_trapping_candle_found = False
        self.sellers_trapping_candle_found = False


#Filter trapping candles
class StrategyChecker:
    def __init__(self):
        self.is_started = False
        self.scripts = {}
        nifty_tickers=pd.read_csv('Nifty_yahoo_ticker.csv')
        # self.tickers_list=nifty_tickers['Yahoo_Symbol'].tolist()
        self.tickers_list = ['7267.T','6758.T','7203.T','7269.T','6330.T']

    def check_for_trapping_candle(self,script,volume_traded,open_value,close_value,high_value,low_value,ticker):
        if(volume_traded > script.prev_volume):
            risk = high_value - low_value
            if(open_value > close_value):
                # Sellers trapping candle
                if(not script.sellers_trapping_candle_found):
                    script.sellers_trapping_candle_found = True
                    cprint(f"************ Sellers Trapping Candle Found for {ticker}************ Risk :{risk}",'red')
                    playsound('alert_tone.mp3')
            else:
                # Buyers trapping candle
                if(not script.buyers_trapping_candle_found):
                    script.buyers_trapping_candle_found = True
                    print(f"************ Buyers Trapping Candle Found for {ticker}************ Risk :{risk}",'green')
                    playsound('alert_tone.mp3')

    def calculate_support_resistance(self,script,high_value,low_value):
        
        if(high_value > script.resistance):
            script.resistance = high_value
            # print("Resistance found")
        if(low_value < script.support or script.support==0):
            script.support = low_value
            # print("Suppory found")

    def start_filter(self):
        while True :
            # for ticker in  self.tickers_list:
            # full_data = yf.download(tickers=ticker, period='1d', interval='5m')
            # for index, data in full_data.iterrows():
            now = datetime.datetime.now()
            market_opne_time = now.replace(hour=9,minute=25,second=0)#India
            # market_opne_time = now.replace(hour=19,minute=10,second=0) # us 
            # if 1:
            if (now > market_opne_time):
                # for index, data in full_data.iterrows():
                for ticker in  self.tickers_list:
                    # for index, data in full_data.iterrows():
                    script = None
                    if (ticker in self.scripts.keys()):
                        script = self.scripts[ticker]
                    else:
                        script = ScriptInfo()
                        self.scripts[ticker] = script
                    data = yf.download(tickers=ticker, period='1d', interval='5m',progress=False)
                    #print(data['Name'])
                    #Taking last 5 min data with volume
                    if(len(data) < 1 or (len(data) == 2) and data['Volume'][1] == 0):
                        continue
                    if(data['Volume'][len(data) -1] != 0):
                        data = data.iloc[len(data) -1]
                    else:
                        data = data.iloc[len(data) -2]
                    if(script.time == data.name):
                        continue
                    
                    # print(ticker,data, end='\r')
                    open_value =data['Open']
                    high_value = data['High']
                    low_value = data['Low']
                    close_value = data['Close']
                    volume_traded = data['Volume']


                    if (script.prev_volume == 0 ):
                        script.prev_volume = volume_traded

                    if(script.resistance < high_value or script.support > low_value ):
                        script.support_found = False
                        script.resistance_found = False
                        script.sellers_trapping_candle_found = False
                        script.buyers_trapping_candle_found = False
                        self.calculate_support_resistance(script,high_value,low_value)

                    if(script.support_found and script.resistance_found):
                        self.check_for_trapping_candle(script,volume_traded,open_value,close_value,high_value,low_value,ticker)

                    elif(volume_traded > script.prev_volume):
                        if(open_value < close_value and not script.support_found):
                            script.support_found = True
                        elif(open_value > close_value and not script.resistance_found):
                            script.resistance_found = True
                    
                    script.time = data.name # date time 
                    script.prev_volume = volume_traded


            time.sleep(1)


if __name__ == "__main__":
    strategy_checker = StrategyChecker()
    strategy_checker.start_filter()