import formula_functions as formulas
from Tkinter import *
import pygubu

class App:
    def __init__(self, master):
        self.master = master

        self.builder = builder = pygubu.Builder()
        # this is to load the ui file
        builder.add_from_file('ass3_UI.ui')
        # add windows here
        self.mainWindow = builder.get_object('mainWindow', master)
        # callbacks
        builder.connect_callbacks(self)

    def open_e_option_window(self):
        self.mainWindow = self.builder.get_object('eOptionWindow', self.master)
        self.builder.connect_callbacks(self)

    def calculate_eOptionPrice(self):

        try:
            S = self.builder.tkvariables['entry_assetPrice'].get()
            v = self.builder.tkvariables['entry_volatility'].get() / 100.0
            r = self.builder.tkvariables['entry_riskFreeRate'].get() / 100.0
            q = self.builder.tkvariables['entry_repoRate'].get() / 100.0
            T = self.builder.tkvariables['entry_timetoM'].get()
            K = self.builder.tkvariables['entry_strikePrice'].get()
            putOrCall = self.builder.tkvariables['optionType'].get()

            if (S <= 0 or K <= 0):
                self.builder.tkvariables['optionPrice'].set(
                    'Unable to compute because of invalid or incomplete entries. Please try again')
                return

            optionPriceLabel = self.builder.tkvariables['optionPrice'].get()
            optionPriceLabel = str(optionPriceLabel) + ' ...Calculating...'

            self.builder.tkvariables['optionPrice'].set(optionPriceLabel)

            putStr = 'put'
            callStr = 'call'
            optionPrice = 'NaN'
            if (str(putOrCall).lower() == putStr.lower()):
                optionPrice = formulas.putValue(S,K,v,r,q,T,0)
            elif (str(putOrCall).lower() == callStr.lower()):
                optionPrice = formulas.callValue(S,K,v,r,q,T,0)
            else:
                raise Exception

            self.builder.tkvariables['optionPrice'].set('Option Price = ' + str(round(optionPrice, 5)))


        except Exception:
            self.builder.tkvariables['optionPrice'].set(
                'Unable to compute because of invalid or incomplete entries. Please try again')

    def open_impV_window(self):
        self.mainWindow = self.builder.get_object('impVolWindow', self.master)
        self.builder.connect_callbacks(self)

    def calculate_impVol(self):

        try:
            S = self.builder.tkvariables['entry_assetPrice'].get()
            r = self.builder.tkvariables['entry_riskFreeRate'].get() / 100.0
            q = self.builder.tkvariables['entry_repoRate'].get() / 100.0
            T = self.builder.tkvariables['entry_timetoM'].get()
            K = self.builder.tkvariables['entry_strikePrice'].get()
            Op_true = self.builder.tkvariables['entry_optionPremium'].get()
            putOrCall = self.builder.tkvariables['optionType'].get()

            impVPriceLabel = self.builder.tkvariables['impV'].get()
            impVPriceLabel = str(impVPriceLabel) + ' ...Calculating...'

            self.builder.tkvariables['impV'].set(impVPriceLabel)

            if (S <= 0 or K <= 0 or Op_true < 0):
                self.builder.tkvariables['impV'].set(
                    'Unable to compute because of invalid or incomplete entries. Please try again')
                return

            putStr = 'put'
            callStr = 'call'
            impV = 'NaN'

            if (str(putOrCall).lower() == putStr.lower()):
                impV = formulas.impliedVolatilityPut(S,K,T,0,q,r,Op_true)

            elif (str(putOrCall).lower() == callStr.lower()):
                impV = formulas.impliedVolatilityCall(S,K,T,0,q,r,Op_true)

            else:
                raise Exception

            self.builder.tkvariables['impV'].set('Implied Volatility = ' + str(round(impV, 5)))

        except Exception:
            self.builder.tkvariables['impV'].set(
                'Unable to compute because of invalid or incomplete entries. Please try again')


    def open_a_option_window(self):
        self.mainWindow = self.builder.get_object('amOptionWindow', self.master)
        self.builder.connect_callbacks(self)

    def calculate_amOptionPrice(self):

        try:
            S = self.builder.tkvariables['entry_assetPrice'].get()
            v = self.builder.tkvariables['entry_volatility'].get() / 100.0
            r = self.builder.tkvariables['entry_riskFreeRate'].get() / 100.0
            T = self.builder.tkvariables['entry_timetoM'].get()
            K = self.builder.tkvariables['entry_strikePrice'].get()
            N = self.builder.tkvariables['entry_noOfSteps'].get()
            putOrCall = self.builder.tkvariables['optionType'].get()

            optionPriceLabel = self.builder.tkvariables['optionPrice'].get()
            optionPriceLabel = str(optionPriceLabel) + ' ...Calculating...'

            self.builder.tkvariables['optionPrice'].set(optionPriceLabel)

            if (S <= 0 or K <= 0):
                self.builder.tkvariables['optionPrice'].set(
                    'Unable to compute because of invalid or incomplete entries. Please try again')
                return

            putStr = 'put'
            callStr = 'call'
            optionPrice = 'NaN'
            if str(putOrCall).lower() == putStr.lower():
                optionPrice = formulas.binomialTreeOptionValue(S,K,v,r,T,N,-1)
            elif str(putOrCall).lower() == callStr.lower():
                optionPrice = formulas.binomialTreeOptionValue(S,K,v,r,T,N,1)
            else:
                raise Exception

            self.builder.tkvariables['optionPrice'].set('Option Price = ' + str(round(optionPrice, 5)))

        except Exception:
            self.builder.tkvariables['optionPrice'].set(
                'Unable to compute because of invalid or incomplete entries. Please try again')


    def open_geo_asian_option_window(self):
        self.mainWindow = self.builder.get_object('geoAsianOptionWindow', self.master)
        self.builder.connect_callbacks(self)

    def calculate_geoAsianOptionPrice(self):

        try:
            S = self.builder.tkvariables['entry_assetPrice'].get()
            v = self.builder.tkvariables['entry_volatility'].get() / 100.0
            r = self.builder.tkvariables['entry_riskFreeRate'].get() / 100.0
            T = self.builder.tkvariables['entry_timetoM'].get()
            K = self.builder.tkvariables['entry_strikePrice'].get()
            N = self.builder.tkvariables['entry_noOfObservations'].get()
            putOrCall = self.builder.tkvariables['optionType'].get()

            optionPriceLabel = self.builder.tkvariables['optionPrice'].get()
            optionPriceLabel = str(optionPriceLabel) + ' ...Calculating...'

            self.builder.tkvariables['optionPrice'].set(optionPriceLabel)

            if (S <= 0 or K <= 0 or v < 0 or T < 0):
                self.builder.tkvariables['optionPrice'].set(
                    'Unable to compute because of invalid or incomplete entries. Please try again')
                return

            putStr = 'put'
            callStr = 'call'
            optionPrice = 'NaN'
            if str(putOrCall).lower() == putStr.lower():
                optionPrice = formulas.geometricAsianPutValue(S,K,v,r,T,N)
            elif str(putOrCall).lower() == callStr.lower():
                optionPrice = formulas.geometricAsianCallValue(S,K,v,r,T,N)
            else:
                raise Exception

            self.builder.tkvariables['optionPrice'].set('Option Price = ' + str(round(optionPrice, 5)))

        except Exception:
            self.builder.tkvariables['optionPrice'].set(
                'Unable to compute because of invalid or incomplete entries. Please try again')

    def open_geo_basket_option_window(self):
        self.mainWindow = self.builder.get_object('geoBasketOptionWindow', self.master)
        self.builder.connect_callbacks(self)

    def calculate_geoBasketOptionPrice(self):

        try:
            S1 = self.builder.tkvariables['entry_assetPrice1'].get()
            S2 = self.builder.tkvariables['entry_assetPrice2'].get()
            v1 = self.builder.tkvariables['entry_assetVolatility1'].get() / 100.0
            v2 = self.builder.tkvariables['entry_assetVolatility2'].get() / 100.0
            r = self.builder.tkvariables['entry_riskFreeRate'].get() / 100.0
            T = self.builder.tkvariables['entry_timetoM'].get()
            K = self.builder.tkvariables['entry_strikePrice'].get()
            corr = self.builder.tkvariables['entry_correlation'].get() * 1.0
            putOrCall = self.builder.tkvariables['optionType'].get()

            optionPriceLabel = self.builder.tkvariables['optionPrice'].get()
            optionPriceLabel = str(optionPriceLabel) + ' ...Calculating...'

            self.builder.tkvariables['optionPrice'].set(optionPriceLabel)


            if (S <= 0 or K <= 0 or v1 < 0 or v2 < 0 or T < 0 or corr > 1 or corr < -1):
                self.builder.tkvariables['optionPrice'].set('Unable to compute because of invalid or incomplete entries\n Please try again')
                return

            optionPrice = formulas.geometricBasketOptionValue(S1, S2, v1, v2, r, T, K, corr, putOrCall)
            self.builder.tkvariables['optionPrice'].set('Option Price = ' + str(round(optionPrice,5)))
        except Exception:
            self.builder.tkvariables['optionPrice'].set('Unable to compute because of invalid or incomplete entries. Please try again')

    def open_arith_asian_option_window(self):
        self.mainWindow = self.builder.get_object('arithAsianOptionWindow', self.master)
        self.builder.connect_callbacks(self)

    def calculate_arithAsianOptionPrice(self):

        try:
            S = self.builder.tkvariables['entry_assetPrice'].get()
            v = self.builder.tkvariables['entry_volatility'].get() / 100.0
            r = self.builder.tkvariables['entry_riskFreeRate'].get() / 100.0
            T = self.builder.tkvariables['entry_timetoM'].get()
            K = self.builder.tkvariables['entry_strikePrice'].get()
            N = self.builder.tkvariables['entry_noOfObservations'].get()
            M = self.builder.tkvariables['entry_noOfSteps'].get()
            controlVariate = self.builder.tkvariables['entry_controlVariate'].get()
            putOrCall = self.builder.tkvariables['entry_optionType'].get()

            optionPriceLabel = self.builder.tkvariables['optionPrice'].get()
            optionPriceLabel = str(optionPriceLabel) + ' ...Calculating...'

            self.builder.tkvariables['optionPrice'].set(optionPriceLabel)

            if (S <= 0 or K <= 0 or v < 0 or T < 0):
                self.builder.tkvariables['optionPrice'].set(
                    'Unable to compute because of invalid or incomplete entries\n Please try again')
                return

            putStr = 'put'
            callStr = 'call'
            optionPrice = 'NaN'
            controlVariateValue = False

            if str(controlVariate).lower() == 'True'.lower():
                controlVariateValue = True
            elif str(controlVariate).lower() == 'False'.lower():
                controlVariateValue = False
            else:
                raise Exception

            if str(putOrCall).lower() == putStr.lower():
                optionPrice = formulas.arithmeticAsianCallValue(S, K, v, r, T, N, M, controlVariateValue, -1)
            elif str(putOrCall).lower() == callStr.lower():
                optionPrice = formulas.arithmeticAsianCallValue(S, K, v, r, T, N, M, controlVariateValue, 1)
            else:
                raise Exception

            optionPrice = round(optionPrice, 5)
            self.builder.tkvariables['optionPrice'].set('Option Price = ' + str(optionPrice))

        except Exception:
            self.builder.tkvariables['optionPrice'].set(
                'Unable to compute because of invalid or incomplete entries\n Please try again')

    def open_arith_basket_option_window(self):
        self.mainWindow = self.builder.get_object('arithBasketOptionWindow', self.master)
        self.builder.connect_callbacks(self)

    def calculate_arithBasketOptionPrice(self):
        try:
            S1 = self.builder.tkvariables['entry_assetPrice1'].get()
            S2 = self.builder.tkvariables['entry_assetPrice2'].get()
            v1 = self.builder.tkvariables['entry_assetVolatility1'].get() / 100.0
            v2 = self.builder.tkvariables['entry_assetVolatility2'].get() / 100.0
            r = self.builder.tkvariables['entry_riskFreeRate'].get() / 100.0
            T = self.builder.tkvariables['entry_timetoM'].get()
            K = self.builder.tkvariables['entry_strikePrice'].get()
            corr = self.builder.tkvariables['entry_correlation'].get()
            M = self.builder.tkvariables['entry_noOfSteps'].get()
            controlVariate = self.builder.tkvariables['entry_controlVariate'].get()
            putOrCall = self.builder.tkvariables['entry_optionType'].get()

            optionPriceLabel = self.builder.tkvariables['optionPrice'].get()
            optionPriceLabel = str(optionPriceLabel) + ' ...Calculating...'

            self.builder.tkvariables['optionPrice'].set(optionPriceLabel)

            if (S <= 0 or K <= 0 or v1 < 0 or v2 < 0 or T < 0):
                self.builder.tkvariables['optionPrice'].set(
                    'Unable to compute because of invalid or incomplete entries\n Please try again')
                return

            putStr = 'put'
            callStr = 'call'
            optionPrice = 'NaN'
            controlVariateValue = False

            if str(controlVariate).lower() == 'True'.lower():
                controlVariateValue = True
            elif str(controlVariate).lower() == 'False'.lower():
                controlVariateValue = False
            else:
                raise Exception

            if str(putOrCall).lower() == putStr.lower():
                optionPrice = formulas.arithmeticBasketOptionValue(S1,S2,v1,v2,r,T,K,corr,-1,M,controlVariateValue)
            elif str(putOrCall).lower() == callStr.lower():
                optionPrice = formulas.arithmeticBasketOptionValue(S1,S2,v1,v2,r,T,K,corr,1,M,controlVariateValue)
            else:
                raise Exception

            optionPrice = round(optionPrice, 5)
            self.builder.tkvariables['optionPrice'].set('Option Price = ' + str(optionPrice))

        except Exception:
            self.builder.tkvariables['optionPrice'].set(
                'Unable to compute because of invalid or incomplete entries\n Please try again')


if __name__ == '__main__':
    root = Tk()
    app = App(root)

    root.mainloop()