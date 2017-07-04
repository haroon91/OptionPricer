from math import exp, log, pi, sqrt
import math as math
import scipy.stats as stats
import scipy as scipy
import numpy as numpy

""" --------------- European Options - Black-Scholes ---------------"""


def callValue(S, K, v, r, q, T, t):

    d1 = calculateD1(S, K, T, v, r, q, t)
    d2 = calculateD2(d1, v, T, t)

    call_value = (S * exp(-q * (T-t)) * stats.norm.cdf(d1)) - (K * exp(-r * (T-t)) * stats.norm.cdf(d2))
    return float(call_value)

def putValue(S, K, v, r, q, T, t):

    d1 = calculateD1(S, K, T, v, r, q, t)
    d2 = calculateD2(d1, v, T, t)

    put_value = (K * exp(-r * (T-t)) * stats.norm.cdf(-d2)) - (S * exp(-q * (T-t)) * stats.norm.cdf(-d1))
    return float(put_value)


""" --------------- American Options - Binomial Tree ---------------"""


def binomialTreeOptionValue(S,K,v,r,T,N,option_type):
    dt = 1.0 * T / N;
    u = exp(v * sqrt(dt))
    d = 1/u
    p = (exp(r*dt) - d) / (u - d);

    stockPriceTree = numpy.zeros((N+1, N+1))
    optionTree = numpy.zeros((N+1, N+1))

    # build the stock price tree
    stockPriceTree[0,0] = S

    for i in range(1,N+1,1):
        stockPriceTree[i,0] = stockPriceTree[i-1,0] * u
        for j in range(1,i+1,1):
            stockPriceTree[i,j] = stockPriceTree[i-1,j-1] * d

    # build the option tree
    for j in range(N+1):
        optionTree[N,j] = max(0,option_type*(stockPriceTree[N,j]-K))

    for i in range(N-1,-1,-1):
        for j in range(i+1):
            optionTree[i,j] = (p * optionTree[i+1,j] + (1-p) * optionTree[i+1,j+1]) * exp(-r*dt)
            optionTree[i,j] = max(optionTree[i,j], option_type*(stockPriceTree[i,j]-K))

    return optionTree[0,0]


""" ---------------- Utility Functions ---------------- """


def sigmaHat(S, K, r, q, T, t):

    sigma_hat = sqrt(2 * abs((log(S/K) + (r-q)*(T-t)) / (T-t)))
    return float(sigma_hat)


def callDelta(S, K, v, r, q, T, t):
    d1 = calculateD1(S, K, T, v, r, q, t)

    call_Delta = S * exp(-q * (T-t)) * stats.norm.cdf(d1)
    return call_Delta


def putDelta(S, K, v, r, q, T, t):
    d1 = calculateD1(S, K, T, v, r, q, t)

    put_Delta = S * exp(-q * (T-t)) * (stats.norm.cdf(d1) - 1)
    return put_Delta


def callOrPutVega (S, K, v, r, q, T, t):
    d1 = calculateD1(S, K, T, v, r, q, t)

    vega = float(S * exp(-q * (T-t)) * sqrt(T-t) * exp(-d1**2/2) / sqrt(2*pi))
    return vega


def calculateD1 (S, K, T, v, r, q, t):
    return float((log(1.0 * S/K) + ((r-q) + v**2/2) * (T-t)) / (v*sqrt(T-t)))


def calculateD2 (d1, v, T, t):
    return float(d1 - v * sqrt(T-t))


""" ---------------- Implied Volatilty - Newton Raphson Method ---------------- """


def impliedVolatilityCall (S, K, T, t, q, r, C_true):

    # initialize all values
    sigma = sigmaHat(S, K, r, q, T, t)
    CVega = callOrPutVega(S, K, sigma, r, q, T, t)

    call_Value = callValue(S,K,sigma,r,q,T,t)

    tol = 1e-8
    sigmaDiff = 1
    n = 1
    nmax = 100

    while (sigmaDiff >= tol and n < nmax) :
        increment = (call_Value-C_true) / CVega
        sigma = sigma - increment
        n = n+1
        sigmaDiff = abs(increment)

        call_Value = callValue(S, K, sigma, r, q, T, t)
        CVega = callOrPutVega(S, K, sigma, r, q, T, t)

    return sigma

def impliedVolatilityPut (S, K, T, t, q, r, P_true):

    # initialize all values
    sigma = sigmaHat(S, K, r, q, T, t)
    PVega = callOrPutVega(S, K, sigma, r, q, T, t)

    put_Value = putValue(S,K,sigma,r,q,T,t)

    tol = 1e-8
    sigmaDiff = 1
    n = 1
    nmax = 100

    while (sigmaDiff >= tol and n < nmax):
        increment = (put_Value - P_true) / PVega
        sigma = sigma - increment
        n = n+1
        sigmaDiff = abs(increment)

        put_Value = putValue(S,K,sigma,r,q,T,t)
        PVega = callOrPutVega(S,K,sigma,r,q,T,t)

    return sigma


""" --------------- Asian Options - Geometric (Closed-form) and Arithmetic (Monte Carlo) ---------------"""


def sigmaHatGeo (v, N):

    sigHatGeo = v * sqrt( (N+1.0)*(2.0*N+1.0) / (6.0*N*N) )
    return sigHatGeo

def geometricAsianCallValue (S0, K, v, r, T, N):

    Dt = T/N
    sigmaHatGeoSq = sigmaHatGeo(v, N) ** 2
    mUHat = (r - (0.5 * v * v)) * ((N+1.0)/(2.0*N)) + (0.5 * sigmaHatGeoSq)

    geoD1 = calculateD1(S0, K, T, sqrt(sigmaHatGeoSq), mUHat, 0, 0)
    geoD2 = calculateD2(geoD1, sqrt(sigmaHatGeoSq), T, 0)

    geoCallValue = exp(-r*T) * ( (S0 * exp(mUHat * T) * stats.norm.cdf(geoD1)) - (K * stats.norm.cdf(geoD2)) )
    return geoCallValue

def geometricAsianPutValue (S0, K, v, r, T, N):

    Dt = T/N
    sigmaHatGeoSq = sigmaHatGeo(v, N) ** 2
    mUHat = (r - (0.5 * v * v)) * ((N+1.0)/(2.0*N)) + (0.5 * sigmaHatGeoSq)

    geoD1 = calculateD1(S0, K, T, sqrt(sigmaHatGeoSq), mUHat, 0, 0)
    geoD2 = calculateD2(geoD1, sqrt(sigmaHatGeoSq), T, 0)

    geoPutValue = exp(-r*T) * ( (K * stats.norm.cdf(-geoD2)) - (S0 * exp(mUHat*T) * stats.norm.cdf(-geoD1)) )
    return geoPutValue

def arithmeticAsianCallValue (S0, K, v, r, T, N, M, controlVariate, callOrPut):

    dt = T*1.0/N
    drift = exp((r-0.5*v*v)*dt)

    Spath = numpy.empty(N, dtype=float)
    arithPayOff = numpy.empty(M, dtype=float)
    geoPayOff = numpy.empty(M, dtype=float)

    scipy.random.seed([100])

    for i in range(0,M,1):
        growthFactor = drift * exp(v*sqrt(dt)*scipy.random.randn(1))
        Spath[0] = S0 * growthFactor
        for j in range(1,N,1):
            growthFactor = drift * exp(v*sqrt(dt)*scipy.random.randn(1))
            Spath[j] = Spath[j-1] * growthFactor

        # Arithmetic mean
        arithMean = numpy.mean(Spath)
        arithPayOff[i] = exp(-r*T)* max(callOrPut*(arithMean-K), 0)

        # Geometric mean
        logs = numpy.log(Spath)
        sumOfLogs = numpy.sum(logs)

        geoMean = exp((1.0/N) * sumOfLogs)
        geoPayOff[i] = exp(-r*T) * max(callOrPut*(geoMean-K),0)

    if controlVariate == True:
        # Control variates
        covXY = numpy.mean(arithPayOff * geoPayOff) - numpy.mean(arithPayOff) * numpy.mean(geoPayOff)
        theta = covXY / numpy.var(geoPayOff)

        # Control variate version
        if callOrPut == 1:
            geo = geometricAsianCallValue(S0, K, v, r, T, N)
        else:
            geo = geometricAsianPutValue(S0, K, v, r, T, N)

        Z = arithPayOff + theta * (geo - geoPayOff)
        Zmean = numpy.mean(Z)
        Zstd = numpy.std(Z)

        confcv = [Zmean - 1.96 * Zstd / sqrt(M), Zmean + 1.96 * Zstd / sqrt(M)]

        finalvalue = numpy.mean(confcv)

    else:
        # Standard Monte Carlo
        Pmean = numpy.mean(arithPayOff)
        Pstd = numpy.std(arithPayOff)

        confmc = [Pmean-1.96*Pstd/sqrt(M), Pmean+1.96*Pstd/sqrt(M)]

        finalvalue = numpy.mean(confmc)

    return finalvalue


""" --------------- Geometric Mean Basket ---------------"""


def geometricBasketOptionValue(S1, S2, v1, v2, r, T, K, corr, callOrPut):

    sigmaBg = 0.5 * sqrt(v1*v1 + v2*v2 + 2*v1*v2*corr)

    muBg = r - 0.25*(v1*v1 + v2*v2) + 0.5*sigmaBg*sigmaBg

    Bg0 = sqrt(S1*S2)

    basketD1 = ( log(Bg0/K) + (muBg + 0.50*sigmaBg*sigmaBg) * T ) / (sigmaBg * sqrt(T))

    basketD2 = float (basketD1 - sigmaBg*sqrt(T))

    optionPrice = 0.0
    if str(callOrPut).lower() == 'call':
        optionPrice = exp(-r*T) * (Bg0 * exp(muBg*T) * stats.norm.cdf(basketD1) - K*stats.norm.cdf(basketD2))
    elif str(callOrPut).lower() == 'put':
        optionPrice = exp(-r*T) * (K*stats.norm.cdf(-basketD2) - Bg0*exp(muBg*T)*stats.norm.cdf(-basketD1))
    else:
        raise Exception

    return optionPrice

def arithmeticBasketOptionValue(S1, S2, v1, v2, r, T, K, corr, callOrPut, M, controlVariate):

    dt = T * 1.0
    drift1 = exp((r-0.5*v1*v1)*dt)
    drift2 = exp((r-0.5*v2*v2)*dt)

    S1next = 0.0
    S2next = 0.0
    arithPayOff = numpy.empty(M, dtype=float)
    geoPayOff = numpy.empty(M, dtype=float)

    scipy.random.seed([100])

    for i in range(0,M,1):
        Rand1 = scipy.random.randn(1)
        Rand2 = scipy.random.randn(1)
        growthFactor1 = drift1 * exp(v1 * sqrt(dt) * Rand1)
        S1next = S1 * growthFactor1
        growthFactor2 = drift2 * exp(v2 * sqrt(dt) * (corr * Rand1 + sqrt(1-corr*corr) * Rand2))
        S2next = S2 * growthFactor2

        # Arithmetic mean
        arithMean = 0.5 * (S1next+S2next)
        arithPayOff[i] = exp(-r*T) * max(callOrPut*(arithMean-K), 0)

        # Geometric mean
        S1log = numpy.log(S1next)
        S2log = numpy.log(S2next)
        sumOfLogs = S1log + S2log
        geoMean = exp(0.5 * sumOfLogs)
        geoPayOff[i] = exp(-r*T) * max(callOrPut*(geoMean-K), 0)

    if (controlVariate == True):
        # Control Variate
        covXY = numpy.mean(arithPayOff * geoPayOff) - numpy.mean(arithPayOff) * numpy.mean(geoPayOff)
        theta = covXY / numpy.var(geoPayOff)

        if callOrPut == 1:
            geo = geometricBasketOptionValue(S1, S2, v1, v2, r, T, K, corr, 'call')
        else :
            geo = geometricBasketOptionValue(S1, S2, v1, v2, r, T, K, corr, 'put')

        Z = arithPayOff + theta * (geo - geoPayOff)
        Zmean = numpy.mean(Z)
        Zstd = numpy.std(Z)

        confcv = [Zmean - 1.96 * Zstd / sqrt(M), Zmean + 1.96 * Zstd / sqrt(M)]

        finalvalue = numpy.mean(confcv)

    else:
        # Standard monte carlo
        Pmean = numpy.mean(arithPayOff)
        Pstd = numpy.std(arithPayOff)

        confmc = [Pmean - 1.96*Pstd/sqrt(M), Pmean + 1.96*Pstd/sqrt(M)]

        finalvalue = numpy.mean(confmc)

    return finalvalue


""" -------------------- END --------------------"""