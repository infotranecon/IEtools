"""
Information Equilibrium tools (IEtools.py) 0.1-beta

A collection of python tools to help constructing Information 
Equilibrium or Dynamic Equilibrium models. 

http://informationtransfereconomics.blogspot.com/2017/04/a-tour-of-information-equilibrium.html

"""

import csv
import datetime
import numpy as np
from scipy.optimize import curve_fit

#File readers

def FREDcsvRead(filename):
    """ Reads a FRED csv file and returns a dictionary where
    'name': string with FRED name of time series
    'data': numpy array with dates in continuous time (years) """
    csvFile = open(filename,newline='')
    outputList = []
    reader = csv.reader(csvFile)
    for row in reader:
        if row[0] == 'DATE':
            outputName = row[1]
        else:
            theDate = datetime.datetime.strptime(row[0],'%Y-%m-%d')
            nextYear = int(theDate.month/12)
            yearLength = (datetime.date(theDate.year+1,1,1) - datetime.date(theDate.year,1,1)).days
            yearToDateLength = (datetime.date(theDate.year,theDate.month,theDate.day) - datetime.date(theDate.year,1,1)).days
            outputList.append([theDate.year + yearToDateLength/yearLength, float(row[1])])
    return {'name':outputName,'data':np.array(outputList)}

#Dynamic equilibrium and entropy minimization tools

def shannon_function(p,base=np.e):
    if base <= 1:
        raise ValueError('Base must be greater than one.')
    if p == 0:
        return 0
    else:
        return 0-p*np.log(p)/np.log(base)

def log_linear_transform(timeSeries, alpha):
    """Returns a time series where {x, y(x)} becomes 
    {x, y'(x) = log y(x) + alpha x + c}"""
    startPoint = timeSeries[0,0]
    transform = map(lambda x,y: np.log(y) - alpha * (x-startPoint), timeSeries[:,0],timeSeries[:,1])
    return np.transpose([timeSeries[:,0], np.array(list(transform))])

def array_relative_entropy(array):
    """Returns the relative entropy of a numpy array compared 
    to a constant (uniform) array"""
    normalizedArray = array/array.sum()
    entropy = np.array(list(map(shannon_function, normalizedArray)))
    return entropy.sum()/np.log(normalizedArray.shape[0])

def log_linear_timeseries_entropy(timeSeries, alpha, binWidth=0.025):
    """Compute the relative entropy of a log-linear transform
    of  a log-linear transform of the time series ordinate"""
    result = np.histogram(log_linear_transform(timeSeries,alpha)[:,1],bins=round(1/binWidth))
    return array_relative_entropy(result[0])

def dynamic_equilibrium_optimize(timeSeries, alphaRange = (-0.1,0.1), alphaDelta=0.01, binWidth=0.025, method='brute'):
    """Find the log linear transformation slope alpha that minimizes the histogram
    entropy. Alpha is in alphaRange and objective function is sampled at resolution
    alphaDelta. Histogram bin width is set by binWidth. Returns the log linear slope
    in alphaRange that minimizes ordinate entropy. 
    
    Methods include:

    'brute': brute force minimum value of objective function over linspace defined
        by alphaRange and alphaDelta.
    'interpolation': minimize the interpolation of the objective function via scipy
        optimize methods. Not yet implemented.
    'localquadratic': estimate quadratic in the neighborhood of the brute force
        minimum. Not yet implemented."""
    alphas = np.linspace(alphaRange[0],alphaRange[1],num=round(1/alphaDelta))
    obj = np.array(list(map(lambda a: log_linear_timeseries_entropy(timeSeries, a, binWidth=binWidth), alphas)))
    if method == 'brute':
        result = alphas[obj.argmin()]
    elif method == 'localquadratic':
        raise NotImplementedError('Local quadratic method has not been implemented yet.')
    elif method == 'interpolation':
        raise NotImplementedError('Interpolation method has not been implemented yet.')
    else:
        raise NotImplementedError('Other methods have not been implemented yet.')
    return result

#Dynamic equilibrium functions and curve fitting

def shock(x,a,b,t):
    return a/(1 + np.exp((x-t)/b))

def dynamic_eq(x,a,b):
    return a * x + b

def one_shock_eq(x,a,b,t, alpha, c):
    return shock(x,a,b,t) + dynamic_eq(x, alpha,c)

def two_shock_eq(x,a1,b1,t1,a2,b2,t2, alpha, c):
    return shock(x,a1,b1,t1) + shock(x,a2,b2,t2) + dynamic_eq(x, alpha,c)

def one_shock(x,a,b,t, c):
    return shock(x,a,b,t) + c

def two_shock(x,a1,b1,t1,a2,b2,t2, c):
    return shock(x,a1,b1,t1) + shock(x,a2,b2,t2) + c

#def dynamic_eq_fit_entropy_min(function, timeSeries, guess):
#    popt,pcov = curve_fit(function, timeSeries[:,0], np.log(timeSeries[:,1]), p0=guess)
#    fitData = np.array(list(map(lambda x:np.exp(function(x,*popt)),timeSeries[:,0])))
#    return {'params':popt, 'cov':pcov, 'fit':fitData}

def dynamic_eq_fit(function, timeSeries, guess):
    popt,pcov = curve_fit(function, timeSeries[:,0], np.log(timeSeries[:,1]), p0=guess)
    fitData = np.array(list(map(lambda x:np.exp(function(x,*popt)),timeSeries[:,0])))
    transitions = []
    shock_widths = []
    shock_mags = []
    for index in range(len(popt)-np.mod(len(popt),3)):
        if np.mod(index,3)==2:
          transitions.append(popt[index])
        elif np.mod(index,3)==1:
          shock_widths.append(np.abs(popt[index]))
        elif np.mod(index,3)==0:
            shock_mags.append(-popt[index]*np.sign(popt[index+1]))
        else:
            raise ValueError('How did you get a postive integer modulo 3 that was not 0, 1, or 2?')  
    return {'params':popt, 'cov':pcov, 'fit':fitData, 'transitions':transitions, 'shock_widths':shock_widths,'shock_mags':shock_mags}
