"""
Information Equilibrium tools (IEtools.py) 0.11-beta

A collection of python tools to help constructing Information 
Equilibrium or Dynamic Equilibrium models. 

http://informationtransfereconomics.blogspot.com/2017/04/a-tour-of-information-equilibrium.html

Imports are installed as part of Anaconda 4.4 (Python 3.6)

beta versions
0.1	Original file
0.11	Added FRED xls support
0.12	Added new interpolations and growth rates as part of data import

"""

#These packages are installed as part of Anaconda 3

import csv
import datetime
import numpy as np
import xlrd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from scipy.misc import derivative

#File readers

#TODO: Probably want to make the file object a class and turn these into methods based on extension.


def FREDcsvRead(filename, growth=True):
    """ Reads a FRED csv file and returns a dictionary where
    'name': string with FRED name of time series
    'data': numpy array with dates in continuous time (years) 
    'interp': a linear interpolating function for the data points
    'growth': a linear interpolating function of the annualized 
              continuously comp growth rate in percent"""
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
    dataOutput =  np.array(outputList)
    interpFunction = interp1d(dataOutput[:,0],dataOutput[:,1],bounds_error = False)
    if growth:
    	der=derivative(lambda t: 100*np.log(interpFunction(t)),interpFunction.x,dx=1e-6)
    	growthFunction = interp1d(interpFunction.x,der, bounds_error = False)
    else:
    	growthFunction = 'disabled'
    return {'name':outputName,'data':dataOutput,'interp':interpFunction,'growth':growthFunction}


def FREDxlsRead(filename,growth=True):
    """ Reads a FRED xls file and returns a dictionary where
    'name':   string with FRED name of time series
    'data':   numpy array with dates in continuous time (years) 
    'interp': a linear interpolating function for the data points
    'growth': a linear interpolating function of the annualized 
              continuously comp growth rate in percent
	      note: disabled with option growth=False"""
    book = xlrd.open_workbook(filename)
    sheet = book.sheet_by_index(0)
    outputName = sheet.cell(10,1).value
    outputList = []
    for rowIndex in range(sheet.nrows-11):
        theDate = datetime.date(*xlrd.xldate_as_tuple(sheet.cell(rowIndex+11,0).value,book.datemode)[0:3])
        nextYear = int(theDate.month/12)
        yearLength = (datetime.date(theDate.year+1,1,1) - datetime.date(theDate.year,1,1)).days
        yearToDateLength = (datetime.date(theDate.year,theDate.month,theDate.day) - datetime.date(theDate.year,1,1)).days
        outputList.append([theDate.year + yearToDateLength/yearLength, sheet.cell(rowIndex+11,1).value])
    dataOutput =  np.array(outputList)
    interpFunction = interp1d(dataOutput[:,0],dataOutput[:,1],bounds_error = False)
    if growth:
    	der=derivative(lambda t: 100*np.log(interpFunction(t)),interpFunction.x,dx=1e-6)
    	growthFunction = interp1d(interpFunction.x,der, bounds_error = False)
    else:
    	growthFunction = 'disabled'
    return {'name':outputName,'data':dataOutput,'interp':interpFunction,'growth':growthFunction}

#Information equilibrium parameter fits

def objectiveFunctionLogGIE(params, source, destination, minyear, maxyear, delta):
    times = np.linspace(minyear,maxyear,num=int(np.round((maxyear-minyear)/delta)),endpoint=True)
    result = np.sum(list(map(lambda t: np.abs(np.log(source(t))- (params[0]*np.log(destination(t)) + params[1])),times)))
    return result

def objectiveFunctionGIE(params, source, destination, minyear, maxyear, delta):
    times = np.linspace(minyear,maxyear,num=int(np.round((maxyear-minyear)/delta)),endpoint=True)
    result = np.sum(list(map(lambda t: np.abs(source(t)- (params[1]*(destination(t)**params[0]))),times)))
    return result

def fitGeneralInfoEq(sourceData,destinationData, guess, minyear=0,maxyear=0,delta=0,fitlog=True, method = 'SLSQP'):
    """
    Operates on numpy array of [time, value].
    
    Fits the general information equilibrium function x = a*y**k 
    by minimizing the difference |x - a*y**k|. Returns the scipy
    minimization result. Information 'source' is x, and 'destination'
    is y.
    
    fitlog=True tells the objective function to take the logarithm of the data
    This is useful for many exponentially growing macroeconomic observables and
    makes the minimization more stable.
    
    Note: fitlog changes the meaning of the guess and result parameters. If
    fitlog is true, the optimzation returns (and the guess format is)
    
    [k, c] where log(x) = k log(y) + c
    
    If fitlog is false, the optimization returns (and the guess format is):
    
    [k, a] where x = a y^k
    
    These parameters are obtained via scipy minimize result. If
    
    result = fitGeneralInfoEq(source, destination, guess = [1, 0])
    
    then result.x is the fit parameters [k, c].
    """
    if minyear == 0:
        minyear = max(sourceData[0,0],destinationData[0,0])
    if maxyear == 0:
        maxyear = min(sourceData[-1,0],destinationData[-1,0])
    if delta == 0:
        delta1 = np.mean(np.diff(sourceData[:,0],1))
        delta2 = np.mean(np.diff(destinationData[:,0],1))
        delta = max(delta1,delta2)
    sourceInterp=interp1d(sourceData[:,0],sourceData[:,1], bounds_error=False)
    destinationInterp=interp1d(destinationData[:,0],destinationData[:,1],bounds_error=False)
    arguments = (sourceInterp, destinationInterp, minyear, maxyear, delta)
    if fitlog:
        result = minimize(objectiveFunctionLogGIE, x0=guess,  args = arguments, method = method)
    else:
        result = minimize(objectiveFunctionGIE, x0=guess,  args = arguments, method = method)
    return result


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

"""
There is probably a better way of doing this for mulitple shocks (in
fact I know there is, and have used it). Unfortunately there is a bit
of art in this.

Future plan is to entropy min the entire data set, and then do what
is essentially a CFAR detector that adds in shocks when they are
detected. This can be used to forecast shocks as well as just fit 
the data available.
"""

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
