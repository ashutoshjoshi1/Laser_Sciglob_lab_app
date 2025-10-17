#Reduced xfus library for spectrometer lib's only.

import time
import sys
import logging
import numpy as np




#create a module logger
logger=logging.getLogger(__name__)

def split_cycles(max_ncy_per_meas, ncy):
    """
    Split the total number of desired cycles (ncy) into a list of calls,
    each with a maximum number of cycles per measurement (max_ncy_per_meas).

    Args:
        max_ncy_per_meas (int): Maximum number of cycles allowed per measurement.
        ncy (int): Total number of desired cycles.

    Returns:
        list: A list of integers representing the number of cycles per call.

    Example:
        max_ncy_per_meas=10
        ncy=31
        return: [10,10,10,1]
    """
    ncalls = []
    while ncy > 0:
        cycles = min(max_ncy_per_meas, ncy)
        ncalls.append(cycles)
        ncy -= cycles

    # Generate the explanation string
    explanation_parts = []
    for value in set(ncalls):
        count = ncalls.count(value)
        explanation_parts.append(str(count)+"x"+str(value)+"cy")
    ncalls_info = "+".join(explanation_parts)

    return ncalls, ncalls_info


def calc_msl(spec_alias,x,sxy,sy,syy,nav=[],syw=[]):
    """
    res,m,s,l=calc_msl(spec_alias,x,sxy,sy,syy,nav=[],syw=[])

    <spec_alias> alias of the spectrometer (will be used for logging purposes only)

    <x> arange with the number of cycles, dimension (nmeas,1). Eg for 10 measurements, x=[0,1,2,...,9]
    <sy> sum of the counts of all cycles, dimension = (npix,1)
    <syy> sum of the counts**2 of all cycles, dimension = (npix,1)
    <sxy> sum of the counts * cycle number [1,2,...,ncy], dimension = (npix,1)

    If <nav> is not empty, it must have dimension (nmeas,1)
    and gives the number of averages that were used for each <x>.
    In that case the formulas used are different and
    <syw> with dimension (npix,1) must be given too.

    This function calculates the mean <m>, standard deviation <s>,
    and the rms to a fitted straight line <l>.

    Note: s=sample standard deviation, not the population standard deviation.

    If <nav> is not empty, the calculated mean <m> is the weighted mean
    using the weights given in <syw>.
    <res> is "OK" if everything was ok. If a negative value was found in l and l**0.5 cannot be calculated,
    this will return "NOK".
    """
    res="OK"
    n=x.shape[0]
    x=np.array(x,dtype=np.float64)
    sxy=np.array(sxy,dtype=np.float64) #must be positive or zero. Not nans.
    sy=np.array(sy,dtype=np.float64)
    syy=np.array(syy,dtype=np.float64)
    #mean
    if n>0:
        m=sy/n
        if nav!=[]:
            nav=np.array(nav,dtype=np.float64)
            syw=np.array(syw,dtype=np.float64)
            mw=syw/nav.sum()
    else:
        m=[]
    #standard deviation
    if n>1:
        s=(syy-n*(m**2))/(n-1)
        if nav!=[]:
            navr=nav-1
            i0=navr==0
            navr[i0]=1
            fact=(navr*nav).sum()/navr.sum()
            s=s*fact
        s=s**(0.5)
    else:
        s=[]
    #rms to fitted straight line
    if n>2:
        sx=x.sum()
        xq=x**2
        sxx=xq.sum()
        delta=n*sxx-sx**2
        d=(sxx*sy-sx*sxy)/delta
        k=(n*sxy-sx*sy)/delta
        l=(syy-2*k*sxy-2*d*sy+(k**2)*sxx+2*k*d*sx+n*(d**2))/(n-2)
        if nav!=[]:
            l=l*fact
        if np.any(l<0):
            logger.warning("Negative value found in l while calculating calc_msl, for spec "+spec_alias+".")
            try:
                logger.warning("l="+str(l))
                nval=np.where(l<0) #array with the indexes of the negative values
                logger.warning("negative values of l found in positions: "+str(nval))
                logger.warning("negative value of l are: "+str(l[nval]))
                logger.warning("x="+str(x)) #Array (nmeas, 1)
                logger.warning("sy="+str(sy))  #Array (npix,1)
                logger.warning("sy[nval]="+str(sy[nval]))
                logger.warning("sxy="+str(sxy)) #Array (npix,1)
                logger.warning("sxy[nval]="+str(sxy[nval]))
                logger.warning("syy="+str(syy))  #Array (npix,1)
                logger.warning("syy[nval]="+str(syy[nval]))
                logger.warning("sx="+str(sx)) #It is a number
                logger.warning("sxx="+str(sxx)) #It is a number
                logger.warning("delta="+str(delta)) #It is a number
                logger.warning("d="+str(d)) #It is an array
                logger.warning("d[nval]="+str(d[nval]))
                logger.warning("k="+str(k)) #It is an array
                logger.warning("k[nval]="+str(k[nval]))
                logger.warning("nav="+str(nav)) #it is usually an empty list []
            except:
                pass
            finally:
                l=[]
                res="NOK"
        else:
            l=l**(0.5)
    else:
        l=[]
    if (n>1)and(nav!=[]):
        m=mw #this is the best estimate for the mean
    return res,m,s,l


class SpecClock:
    """
    Internal monotonically increasing clock for spectrometers.
    That does not depend on the system clock.
    Useful to avoid time jumps when the system clock is changed.

    Usage:
    my_clock=Spec_Clock()
    print(my_clock.now)
    """
    def __init__(self):
        if sys.version_info[0] < 3: #python 2.x
            self._base_time, self._base_clock = time.time(), time.clock()
        else: #python 3.x
            self._base_time, self._base_clock = time.time(), time.perf_counter()

    def now(self):
        """
        Returns the current time as seconds since 1.1.1970 00:00 (Unix epoch)
        (monotonically increasing clock, independent of the system clock)
        """
        if sys.version_info[0] < 3: #python 2.x
            return self._base_time + time.clock() - self._base_clock
        else: #python 3.x
            return self._base_time + time.perf_counter() - self._base_clock


spec_clock=SpecClock()


