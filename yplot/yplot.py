#!/usr/bin/env python3

import tensorflow as tf
from io import BytesIO
from PIL import Image
import numpy as np
from math import atan2,degrees
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Calibri') 
matplotlib.rc('text', usetex='false') 
matplotlib.rc('image', cmap='gist_earth')

def read_tfevents(fn):
    s = list(tf.train.summary_iterator(fn))[1:]
    return s


def extract_values(summaries, tag_or_tags, get_val_fn):
    if isinstance(tag_or_tags, list):
        return extract_multiple_values(summaries, tag_or_tags, get_val_fn)
    return extract_multiple_values(summaries, [tag_or_tags], get_val_fn)[0]

def extract_multiple_values(summaries, tags, get_val_fn):
    tagset = set(tags)
    steps_values = dict((tag, ([], [])) for tag in tagset)
    for s in summaries:
        for v in s.summary.value:
            if v.tag in tagset:
                steps, values = steps_values[v.tag]
                steps.append(s.step)
                values.append(get_val_fn(v))
    return [steps_values[tag] for tag in tags]


def extract_scalar(summaries, tag):
    return extract_values(summaries, tag, lambda v: v.simple_value)


def extract_image(summaries, tag):
    return extract_values(summaries, tag, lambda v: v.image.encoded_image_string)


def string_to_image(encoded_string):
    img = Image.open(BytesIO(encoded_string))
    return img


def strings_to_images(encoded_strings):
    return [string_to_image(s) for s in encoded_strings]


def first_ge(steps, limit):
    steps = np.asarray(steps)
    if limit < 0 or np.all(limit > steps):
        return np.argmax(steps)
    return np.argmax(steps >= limit)


def firsts_ge(steps, limits):
    return [first_ge(steps, l) for l in limits]


def extract_image_at_steps(summaries, tag, steps):
    s, v = extract_image(summaries, tag)
    l = firsts_ge(s, steps)
    return [v[i] for i in l]



#http://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.get_axes()
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if 'fontsize' not in kwargs:
        kwargs['fontsize'] = 14

    if 'weight' not in kwargs:
        kwargs['weight'] = 'bold'

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_axis_bgcolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].get_axes()
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)


def apply_mp_style(ax):
    ax.set_axis_bgcolor('#dddddd')
    ax.grid(color='white', axis='y', which='major', linewidth=1)
    ax.grid(color='white', axis='y', which='minor', linewidth=0.3)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', which='both', left='off', right='off')
    for line in ax.get_lines():
        line.set_linewidth(2.25)
    # ax.yaxis.get_label().set_rotation(0)
    # ax.yaxis.set_label_coords(0, 1)
    # ax.yaxis.get_label().set_horizontalalignment('left')
    ax.title.set_position((0, 1.04))
    ax.title.set_horizontalalignment('left')
    ax.title.set_size(15.)
    ax.title.set_weight(550)


#http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    y = np.asarray(y)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
