#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated GILDAS-CLASS Pipeline
-------------------------------
Reduction mode
Version 1.5

Copyright (C) 2025 - Andrés Megías Toledano

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

# Libraries and functions.
import os
import sys
import copy
import time
import glob
import platform
import argparse
import subprocess
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import median_abs_deviation
from scipy.interpolate import UnivariateSpline
from matplotlib.backend_bases import MouseEvent, KeyEvent
# Matplotlib backend.
matplotlib_backend = 'qtagg'
plt.matplotlib.use(matplotlib_backend)
if matplotlib_backend == 'qtagg':
    from PyQt5.QtWidgets import QInputDialog

# Custom functions.

def rolling_function(func, y, size, **kwargs):
    """
    Apply a function in a rolling way, in windows of the specified size.

    Parameters
    ----------
    y : array
        Input data.
    func : function
        Function to be applied.
    size : int
        Size of the windows to group the data. It must be odd.
    **kwargs : (various)
        Keyword arguments of the function to be applied.

    Returns
    -------
    yr : array
        Resultant array.
    """
    yp = pd.Series(y)
    ypr = pd.Series(yp).rolling(size, min_periods=1, center=True)
    yr = ypr.apply(lambda x: func(x, **kwargs)).values
    return yr

def get_mask_from_windows(x, windows):
    """
    Select the regions of the input array specified by the given windows.

    Parameters
    ----------
    x : array
        Input data.
    windows : array
        Windows that specify the regions of the data.

    Returns
    -------
    mask : array (bool)
        Resultant mask array.
    """
    mask = np.ones(len(x), dtype=bool)
    for (x1, x2) in windows:
        mask *= (x < x1) + (x > x2)
    return mask

def get_windows_from_mask(x, mask, margin=1., ref_width=8.):
    """
    Return the windows of the masked regions of the input array.

    Parameters
    ----------
    x : array
        Input data.
    mask : array (bool)
        Indices of the empty regions of data.
    margin : float, optional
        Relative margin added to the windows found initially.
        The default is 0.0.
    ref_width: float, optional
        Reference width of the windows.

    Returns
    -------
    windows : array (float)
        List of the inferior and superior limits of each window.
    inds : array (int)
        List of indices that define the filled regions if the data.
    """
    resolution = np.median(abs(np.diff(x)))
    all_inds = np.arange(len(x))
    diff_inds = np.diff(np.concatenate(([0], np.array(mask, dtype=int), [0])))
    cond1 = (diff_inds == 1)[:-1]  # from 0 to 1 
    cond2 = (diff_inds == -1)[1:]  # from 1 to 0 
    inds = np.append(all_inds[cond1], all_inds[cond2])
    inds = np.sort(inds).reshape(-1,2)
    windows = x[inds]
    i = 0
    while i+1 < len(windows):
        diff = windows[i+1,0] - windows[i,1]
        if diff <= ref_width / 6.:
            windows[i,1] = windows[i+1,1]
            windows = np.delete(windows, i+1, axis=0)
        else:
            i += 1
    i = 0
    while i < len(windows):
        x1, x2 = windows[i]
        if (x2 - x1) > 8.*ref_width*resolution:
            windows = np.delete(windows, i, axis=0)
        else:
            i += 1
    for (i, window) in enumerate(windows):
        center = (window[0] + window[1]) / 2
        semiwidth = (window[1] - window[0]) / 2
        semiwidth += max(margin, ref_width/6.) * resolution
        windows[i,:] = [center - semiwidth, center + semiwidth]
    if len(windows) > 0 and windows[0,0] <= x.min():
        windows = np.delete(windows, 0, axis=0)
    if len(windows) > 0 and windows[-1,1] >= x.max():
        windows = np.delete(windows, -1, axis=0)
    return windows

def get_windows_from_points(selected_points):
    """Format the selected points into windows."""
    are_points_even = len(selected_points) % 2 == 0
    windows = selected_points[:] if are_points_even else selected_points[:-1]
    windows = np.array(windows).reshape(-1,2)
    for (i, x1x2) in enumerate(windows):
        x1, x2 = min(x1x2), max(x1x2)
        windows[i,:] = [x1, x2]
    return windows

def invert_windows(x, windows):
    """Obtain the complementary of the input windows for the array x."""
    mask = get_mask_from_windows(x, windows)
    windows = get_windows_from_mask(x, ~mask)
    return windows

def sigma_clip_mask(y, sigmas=6.0, iters=2):
    """
    Apply a sigma clip and return a mask of the remaining data.

    Parameters
    ----------
    y : array
        Input data.
    sigmas : float, optional
        Number of standard deviations used as threshold. The default is 4.0.
    iters : int, optional
        Number of iterations performed. The default is 3.

    Returns
    -------
    mask : array (bool)
        Mask of the remaining data after applying the sigma clip.
    """
    mask = np.ones(len(y), dtype=bool)
    abs_y = abs(y)
    for i in range(iters):
        mad = median_abs_deviation(abs_y[mask], scale='normal')
        mask *= abs_y < sigmas*mad
    return mask

def fit_baseline(x, y, windows, smooth_factor):
    """
    Fit the baseline of the curve ignoring the specified windows.

    Parameters
    ----------
    x : array
        Independent variable.
    y : array
        Dependent variable.
    windows : array
        Windows that specify the regions of the data.
    smooth_factor : int
        Size of the filter applied for the fitting of the baseline.

    Returns
    -------
    yf : array
        Baseline of the curve.
    """
    mask = get_mask_from_windows(x, windows)
    x_ = x[mask]
    y_ = y[mask]
    y_s = rolling_function(np.median, y_, smooth_factor)
    s = sum((y_s - y_)**2)
    spl = UnivariateSpline(x_, y_, s=s)
    yf = spl(x)
    return yf

def identify_lines(x, y, smooth_factor, ref_width, sigmas, iters=2):
    """
    Identify the lines of the spectrum and fit the baseline.

    Parameters
    ----------
    x : array
        Frequency.
    y : array
        Intensity.
    smooth_factor : int
        Size of the filter applied for the fitting of the baseline.
    line_width : float
        Reference line width for merging close windows.
    sigmas : float
        Threshold for identifying the outliers.
    iters : int, optional
        Number of iterations of the process. The default is 2.

    Returns
    -------
    windows: array
        Values of the windows of the identified lines.
    """
    y_ = rolling_function(np.median, y, smooth_factor)  
    for i in range(iters):
        mask = sigma_clip_mask(y-y_, sigmas=sigmas, iters=2)
        windows = get_windows_from_mask(x, ~mask, margin=1.5, ref_width=ref_width)
        if i+1 < iters:
            y_ = fit_baseline(x, y, windows, smooth_factor)  
    return windows

def load_spectrum(filename, load_fits=False):
    """
    Load the spectrum from the given input file.

    Parameters
    ----------
    filename : str
        Path of the plain text file (.dat) to load, without the extenxion.
    load_fits : bool
        If True, load also a .fits file and return the HDU list. 

    Returns
    -------
    x : array
        Frequency.
    y : array
        Intensity.
    hdul : HDU list (astropy)
        List of the HDUs (Header Data Unit).
    """
    data = np.loadtxt(f'{filename}.dat')
    x = data[:,0]
    y = data[:,1]
    if np.sum(np.isnan(data)) != 0:
        raise Exception(f'Data of file {file} is corrupted.')
    if load_fits:
        hdul = fits.open(f'{filename}.fits')
        if 'BLANK' in hdul[0].header:
            del hdul[0].header['BLANK']
        return x, y, hdul
    else:
        return x, y

def save_yaml_dict(dictionary, file_path, default_flow_style=False, replace=False):
    """
    Save the input YAML dictionary into a file.

    Parameters
    ----------
    dictionary : dict
        Dictionary that wants to be saved.
    file_path : str
        Path of the output file.
    default_flow_style : bool, optional
        The flow style of the output YAML file. The default is False.
    replace : bool, optional
        If True, replace the output file in case it existed. If False, load the
        existing output file and merge it with the input dictionary.
        The default is False.

    Returns
    -------
    None.
    """
    file_path = os.path.realpath(file_path)
    if not replace and os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            old_dict = yaml.safe_load(file)
        new_dict = {**old_dict, **dictionary}
    else:
        new_dict = dictionary
    with open(file_path, 'w') as file:
        yaml.dump(new_dict, file, default_flow_style=default_flow_style)

def get_rms_noise(x, y, windows=[], sigmas=6., freq_margin=0., iters=3):
    """
    Obtain the RMS noise of the input data, ignoring the given windows.

    Parameters
    ----------
    x : array
        Independent variable.
    y : array
        Dependent variable.
    windows : array, optional
        Windows of the independent variable that will be avoided in the
        calculation of the RMS noise. The default is [].
    sigmas : float, optional
        Number of deviations used as threshold for the sigma clip applied to
        the data before the calculation of the RMS noise. The default is 6.0.
    freq_margin : float, optional
        Relative frequency margin that will be ignored for calculating the RMS
        noise. The default is 0.
    iters : int, optional
        Number of iterations performed for the sigma clip applied to the data
        before the calculation of the RMS noise. The default is 3.

    Returns
    -------
    rms_noise : float
        Value of the RMS noise of the data.
    """
    N = len(x)
    i1, i2 = int(freq_margin*N), int((1-freq_margin)*N)
    x = x[i1:i2]
    y = y[i1:i2]
    mask = get_mask_from_windows(x, windows)
    y = y[mask]
    mask = sigma_clip_mask(y, sigmas=sigmas, iters=iters)
    y = y[mask]
    rms_noise = np.sqrt(np.mean(y**2)) 
    return rms_noise

def find_rms_region(x, y, rms_noise, windows=[], rms_threshold=0.1,
                    offset_threshold=0.05, reference_width=200, min_width=120,
                    max_iters=800):
    """
    Find a region of the input data that has a similar noise than the one given.

    Parameters
    ----------
    x : array
        Independent variable.
    y : array
        Dependent variable.
    rms_noise : float
        The value of the RMS used as a reference.
    windows : array, optional
        The regions of the independent variable that should be ignored.
        The default is [].
    rms_threshold : float, optional
        Maximum relative difference that can exist between the RMS noise of the
        searched region and the reference RMS noise. The default is 0.05.
    offset_threshold : float, optional
        Maximum value, in units of the reference RMS noise, that the mean value
        of the dependent variable can have in the searched region.
        The default is 0.05.
    reference_width : int, optional
        Size of the desired region, in number of channels. The default is 200.
    min_width : int, optional
        Minimum size of the desired region, in number of channels.
        The default is 120.
    max_iters : int, optional
        Maximum number of iterations that will be done to find the desired
        region. The default is 800

    Returns
    -------
    rms_region : list
        Frequency regions of the desired region.
    """
    i = 0
    local_rms = 0
    offset = 1*rms_noise
    while not (abs(local_rms - rms_noise) / rms_noise < rms_threshold
               and abs(offset) / rms_noise < offset_threshold):
        width = max(min_width, reference_width)
        resolution = np.median(np.diff(x))
        central_freq = np.random.uniform(x[0] + width/2*resolution,
                                         x[-1] - width/2*resolution)
        region_inf = central_freq - width/2*resolution
        region_sup = central_freq + width/2*resolution
        cond = (x > region_inf) & (x < region_sup)
        y_ = y[cond]
        valid_range = True
        for x1, x2 in windows:
            if (region_inf < x1 < region_sup) or (region_inf < x2 < region_sup):
                valid_range = False
        if valid_range:
            local_rms = float(np.sqrt(np.mean(y_**2)))
            offset = np.mean(y_)
        i += 1
        if i > max_iters:
            return []
    rms_region = [float(central_freq - width/2*resolution),
                  float(central_freq + width/2*resolution)]
    return rms_region

def remove_extra_spaces(input_text):
    """
    Remove extra spaces from a text string.

    Parameters
    ----------
    input_text : str
        Input text string.

    Returns
    -------
    text : str
        Resulting text.
    """
    text = input_text
    for i in range(12):
        if '  ' in text:
            text = text.replace('  ', ' ')
    if text.startswith(' '):
        text = text[1:]
    return text

def format_windows(selected_points):
    """Format the selected points into windows."""
    are_points_even = len(selected_points) % 2 == 0
    windows = selected_points[:] if are_points_even else selected_points[:-1]
    windows = np.array(windows).reshape(-1,2)
    for (i, x1x2) in enumerate(windows):
        x1, x2 = min(x1x2), max(x1x2)
        windows[i,:] = [x1, x2]
    return windows

def plot_windows(selected_points):
    """Plot the current selected windows."""
    for x in selected_points:
        plt.axvline(x, color='darkgray', alpha=1., zorder=1.5)
    windows = format_windows(selected_points)
    for (x1,x2) in windows:
        plt.axvspan(x1, x2, transform=plt.gca().transAxes,
                    color='lightgray', alpha=1., zorder=1.5)

def reduce_spectrum(spectrum, selected_points, smooth_factor):
    """Reduce the data."""
    windows = format_windows(selected_points)
    frequency = spectrum['frequency']
    intensity = spectrum['intensity']
    baseline = fit_baseline(frequency, intensity, windows, smooth_factor)
    spectrum['baseline'] = baseline
    spectrum['reduced intensity'] = intensity - baseline
    
def calculate_ylims(x, y, x_lims, perc1=0., perc2=100., rel_margin=0.):
    """Calculate vertical limits for the given spectra."""
    x1, x2 = x_lims
    mask = (x >= x1) & (x <= x2) & np.isfinite(y)
    y_ = y[mask]
    y1 = np.percentile(y_, perc1)
    y2 = np.percentile(y_, perc2)
    margin = rel_margin * (y2 - y1)
    y_lims = [y1 - margin, y2 + margin]
    return y_lims 

def custom_input(prompt, window_title=''):
    """Custom input call that uses Qt if using qtagg backend."""
    if matplotlib_backend == 'qtagg':
        prompt = prompt.replace('- ', '')
        text, _ = QInputDialog.getText(None, window_title, prompt)
    else:
        text = input(prompt)
    return text
 
def plot_data(spectrum):
    """
    Plot spectrum contained in the input dictionary.
    
    Parameters
    ----------
    spectrum : dict
        Dictionary containing the following elements of the spectrum:
            frequency : array (float)
            intensity : array (float)
            baseline : array (float)
    """

    frequency = spectrum['frequency']
    intensity = spectrum['intensity']
    intensity_cont = spectrum['baseline']
    intensity_red = intensity - intensity_cont

    fig = plt.figure('Automated GILDAS-CLASS Pipeline')
    plt.clf()
    
    sp1 = plt.subplot(2,1,1)
    plt.step(frequency, intensity, where='mid', color='black', ms=6)
    plt.plot(frequency, intensity_cont, 'tab:green', label='fitted baseline')
    if not args.rms_check:
        plot_windows(selected_points)
    plt.axvspan(0., 0., 0., 0., facecolor='lightgray', edgecolor='darkgray',
                label='masked windows')
    plt.locator_params(axis='both', nbins=10)
    plt.ticklabel_format(style='sci', useOffset=False)
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.xlabel('frequency (MHz)')
    plt.ylabel('original intensity (K)')
    plt.legend(loc='upper right')

    plt.subplot(2,1,2, sharex=sp1)
    plt.step(frequency, intensity_red, where='mid', color='black')
    if not args.rms_check:
        plot_windows(selected_points)
    plt.locator_params(axis='both', nbins=10)
    plt.ticklabel_format(style='sci', useOffset=False)
    plt.xlim(x_lims)
    plt.ylim(yr_lims)
    plt.xlabel('frequency (MHz)')
    plt.ylabel('reduced intensity (K)')

    title = f'Full spectrum - {filename}\n'
    fontsize = max(7., 12. - 0.1*max(0, len(title) - 85.))
    plt.suptitle(title, fontsize=fontsize, fontweight='semibold', y=0.96)
    plt.tight_layout(h_pad=1.5, rect=(0.01, 0.01, 0.99, 1.))
    plt.text(0.98, 0.96, 'check terminal\nfor instructions',
             ha='right', va='top', transform=plt.gca().transAxes,
             bbox=dict(edgecolor=[0.8]*3, facecolor='white'))
    
    return fig
 
#%% Functions to create interactive mode.

def click1(event):
    """Interact with the plot by clicking on it."""
    if type(event) is not MouseEvent:
        pass
    global click_time
    click_time = time.time()

def click2(event):
    """Interact with the plot by clicking on it."""
    if type(event) is not MouseEvent:
        pass
    button = str(event.button).lower().split('.')[-1]
    if button in ('left', 'right', '1', '3'):
        global click_time
        elapsed_click_time = time.time() - click_time
        x = event.xdata
        if (elapsed_click_time > 0.5  # s
                or x is None or x is not None and not np.isfinite(x)):
            return 
        global spectrum, selected_points, data_log, ilog
        if button in ('left', '1'):
            selected_points += [x]
            for i in (1, 2):
                plt.subplot(2,1,i)
                plt.axvline(x, color='darkgray', alpha=1.)
            are_points_even = len(selected_points) % 2 == 0
            if are_points_even:
                x1, x2 = selected_points[-2:]
                for i in (1, 2):
                    plt.subplot(2,1,i)
                    plt.axvspan(x1, x2, transform=plt.gca().transAxes,
                                color='lightgray', alpha=1.)
        else:
            if len(selected_points) == 0:
                return
            are_points_even = len(selected_points) % 2 == 0
            was_removed = False
            if are_points_even:
                windows = np.array(selected_points).reshape(-1,2)
                for x1x2 in windows:
                    x1, x2 = min(x1x2), max(x1x2)
                    if x1 < x < x2:
                        selected_points.remove(x1)
                        selected_points.remove(x2)
                        was_removed = True
                        break
            if not was_removed:
                del selected_points[-1]
            plot_data(spectrum)
            plot_windows(selected_points)
        data = {'spectrum': spectrum, 'selected_points': selected_points}
        data_log = data_log[:ilog+1] + [copy.deepcopy(data)]
        ilog += 1
        plt.draw()    

def press_key(event):
    """Interact with the plot when pressing a key."""
    if type(event) is not KeyEvent:
        pass
    global spectrum, selected_points, data, data_log, ilog, reduction_params
    global x_lims, y_lims, yr_lims, x_lims_orig, set_auto_ylims
    if event.key == 'ctrl+enter':
        if selected_points != reduction_params['selected_points']:
            smooth_factor = reduction_params['smooth_factor']
            reduce_spectrum(spectrum, selected_points, smooth_factor)
            print('Reduced spectrum by subtracting baseline with smoothing'
                  ' factor'
                  f' {smooth_factor}.')
        plt.close('all')
    elif event.key == 'escape':
        sys.exit(1)
    plt.subplot(2,1,1)
    x_lims = list(plt.xlim())
    y_lims = list(plt.ylim())
    plt.subplot(2,1,2)
    yr_lims = plt.ylim()
    if event.key in ('z', 'Z', '<', 'left', 'right'):
        x_range = x_lims[1] - x_lims[0]
        if event.key in ('z', 'Z', '<'):
            if event.key == 'z':
                x_lims = [x_lims[0] + x_range/6, x_lims[1] - x_range/6]
            else:
                x_lims = [x_lims[0] - x_range/4, x_lims[1] + x_range/4]
                x_lims = [max(x_lims[0], x_lims_orig[0]),
                          min(x_lims[1], x_lims_orig[1])]
        else:
            if event.key == 'left':
                x_lims = [x_lims[0] - x_range/4, x_lims[1] - x_range/4]
            else:
                x_lims = [x_lims[0] + x_range/2, x_lims[1] + x_range/2]
        if set_auto_ylims:
            y_lims = calculate_ylims(spectrum['frequency'], spectrum['intensity'],
                                     x_lims, perc1=1., perc2=99., rel_margin=0.30)
    elif event.key == 'y':
        x = spectrum['frequency']
        y = spectrum['intensity']
        y_red = spectrum['reduced intensity']
        perc1, perc2 = 0.1, 99.9
        prev_y_lims = copy.copy(y_lims)
        y_lims = calculate_ylims(x, y, x_lims, perc1, perc2, rel_margin=0.10)
        if y_lims == prev_y_lims:
            perc1, perc2 = 0., 100.
            y_lims = calculate_ylims(x, y, x_lims, perc1, perc2, rel_margin=0.10)
        yr_lims = calculate_ylims(x, y_red, x_lims, perc1, perc2, rel_margin=0.10)
    elif event.key == 'Y':
        set_auto_ylims = not set_auto_ylims
        if set_auto_ylims:
            print('Automatic vertical limits is now on.')
        else:
            print('Automatic vertical limits is now off. ')
        y_lims = calculate_ylims(spectrum['frequency'], spectrum['intensity'],
                                 x_lims, perc1=1., perc2=99., rel_margin=0.30)
    elif event.key in ('i', 'I'):
        if event.key == 'i':
            smooth_factor = copy.copy(args.smooth_factor)
            ref_width = copy.copy(args.ref_width)
            line_sigmas = copy.copy(args.line_sigmas)
        else:
            text = custom_input(' - Enter smoothing factor, reference width'
                                ' and sigmas threshold: ', 'Identify lines')
            text = text.replace(' ', '')
            smooth_factor, ref_width, line_sigmas = np.array(text.split(','),
                                                             float).tolist()
            smooth_factor = round(smooth_factor)
            ref_width = round(smooth_factor)
        windows = identify_lines(spectrum['frequency'], spectrum['intensity'],
                                 smooth_factor, ref_width, line_sigmas, iters=2)
        selected_points = list(np.array(windows).flatten())
        data['selected_points'] = selected_points
        num_windows = len(windows)
        print(f'{num_windows} lines identified with smoothing factor {smooth_factor},'
              f' reference width {ref_width} and sigmas threshold {line_sigmas}.')
    elif event.key in ('r', 'R'):
        if event.key == 'r':
            smooth_factor = copy.copy(args.smooth_factor)
        else:
            text = custom_input('- Enter smoothing factor: ',
                                'Baseline & Reduction')
            smooth_factor = float(''.join([char for char in text if char.isdigit()]))
            smooth_factor = round(smooth_factor)
        new_reduction_params = {'selected_points': selected_points,
                                'smooth_factor': smooth_factor}
        reduce_spectrum(spectrum, selected_points, smooth_factor)
        reduction_params = new_reduction_params
        print('Reduced spectrum by subtracting baseline with smoothing factor'
              f' {smooth_factor}.')
    elif event.key in ('n', 'N'):
        if event.key == 'n':
            rms_sigmas = copy.copy(args.rms_sigmas)
            rms_freq_margin = copy.copy(args.rms_freq_margin)
        else:
            text = custom_input('- Enter sigmas threshold and frequency margin: ',
                                'RMS noise')
            text = text.replace(' ', '')
            rms_sigmas, rms_freq_margin = np.array(text.split(','), float).tolist()
        windows = get_windows_from_points(selected_points)
        windows_inv = invert_windows(spectrum['frequency'], windows)
        rms = get_rms_noise(spectrum['frequency'], spectrum['reduced intensity'],
                            windows_inv, rms_sigmas, rms_freq_margin, iters=3)
        print(f'RMS noise: {1e3*rms:.3g} mK.')
    elif event.key in ('tab', '\t'):
        selected_points = []
    elif event.key in ('ctrl+z', 'cmd+z', 'ctrl+Z', 'cmd+Z'):
        if 'z' in event.key and ilog == 0:
            print('Error: Cannot undo.')
        elif 'z' not in event.key and ilog == len(data_log)-1:
            print('Error: Cannot redo.')
        else:
            ilog = (max(0, ilog-1) if 'z' in event.key
                    else min(len(data_log)-1, ilog+1))
            data = copy.deepcopy(data_log[ilog])
            spectrum = data['spectrum']
            selected_points = data['selected_points']
            if 'z' in event.key:
                print('Action undone.')
            else:
                print('Action redone.')
    if event.key in ('i', 'I', 'r', 'R', 'tab', '\t'):
            data = {'spectrum': spectrum, 'selected_points': selected_points}
            data_log = data_log[:ilog+1] + [copy.deepcopy(data)]
            ilog += 1
    plot_data(spectrum)
    plot_windows(selected_points)
    plt.draw()

# Arguments.
parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-folder', default='.', type=str)
parser.add_argument('-smooth_factor', default=20, type=int)
parser.add_argument('-ref_width', default=6, type=int)
parser.add_argument('-line_sigmas', default=8., type=float)
parser.add_argument('-rms_sigmas', default=6., type=float)
parser.add_argument('-rms_freq_margin', default=0.1, type=float)
parser.add_argument('-plots_folder', default='plots')
parser.add_argument('--not_interactive', action='store_true')
parser.add_argument('--save_plots', action='store_true')
parser.add_argument('--rms_check', action='store_true')
args = parser.parse_args()
interactive_check = not args.not_interactive
original_folder = os.path.realpath(os.getcwd())
os.chdir(args.folder)
sep = '\\' if platform.system() == 'Windows' else '/'
if not args.folder.endswith(sep):
    args.folder += sep

# Instructions for interactive mode.
instructions = \
"""Interactive check
-----------------
- Use Z/<, Left/Right or the plot buttons to explore the spectrum region.
- Press Y to adjust the vertical axis limits, or Shift+Y to activate/deactivate
  automatic vertical limits adjustment.
- Left/Right click to add/remove a window edge.
- Press Tab to remove all the windows.
- Press I / Shift+I to automatically identify line windows.
- Press R / Shift+R to reduce the spectrum using the current windows.
- Press Ctrl+Z / Ctrl+Shift+Z to undo/redo.
- Press N / Shift+N to compute the RMS noise.
- Press Ctrl+Enter or close the window to accept the reduction and continue,
  or press Escape to cancel and exit.
"""

# Change backend and remove keymaps for interactive mode.
if interactive_check:
    keymaps = ('back', 'copy', 'forward', 'fullscreen', 'grid', 'grid_minor',
               'help', 'home', 'pan', 'quit', 'quit_all', 'save', 'xscale',
               'yscale', 'zoom')
    for keymap in keymaps:            
        plt.rcParams.update({f'keymap.{keymap}': []})

#%%

if not args.rms_check:
    if os.path.isfile('frequency_windows.yaml'):
        with open('frequency_windows.yaml') as file:
            all_windows = yaml.safe_load(file)
    else:
        print('Warning: The file frequency_windows.yaml is missing.'
              ' They will be automatically computed by identifying'
              ' the most prominent lines.')
        print()
        all_windows = {}
were_windows_updated = False
        
rms_noises = {}
frequency_ranges = {}
reference_frequencies = {}
rms_regions = {}
resolutions = {}
filenames = args.filename.split(',')
num_files = len(filenames)

# Processing of each spectrum.
for filename in filenames:
    
    # Remove extension.
    ext = filename.split('.')[-1] if '.' in filename else ''
    if ext != '':
        filename = filename[:-len(ext)-1]
    # If .dat and .fits files do not exist, export them from given CLASS file.
    if ext not in ('', 'dat', 'fits'):
        file_dat_exists = os.path.exists(f'{filename}.dat')
        file_fits_exists = os.path.exists(f'{filename}.fits')
        script = [f'file in {filename}.{ext}', 'find /all', 'list', 'get first',
                  'modify doppler', 'modify doppler *']
        if not file_dat_exists:
            script += [f'greg {filename}.dat /formatted']
        if not file_fits_exists:
            script += [f'fits write {filename}.fits /mode spectrum']
        script = [line + '\n' for line in script] + ['exit']
        with open('reduction-input.class', 'w') as file:
            file.writelines(script)
        p = subprocess.run(['class', '@reduction-input.class'],
                            capture_output=True, text=True)
        if p.returncode == 1:
            sys.exit()
        doppler_corr, prev_line = '', ''
        for line in p.stdout.split('\n'):
            print(line)
            if doppler_corr == '':
                if ('Doppler factor' in line and '***' not in line
                        and (not 'Doppler factor' in prev_line
                             or 'Doppler factor' in prev_line and '***' in prev_line)): 
                    doppler_corr = remove_extra_spaces(line).split(' ')[-1]
                if 'I-OBSERVATORY' not in line:
                    prev_line = line
        print()
    # Loading of the data file.
    frequency, intensity, hdul = load_spectrum(filename, load_fits=True)
    fits_data = hdul[0]
    frequency_ranges[filename] = [float(frequency[0]), float(frequency[-1])]
    resolutions[filename] = hdul[0].header['cdelt1'] / 1e6
    reference_frequencies[filename] = hdul[0].header['restfreq'] / 1e6
    # Reduction.
    if args.rms_check or filename not in all_windows:
        original_windows = []
        windows = identify_lines(frequency, intensity, args.smooth_factor,
                                 args.ref_width, args.line_sigmas, iters=2)
    else:
        windows = original_windows = all_windows[filename]
    intensity_cont = fit_baseline(frequency, intensity, windows, args.smooth_factor)
    intensity_red = intensity - intensity_cont
    spectrum = {'frequency': frequency, 'intensity': intensity,
                'baseline': intensity_cont, 'reduced intensity': intensity_red}
    reduction_params = {'selected_points': np.array(windows).flatten().tolist(),
                        'smooth_factor': args.smooth_factor}
    
    # Interactive check of windows.
    selected_points = list(np.array(windows).flatten())
    x1, x2 = frequency.min(), frequency.max()
    margin = (x2 - x1) * 0.02
    x_lims = [x1 - margin, x2 + margin]
    y_lims = calculate_ylims(frequency, intensity, x_lims,
                             perc1=0.1, perc2=99.9, rel_margin=0.15)
    yr_lims = calculate_ylims(frequency, intensity_red, x_lims,
                              perc1=0.1, perc2=99.9, rel_margin=0.40)
    if interactive_check:
        plt.close('all')
        plt.figure('Automated GILDAS-CLASS Pipeline', figsize=(9.,7.))
        print(instructions)
        print('Reduced spectrum by subtracting baseline with smoothing factor'
              f' {args.smooth_factor}.')
        ilog = 0
        data = {'spectrum': spectrum, 'selected_points': selected_points}
        data_log = [copy.deepcopy(data)]
        x_lims_orig = copy.copy(x_lims)
        set_auto_ylims = False
        fig = plot_data(spectrum)
        fig.canvas.mpl_connect('button_press_event', click1)
        fig.canvas.mpl_connect('button_release_event', click2)
        fig.canvas.mpl_connect('key_press_event', press_key)
        plt.show()  # interactive mode
        intensity = spectrum['intensity']
        windows = format_windows(selected_points)
        if not np.array_equal(original_windows, windows):
            windows = [[round(float(x1), 6), round(float(x2), 6)]
                       for (x1,x2) in windows]
            all_windows[filename] = windows
            were_windows_updated = True
            save_yaml_dict(all_windows, 'frequency_windows.yaml',
                           default_flow_style=None)
            print(f'Updated windows for file {filename}.')
    
    # Noise.
    rms_noise = get_rms_noise(frequency, intensity_red, windows,
                              args.rms_sigmas, args.rms_freq_margin, iters=3,)

    rms_noises[filename] = float(1e3*rms_noise)
    # Noise regions.
    if not args.rms_check:
        rms_region = find_rms_region(frequency, intensity_red, rms_noise,
                        windows=windows, rms_threshold=0.1, offset_threshold=0.05,
                        reference_width=2*args.smooth_factor)
        if len(rms_region) == 0:
            print(f'Warning: No RMS region was found for spectrum {filename}.')
            rms_region = [float(frequency[0]), float(frequency[-1])]
        rms_regions[filename] = rms_region
    
    # Output.
    output_filename = filename + '-r'
    fits_data = np.float32(np.zeros((1,1,1,len(intensity))))
    fits_data[0,0,0,:] = np.float32(intensity_red)
    hdul[0].data = fits_data
    hdul[0].scale('float32')
    hdul.writeto(f'{output_filename}.fits', overwrite=True)
    hdul.close()
    output_data = np.array([frequency, intensity_red]).transpose()
    np.savetxt(f'{output_filename}.dat', output_data, fmt='%.4f %.4e')
    if ext in ('', '.dat', '.fits'):
        print(f'Saved reduced spectrum in {args.folder}{output_filename}.fits.')
        print(f'Saved reduced spectrum in {args.folder}{output_filename}.dat.')
    
    if not interactive_check and args.save_plots:
        plt.close('all')
        plt.figure(1, figsize=(9.,7.))
        plot_data(spectrum)
        plt.suptitle(f'RMS region - {filename}', fontweight='bold')
        os.chdir(original_folder)
        os.chdir(os.path.realpath(args.plots_folder))
        dpi = 100 if args.rms_check else 200
        imagename = f'spectrum-{filename}.png'
        if args.rms_check:
            imagename = imagename.replace('spectrum-rms', 'rms-spectrum')
        plt.savefig(imagename, dpi=dpi)    
        print(f"    Saved plot in {args.plots_folder}{imagename}.")
        if not args.rms_check:
            plt.xlim(*rms_region)
            plt.subplot(2,1,2)
            plt.ylim(-5*rms_noise, 5*rms_noise)
            filename = 'rms-' + filename
        plt.savefig(imagename, dpi=dpi)
        print(f"    Saved plot in {args.plots_folder}{imagename}.")
        os.chdir(original_folder)
        os.chdir(os.path.realpath(args.folder))   
            
    print()
        
# Export of the rms noise of each spectrum.
save_yaml_dict(rms_noises, 'rms_noises.yaml', default_flow_style=False)
print(f'Saved RMS noises in {args.folder}rms_noises.yaml.')        

# Export of the frequency ranges of each spectrum.
save_yaml_dict(frequency_ranges, 'frequency_ranges.yaml',
               default_flow_style=None)
print(f'Saved frequency ranges in {args.folder}frequency_ranges.yaml.')

# Export of the reference frequencies ranges of each spectrum.
save_yaml_dict(reference_frequencies, 'reference_frequencies.yaml',
               default_flow_style=None)
print(f'Saved frequency ranges in {args.folder}reference_frequencies.yaml.')

# Export of the RMS regions of each spectrum.
save_yaml_dict(rms_regions, 'rms_regions.yaml',
               default_flow_style=None)
print(f'Saved RMS regions in {args.folder}rms_regions.yaml.')

# Export of the frequency resolution of each spectrum.
save_yaml_dict(resolutions, 'frequency_resolutions.yaml',
               default_flow_style=False)
print(f'Saved frequency resolutions in {args.folder}frequency_resolutions.yaml.')

# Export of the frequency windows of each spectrum.
if interactive_check and were_windows_updated:
    save_yaml_dict(all_windows, 'frequency_windows.yaml', default_flow_style=None)
    print(f'Saved windows in {args.folder}frequency_windows.yaml.')
    
# Export final CLASS files.
for filename in filenames:
    ext = filename.split('.')[-1] if '.' in filename else ''
    if ext != '':
        filename = filename[:-len(ext)-1]
    if ext not in ('', 'dat', 'file'):
        print()
        output_filename = f'{filename}-r.{ext}'
        script = [f'file out {output_filename} m /overwrite',
                  f'fits read {filename}-r.fits',
                  f'modify doppler {doppler_corr}', 'write']
        script = [line + '\n' for line in script] + ['exit']
        with open('reduction-output.class', 'w') as file:
            file.writelines(script)
        p = subprocess.run(['class', '@reduction-output.class'])
        for filename_ in ([f'{filename}.dat', f'{filename}.fits',
                           f'{filename}-r.dat', f'{filename}-r.fits']):
            os.remove(filename_)
        print()
        print(f'Saved reduced spectrum in {args.folder}{output_filename}.')
    
# Remove temporal files.
for filename in ['reduction-input.class', 'reduction-output.class']:
    if os.path.exists(filename):
        os.remove(filename)
backup_files = glob.glob('*.dat~')
for file in backup_files:
    os.remove(file)

print()