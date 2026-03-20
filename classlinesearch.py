#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated GILDAS-CLASS Pipeline
-------------------------------
Line search mode
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
import gc
import sys
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


# Custom functions.

def full_path(text):
    """
    Obtain the full path described in a text string.
    
    Corrects the format of the path, allowing the use of operating system
    abbreviations (for example, './', '../' and '~' for Unix).
    
    Parameters
    ----------
    text : str
        Text of the path, which can include operating system abbreviations.
        
    Returns
    -------
    path : str
        Text of the full path, so that Python can use it.
    """
    path = text
    if path.startswith('~'):
        path = os.path.expanduser('~') + path[1:]  
    path = str(os.path.realpath(path))
    return path

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
    y3 : array
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
        Path of the plain text file (.dat) to load, without the extension.
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
        raise Exception(f'Data of file {filename} is corrupted.')
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

# Arguments.
parser = argparse.ArgumentParser()
parser.add_argument('filenames')
parser.add_argument('-folder', default='.', type=str)
parser.add_argument('-smooth_factor', default=20, type=int)
parser.add_argument('-ref_width', default=6, type=int)
parser.add_argument('-line_sigmas', default=8., type=float)
parser.add_argument('-plots_folder', default='plots', type=str)
parser.add_argument('--save_plots', action='store_true')
args = parser.parse_args()
original_folder = full_path(os.getcwd())
os.chdir(full_path(args.folder))
sep = '\\' if platform.system() == 'Windows' else '/'
if not args.folder.endswith(sep):
    args.folder += sep
    
#%% Calculations.

windows_dict = {}
filenames = args.filename.split(',')
num_files = len(filenames)

# Processing of each spectrum.
for filename in filenames:

    # Remove extension.
    ext = filename.split('.')[-1] if '.' in filename else ''
    if ext != '':
        filename = filename[:-len(ext)-1]
    # If .dat file does not exist, export it from CLASS file.
    if not os.path.exists(f'{filename}.dat'):
        script = [f'file in {filename}', 'find /all', 'list', 'get first',
                  f'greg {filename}.dat /formatted']
        script = [line + '\n' for line in script] + ['exit']
        with open('linesearch-input.class', 'w') as file:
            file.writelines(script)
        p = subprocess.run(['class', '@linesearch-input.class'])
        print()
        if p.returncode == 1:
            sys.exit()
    # Loading of the data file.
    frequency, intensity = load_spectrum(filename)

    # Identification of the lines and reduction of the spectrum.
    windows = identify_lines(frequency, intensity, smooth_factor=args.smooth_factor,
                       ref_width=args.ref_width, sigmas=args.line_sigmas, iters=2)
    intensity_cont = fit_baseline(frequency, intensity, windows, args.smooth_factor)
    intensity_red = intensity - intensity_cont
    
    # Windows.
    if len(windows) != 0:
        num_windows = len(windows)
        print(f'{num_windows} windows identified for {filename}.')
    else:
        windows = np.array([])
        print(f'No lines identified for {filename}.')
    windows_dict[filename] = [[round(float(x1), 6), round(float(x2), 6)]
                          for x1, x2 in windows.tolist()]
    
    gc.collect()
        
    #%% Plots.
    
    if args.save_plots:
        
        y_lims = [np.quantile(intensity, 1e-3), np.quantile(intensity, 1.-1e-3)]
        y_lims = [y_lims[0] - np.diff(y_lims)/10, y_lims[1] + np.diff(y_lims)/10]
        yr_lims = [np.quantile(intensity_red, 5e-4), np.quantile(intensity_red, 1.-5e-4)]
        yr_lims = [yr_lims[0] - np.diff(yr_lims)/5, yr_lims[1] + np.diff(yr_lims)/5]
        
        fig = plt.figure(1, figsize=(10,7))
        plt.clf()
        plt.subplots_adjust(hspace=0, wspace=0)
        
        sp1 = plt.subplot(2,1,1)
        plt.step(frequency, intensity, where='mid', color='black', ms=6)
        for x1, x2 in windows:
            plt.axvspan(x1, x2, color='gray', alpha=0.3)
        plt.plot(frequency, intensity_cont, 'tab:green', label='fitted baseline')
        plt.locator_params(axis='both', nbins=10)
        plt.ticklabel_format(style='sci', useOffset=False)
        plt.margins(x=0)
        plt.ylim(y_lims)
        plt.xlabel('frequency (MHz)')
        plt.ylabel('original intensity (K)')
        plt.legend(loc='upper right')
        plt.title(f'Full spectrum - {filename}', fontweight='semibold', pad=12.)
    
        plt.subplot(2,1,2, sharex=sp1)
        for (x1, x2) in windows:
            plt.axvspan(x1, x2, color='gray', alpha=0.3)
        plt.step(frequency, intensity_red, where='mid', color='black')
        plt.locator_params(axis='both', nbins=10)
        plt.ticklabel_format(style='sci', useOffset=False)
        plt.margins(x=0)
        plt.ylim(yr_lims)
        plt.xlabel('frequency (MHz)')
        plt.ylabel('reduced intensity (K)')

        fig.align_ylabels()
        plt.tight_layout(pad=0.7, h_pad=1.0)
        # plt.show()
        # input('stop')

        dx = np.median(np.diff(frequency)) 
        plt.rcParams['font.size'] = 8
        num_lines = len(windows)
        num_plots = 1 + (num_lines - 1)//15
        for i in range(num_plots):
            fig = plt.figure(2+i, figsize=(12, 6))
            plt.clf()
            for j in range(min(num_lines - 15*i, 15)):
                plt.subplot(3, 5, j+1)
                j += 15*i
                x1, x2 = windows[j]
                margin = max(args.ref_width*dx, 0.4*(x2-x1))
                mask = (frequency > x1 - margin) * (frequency < x2 + margin)
                xj = frequency[mask]
                yrj = intensity_red[mask]
                plt.step(xj, yrj, where='mid', color='black')
                plt.axvspan(x1, x2, color='gray', alpha=0.2)
                plt.margins(x=0, y=0.1)
                plt.locator_params(axis='both', nbins=10)
                if j+1 > min(15*(i+1), num_lines) - 5:
                    plt.xlabel('frequency (MHz)')
                if j%5 == 0:
                    plt.ylabel('reduced intensity (K)')
                plt.xticks(fontsize=6.)
                plt.yticks(fontsize=6.)
                plt.locator_params(axis='x', nbins=1)
                plt.locator_params(axis='y', nbins=3)
                plt.ticklabel_format(style='sci', useOffset=False)
            window_num = f' ({i+1})' if num_plots > 1 else ''
            plt.suptitle(f'Identified lines{window_num} - {filename}',
                         fontweight='semibold')
            fig.align_ylabels()
            plt.tight_layout(pad=1.2, h_pad=0.6, w_pad=0.1)
        
        if args.save_plots:
            os.chdir(original_folder)
            os.chdir(full_path(args.plots_folder))
            plt.figure(1)
            plt.savefig(f'spectrum-{filename}.png')
            print(f"    Saved plot in {args.plots_folder}spectrum-{filename}.png.")
            for i in range(num_plots):
                plt.figure(2+i)
                plt.savefig(f'lines-{filename}_{i+1}.png')
                print(f"    Saved plot in {args.plots_folder}lines-{filename}_{i+1}.png.")
            print()
            os.chdir(original_folder)
            os.chdir(full_path(args.folder))

        plt.close('all')
            
    gc.collect()
    
# Export of the frequency windows of each spectrum.
save_yaml_dict(windows_dict, 'frequency_windows.yaml', default_flow_style=None)
print(f'Saved windows in {args.folder}frequency_windows.yaml.')

# Remove temporal CLASS file.
if os.path.exists('linesearch-input.class'):
    os.remove('linesearch-input.class')

print()