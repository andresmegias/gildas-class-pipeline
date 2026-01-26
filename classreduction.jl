#!/usr/bin/env julia
"""
Automated GILDAS-CLASS Pipeline
-------------------------------
Reduction mode
Version 1.4

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
import PyPlot as plt
import RollingFunctions: rolling
import Statistics as stats
import Random as random
import Distributions as distr
import StatsBase: mad
import Interpolations: LinearInterpolation, Line
import SciPy.interpolate as scipy_interpolate
import Distributions: Normal
import Format: printfmtln
import ArgParse as argparse
import YAML as yaml
import FITSIO as fitsio
import DelimitedFiles as dlmfiles

# Custom functions.

function safe_realpath(path)
    """
    Return the full path corresponding to the input text.

    Parameters
    ----------
    path : String
        Text of the path to be converted.

    Returns
    -------
    full_path : String
        Text of the full path corresponding to the input text.
    """
    if ispath(path)
        return realpath(path)
    end
    if path != ""
        a, b = splitdir(path)
        return joinpath(safe_realpath(a), b)
    else
        return ""
    end
 end

function madn(x::Array{Float64})
    """
    Return the normalized median absolute deviation (MAD) of the input data.
    """
    y = mad(x, normalize=true)
    return y
end

function rolling_function(func::Function, y::Vector{Float64}, roll_size::Int)
    """
    Apply a function in a rolling way, in windows of the specified size.

    Parameters
    ----------
    y : Vector
        Input data.
    func : Function
        Function to be applied.
    size : Int
        Size of the windows to group the data. It must be odd.

    Returns
    -------
    y_f : array
        Resultant array.
    """
    min_size = 3
    roll_size = Int(roll_size) + (Int(roll_size) + 1) % 2
    roll_size = max(min_size, roll_size)
    N = length(y)
    y_c = rolling(func, y, roll_size)
    M = min(N, roll_size) ÷ 2
    y_1 = zeros(M)
    y_2 = zeros(M)
    for i in 0:M-1
        j1 = 1
        j2 = max(min_size, 2*i)
        y_1[1+i] = func(y[j1:j2])
        j1 = N + 1 - max(min_size, 2*i)
        j2 = N
        y_2[end-i] = func(y[j1:j2])
    end
    y_f = vcat(y_1, y_c, y_2)
    return y_f
end

function get_windows_from_mask(x::Vector{Float64}, mask::Vector{Bool};
                               margin::Float64=1.0, ref_width::Float64=8.0)
    """
    Return the windows of the empty regions of the input array.

    Parameters
    ----------
    x : Vector
        Input data.
    mask : Vector
        Indices of the empty regions of data.
    margin : Float, optional
        Relative margin added to the windows found initially.
    ref_width: Float, optional
        Reference separation in points between two consecutive windows.

    Returns
    -------
    windows : Vector
        List of the inferior and superior limits of each window.
    """
    resolution = stats.median(abs.(diff(x)))
    all_inds = 1:length(x)
    var_inds = diff(vcat([0], Int.(mask), [0]))
    cond1 = (var_inds .== 1)[1:end-1]  # from 0 to 1
    cond2 = (var_inds .== -1)[2:end]  # from 1 to 0
    inds = vcat(all_inds[cond1], all_inds[cond2])
    inds = reshape(sort(inds), (2,length(inds)÷2))'
    windows = x[inds]
    for i in 1:size(windows)[1]
        window = windows[i,:]
        center = stats.mean(window)
        semiwidth = (window[2] - window[1]) / 2
        semiwidth = max(semiwidth, ref_width/4)
        semiwidth += margin
        windows[i,:] = [center - semiwidth, center + semiwidth]
    end
    i = 1
    while i <= size(windows)[1]
        x1, x2 = windows[i,:]
        if (x2 - x1) > 6.0*ref_width*resolution
            windows = windows[1:end .!= i, :]
        else
            i += 1
        end
    end
    if windows[1,1] <= x[1]
        windows = windows[1:end .!= 1, :]
    end
    if windows[end,2] >= x[end]
        windows = windows[1:end .!= length(windows), :]
    end
    return windows
end

function get_mask_from_windows(x::Vector{Float64}, windows::Matrix{Float64};
                               margin::Float64=0.0)
    """
    Obtain the mask of the input array defined by the given windows.

    Parameters
    ----------
    x : Vector
        Input data.
    windows : Matrix
        Windows that specify the regions of the data.

    Returns
    -------
    mask : Vector
        Resultant condition array.
    """
    mask = ones(Bool, length(x))
    dx = stats.median(diff(x))
    for i in 1:size(windows)[1]
        x1, x2 = windows[i,1], windows[i,2]
        mask .*= ((x .<= x1 - dx*margin) .+ (x .>= x2 + dx*margin))
    end
    return mask
end

function sigma_clip_mask(y::Vector{Float64}; sigmas::Float64=6.0, iters::Int=2)
    """
    Apply a sigma clip and return a mask of the remaining data.

    Parameters
    ----------
    y : Vector
        Input data.
    sigmas : Float, optional
        Number of standard deviations used as threshold. The default is 6.0.
    iters : Int, optional
        Number of iterations performed. The default is 2.

    Returns
    -------
    mask : Vector
        Mask of the remaining data after applying the sigma clip.
    """
    mask = ones(Bool, length(y))
    abs_y = abs.(y)
    for i in 1:iters
        mask .*= abs_y .< sigmas*madn(abs_y[mask])
    end
    return mask
end

function fit_baseline(x::Vector{Float64}, y::Vector{Float64};
                      windows::Matrix{Float64}, smooth_factor::Int)
    """
    Fit the baseline of the curve ignoring the specified windows.

    Parameters
    ----------
    x : Vector
        Independent variable.
    y : Vector
        Dependent variable.
    windows : Matrix
        Windows that specify the regions of the data.
    smooth_factor : Int
        Size of the filter applied for the fitting of the baseline.

    Returns
    -------
    yf : Vector
        Baseline of the curve.
    """
    mask = get_mask_from_windows(x, windows)
    x_ = x[mask]
    y_ = y[mask]
    y_s = rolling_function(stats.median, y_, smooth_factor)
    s = sum((y_s - y_).^2)
    spl = scipy_interpolate.UnivariateSpline(x_, y_, s=s)
    yf = spl(x)
    return yf
end

function load_spectrum(file::String; load_fits::Bool=false)
    """
    Load the spectrum from the given input file.

    Parameters
    ----------
    file : String
        Path of the plain text file (.dat) to load, without the extension.
    load_fits : Bool
        If true, load also a .fits file and return the HDU list.

    Returns
    -------
    x : Array
        Frequency.
    y : Array
        Intensity.
    hdul : HDU list (FITSIO)
        List of the HDUs (Header Data Unit).
    """
    data = dlmfiles.readdlm("$file.dat")
    x = data[:,1]
    y = data[:,2]
    if sum(isnan.(data)) != 0
        println("Data of file $file is corrupted.")
        throw(Exception)
    end
    if load_fits
        hdul = fitsio.FITS("$file.fits")
    else
        hdul = nothing
    end
    return x, y, hdul
end

function save_yaml_dict(dictionary::Dict, file_path::String, replace::Bool=false)
    """
    Save the input YAML dictionary into a file.

    Parameters
    ----------
    dictionary : Dict
        Dictionary that wants to be saved.
    file_path : String
        Path of the output file.
    default_flow_style : Bool, optional
        The flow style of the output YAML file. The default is False.
    replace : Bool, optional
        If true, replace the output file in case it existed. If false, load the
        existing output file and merge it with the input dictionary.
        The default is false.

    Returns
    -------
    nothing.
    """
    file_path = safe_realpath(file_path)
    if ! replace & isfile(file_path)
        old_dict = yaml.load_file(file_path)
        new_dict = merge(old_dict, dictionary)
    else
        new_dict = dictionary
    end
    yaml.write_file(file_path, new_dict)
end

function get_rms_noise(x::Vector{Float64}, y::Vector{Float64},
                       windows::Matrix{Float64}=[];
                       sigmas::Float64=6., margin::Float64=0., iters::Int=3)
    """
    Obtain the RMS noise of the input data, ignoring the given windows.

    Parameters
    ----------
    x : Vector
        Independent variable.
    y : Vector
        Dependent variable.
    windows : Matrix, optional
        Windows of the independent variable that will be avoided in the
        calculation of the RMS noise. The default is [].
    sigmas : Float, optional
        Number of deviations used as threshold for the sigma clip applied to
        the data before the calculation of the RMS noise. The default is 6.0.
    margin : Float, optional
        Relative frequency margin that will be ignored for calculating the RMS
        noise. The default is 0.
    iters : Int, optional
        Number of iterations performed for the sigma clip applied to the data
        before the calculation of the RMS noise. The default is 3.

    Returns
    -------
    rms_noise : Float
        Value of the RMS noise of the data.
    """
    N = length(x)
    i1, i2 = Int(round(margin*N)), Int(round((1-margin)*N))+1
    x = x[i1:i2]
    y = y[i1:i2]
    mask = get_mask_from_windows(x, windows)
    y = y[mask]
    mask = sigma_clip_mask(y, sigmas=sigmas, iters=iters)
    y = y[mask]
    rms_noise = √(stats.mean(y.^2))
    return rms_noise
end

function find_rms_region(x::Array{Float64}, y::Array{Float64}; rms_noise::Float64,
                         windows::Matrix{Float64}=[], rms_threshold::Float64=0.1,
                         offset_threshold::Float64=0.05, reference_width::Int=200,
                         min_width::Int=120, max_iters::Int=1000)
    """
    Find a region of the input data that has a similar noise than the one given.

    Parameters
    ----------
    x : Vector
        Independent variable.
    y : Vector
        Dependent variable.
    rms_noise : Float
        The value of the RMS used as a reference.
    windows : Matrix, optional
        The regions of the independent variable that should be ignored.
        The default is [].
    rms_threshold : Float, optional
        Maximum relative difference that can exist between the RMS noise of the
        searched region and the reference RMS noise. The default is 0.1.
    offset_threshold : Float, optional
        Maximum value, in units of the reference RMS noise, that the mean value
        of the dependent variable can have in the searched region.
        The default is 0.05.
    reference_width : Int, optional
        Size of the desired region, in number of channels. The default is 200.
    min_width : Int, optional
        Minimum size of the desired region, in number of channels.
        The default is 120.
    max_iters : Int, optional
        Maximum number of iterations that will be done to find the desired
        region. The default is 1000.

    Returns
    -------
    rms_region : Vector
        Frequency regions of the desired region.
    """
    central_freq, width, resolution = 0, 0, 0
    i = 0
    local_rms = 0
    offset = 1*rms_noise
    while ! (((abs(local_rms - rms_noise) / rms_noise) < rms_threshold)
               & ((abs(offset) / rms_noise) < offset_threshold))
        width = max(min_width, reference_width)
        resolution = stats.median(stats.diff(x))
        central_freq = distr.rand(distr.Uniform(x[1] + width*resolution,
                                                x[end] - width*resolution))
        region_inf = central_freq - width/2*resolution
        region_sup = central_freq + width/2*resolution
        mask = (x .> region_inf) .* (x .< region_sup)
        y_ = y[mask]
        valid_range = true
        for j in 1:size(windows)[1]
            x1, x2 = windows[j,:]
            if (region_inf < x1 < region_sup) | (region_inf < x2 < region_sup)
                valid_range = false
            end
        end
        if valid_range
            local_rms = √(stats.mean(y_.^2))
            offset = stats.mean(y_)
        end
         i += 1
        if i > max_iters
            return []
        end
    end
    rms_region = [central_freq - width/2*resolution,
                  central_freq + width/2*resolution]
    return rms_region
end

# Arguments.
aps = argparse.ArgParseSettings()

argparse.@add_arg_table! aps begin
    "folder"
    arg_type = String
    required = true
    "file"
    arg_type = String
    required = true
    "--smooth_factor"
    arg_type = Int
    default = 20
    "--rms_margin"
    arg_type = Float64
    default = 0.1
    "--plots_folder"
    arg_type = String
    default = "plots"
    "--save_plots"
    action = :store_true
end

#%%

args = argparse.parse_args(aps)
original_folder = safe_realpath(pwd())
cd(safe_realpath(args["folder"]))

all_windows = yaml.load_file("frequency_windows.yaml")
rms_noises = Dict()
frequency_ranges = Dict()
reference_frequencies = Dict()
rms_regions = Dict()
resolutions = Dict()

for file in split(args["file"], ",")
    # Loading of the data files.
    frequency, intensity, hdul = load_spectrum(String(file), load_fits=true)
    fits_data = fitsio.read(hdul[1])
    fits_header = fitsio.read_header(hdul[1])
    resolutions[file] = fits_header["CDELT1"] / 1e6
    reference_frequencies[file] = fits_header["RESTFREQ"] / 1e6
    frequency_ranges[file] = [frequency[1], frequency[end]]
    # Reduction.
    if haskey(all_windows, file)
        windows = all_windows[file]
    else
        windows = []
    end
    windows = Matrix(hcat(windows...)')
    intensity_cont = fit_baseline(frequency, intensity, windows=windows,
                                  smooth_factor=args["smooth_factor"])
    intensity_red = intensity .- intensity_cont
    # Noise.
    rms_noise = get_rms_noise(frequency, intensity_red, windows,
                              sigmas=6.0, margin=args["rms_margin"], iters=3)
    rms_noises[file] = 1e3*rms_noise
    # Noise regions.
    rms_region =
        find_rms_region(frequency, intensity_red, rms_noise=rms_noise,
                        windows=windows, rms_threshold=0.1,
                        offset_threshold=0.05, reference_width=2*args["smooth_factor"])
    if length(rms_region) == 0
        println("Warning: No RMS region was found for spectrum $file.")
    rms_region = [Float32(frequency[1]), Float32(frequency[end])]
    rms_regions[file] = rms_region
    end

    # Output.
    output_file = "$file-r"
    fits_data[:,1,1,1] = intensity_red
    fits_data = convert(Array{Float32}, fits_data)
    output_fits = fitsio.FITS("$output_file.fits", "w")
    fitsio.write(output_fits, fits_data, header=fits_header)
    fitsio.close(output_fits)
    output_data = zeros(length(frequency), 2)
    output_data[:,1] = frequency
    output_data[:,2] = intensity_red
    dlmfiles.writedlm("$output_file.dat", output_data, "\t")
    println("Saved reduced spectrum in $(args["folder"])$file.fits.")
    println("Saved reduced spectrum in $(args["folder"])$file.dat.")

    if args["save_plots"]

        plt.figure(1, figsize=(10,7))
        plt.clf()
        plt.subplots_adjust(hspace=0, wspace=0)

        sp1 = plt.subplot(2,1,1)
        plt.step(frequency, intensity, where="mid", color="black", ms=6)
        for i in 1:size(windows)[1]
            x1, x2 = windows[i,:]
            plt.axvspan(x1, x2, color="gray", alpha=0.3)
        end
        plt.plot(frequency, intensity_cont, "tab:green", label="fitted baseline")
        plt.ticklabel_format(style="sci", useOffset=false)
        plt.margins(x=0)
        plt.xlabel("frequency (MHz)")
        plt.ylabel("original intensity (K)")
        plt.legend(loc="upper right")

        plt.subplot(2,1,2, sharex=sp1)
        plt.step(frequency, intensity_red, where="mid", color="black")
        for i in 1:size(windows)[1]
            x1, x2 = windows[i,:]
            plt.axvspan(x1, x2, color="gray", alpha=0.3)
        end
        plt.ticklabel_format(style="sci", useOffset=false)
        plt.margins(x=0)
        plt.xlabel("frequency (MHz)")
        plt.ylabel("reduced intensity (K)")

        title = "Full spectrum - $file"
        fontsize = max(7, 12 - 0.1*max(0, length(title) - 85))
        plt.suptitle(title, fontsize=fontsize, fontweight="semibold")
        plt.tight_layout(pad=0.7, h_pad=1.0)

        if args["save_plots"]
            cd(original_folder)
            cd(safe_realpath(args["plots_folder"]))
            plt.savefig("spectrum-$file.png", dpi=200)
            cd(original_folder)
            cd(safe_realpath(args["folder"]))
            print("    ")
            println("Saved plot in $(args["plots_folder"])spectrum-$file.png.")
        end

    end

    println(" ")

end

# Export of the rms noise of each spectrum.
save_yaml_dict(rms_noises, "rms_noises.yaml")
println("Saved RMS noises in $(args["folder"])rms_noises.yaml.")

# Export of the frequency ranges of each spectrum.
save_yaml_dict(frequency_ranges, "frequency_ranges.yaml")
println("Saved frequency ranges in $(args["folder"])frequency_ranges.yaml.")

# Export of the reference frequencies of each spectrum.
save_yaml_dict(reference_frequencies, "reference_frequencies.yaml")
println("Saved RMS regions in $(args["folder"])reference_frequencies.yaml.")

# Export of the RMS regions of each spectrum.
save_yaml_dict(rms_regions, "rms_regions.yaml")
println("Saved RMS regions in $(args["folder"])rms_regions.yaml.")

# Export of the frequency resolution of each spectrum.
save_yaml_dict(resolutions, "frequency_resolutions.yaml")
println("Saved frequency resolutions in $(args["folder"])frequency_resolutions.yaml.")

println()