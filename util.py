"""util.py

This file contains helper functions for the main source code in Vulcan.

Created by: McKenna Leichty
Last Updated: Oct 13, 2025

"""
#import modules
import astropy.units as u
import astropy.io.fits as fits
from astropy.wcs import WCS
import numpy as np
from spectral_cube import SpectralCube
import streamlit as st
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from scipy.ndimage import generic_filter
from vorbin.voronoi_2d_binning import voronoi_2d_binning
import plotly.express as px

def open_file(data_file):
    #this function is just to get the science cube information
    with fits.open(data_file) as hdul:
        cube = SpectralCube.read(hdul, hdu=1, format='fits') #science data
    return cube

def spectral_convert(data_file, z, plot = False):
    """
    Convert the spectral axis of the IFU cube to the rest frame.

    Parameters
    ----------
    data_file: Path to the MRS cube FITS file (oo3).
    z: Redshift of the cluster.

    Returns
    -------
    adjusted_cube: Redshift-corrected cube.
    adjusted_cube_err: Redshift-corrected error cube.
    adjusted_spectral_axis: wavelength array corrected for redshift.
    header_primary: FITS header.
    """
    #read in the file
    with fits.open(data_file) as hdul:
        header_primary = hdul[0].header #header of FITS file
        header_cube = hdul[1].header #header for science cube
        cube_err_data = hdul[2].data #error data
        data_quality = hdul[3].data

        #need to do a data quality check of our pixels
        #mask where True = good pixel, False = bad pixel
        dq_mask = data_quality == 0  #pixels with a value of 0 are good pixels (pixels with a value of 513 just 
                                    # mean they're not included in the FOV of the instrument)
        
        if plot:
            
            col1, col2 = st.columns(2)
            with col1:
                good_fraction = dq_mask.sum(axis=0) / dq_mask.shape[0]  #fraction of good pixels at each pixel
                fig = px.imshow(
                    good_fraction,
                    origin='lower',
                    title="Fraction of Good Pixels per Spaxel",
                )
                # Add custom hover label
                fig.update_traces(
                    hovertemplate="X: %{x}<br>Y: %{y}<br>Fraction: %{z:.3f}<extra></extra>"
                )
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        title=dict(text="", font=dict(color='black')),
                        #tickvals=tickvals,
                        #ticktext=quality_labels,
                        tickfont=dict(color='black')
                    ),
                    xaxis=dict(title='X Pixel', title_font=dict(color='black'), tickfont=dict(color='black')),
                    yaxis=dict(title='Y Pixel', title_font=dict(color='black'), tickfont=dict(color='black')),
                    hoverlabel=dict(font=dict(color='black')),
                    width=600,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                #ChatGPT-4o was used on Oct 14, 2025 to generate the html code to center my words with the above plot
                st.markdown(
                    """
                    <div style='display: flex; flex-direction: column; justify-content: center; height: 500px;'>
                        <p>
                        JWST IFU products come with a data quality cube that labels each pixel with 
                        a number that corresponds to some flag. A list of these data quality flags can be found
                        in the <a href='https://jwst-pipeline.readthedocs.io/en/stable/jwst/references_general/references_general.html#data-quality-flags' target='_blank'>JWST docs</a>.
                        Based on these flags, Vulcan will automatically apply a mask to our data and error cubes to
                        ignore any pixels that are not "good".
                        </p>
                        <p>
                        The plot on the right shows the fraction of good pixels per spaxel in our A2597 dataset. 
                        Pixels near the edge of our FOV are more likely to deal with something called "edge effects",
                        which are artifacts of how the data was collected and sampled. It is something to keep in mind
                        while exploring the rest of Vulcan's 2D maps, as pixels near the edges could be less reliable
                        when telling our story.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        cube = SpectralCube.read(hdul, hdu=1, format='fits') #science data
        wcs = WCS(header_cube, hdul) #wcs information
        cube_err = SpectralCube(data=cube_err_data, wcs=wcs) #creating a SpectralCube instance of the error data
        spectral_axis = cube.spectral_axis #wavelength
        
        #apply the data quality mask to the science cube and error cube
        masked_cube = cube.with_mask(dq_mask)
        masked_cube_err = cube_err.with_mask(dq_mask)
    
    #put spectral axis in rest frame wavelength
    if z == 0.00:
        st.info("Redshift currently set to 0. To convert to rest frame, enter in the BCG's redshift in the sidebar input.")
        adjusted_cube = masked_cube
        adjusted_cube_err = masked_cube_err
        adjusted_spectral_axis = spectral_axis
    else:
        with st.spinner("Converting to rest frame wavelength..."):
            adjusted_spectral_axis = (spectral_axis / (1 + z))  #in microns

            #our adjusted WCS information
            new_wcs = masked_cube.wcs.deepcopy()

            original_crval = new_wcs.wcs.crval[2]
            original_cdelt = new_wcs.wcs.cdelt[2]
            
            new_crval = original_crval / (1 + z)
            new_cdelt = original_cdelt / (1 + z)

            new_wcs.wcs.crval[2] = new_crval
            new_wcs.wcs.cdelt[2] = new_cdelt

            adjusted_cube = SpectralCube(data=masked_cube.unmasked_data[:], wcs=new_wcs)
            adjusted_cube_err = SpectralCube(data=masked_cube_err.unmasked_data[:], wcs=new_wcs)
            
        st.success("Rest frame conversion complete!")

    return adjusted_cube, adjusted_cube_err, adjusted_spectral_axis, header_primary

def isolate_wavelength(cube_um, lower_l, upper_l, local=0.1):
    """
    This function takes the cube data and isolates at a given lower and upper wavelength to create a subcube. It will return this subcube as a 2D image.
    Then, it will grab a spectral slab of the local continuum on either side of the isolated wavelength based on the definition of local.
    The original subcube will have these local cont. slabs subtracted from it.

    Parameters
    ----------
    cube: data read in as a cube
    lower_l (float): lower wavelength bound in microns
    upper_l (float): upper wavelength bound in microns
    local (float): how many microns out from the isolated wavelength to define the local continuum

    Returns
    -------
    cont_img: 2D image before subtraction
    subtract_img: 2D image of subtracted local continuum
    loc_cont: 2D image of the local continuum
    
    """
    #spectral slab of the actual emission line
    subcube = cube_um.spectral_slab(lower_l*u.micron, upper_l*u.micron)

    #make a 2D image from the 3D subcube with sum
    cont_img = subcube.sum(axis=0) #sum along spectral axis

    loc_cont1 = cube_um.spectral_slab(lower_l*u.um - local*u.um, lower_l*u.um) #left side of the isolated wavelength        
    loc_cont2 = cube_um.spectral_slab(upper_l*u.um, upper_l*u.um + local*u.um) #right side

    #subtracted continuum final image
    loc_cont = loc_cont1.mean(axis=0)+loc_cont2.mean(axis=0) #total local continuum on both sides        
    subtract_img = cont_img - loc_cont

    return cont_img, subtract_img, loc_cont

def compute_sn_map(cube, cube_err):
    """
    Computes S/N map. Assumes Poisson error when calculating noise. Sums flux over
    spectral axis and converts to Jy / pixel. If the user wishes to get accurate signal,
    the inputted cube and cube_err should be spectral slabs surrounding the signal.

    The error is assumed to be the error from the cube_err added in quadrature.
    """
    flux_data = cube.unmasked_data[:].value
    err_data = cube_err.unmasked_data[:].value

    flux_map = np.sum(flux_data, axis=0) # units of MJy/sr
    error_map = np.sqrt(np.sum(err_data**2, axis=0)) # units of MJy/sr, errors added in quadrature

    return flux_map, error_map

def run_voronoi(signal_map, noise_map, target_sn, pixel_size = None, plot=False, quiet = False):
    """
    Uses voronoi_2d_binning() from the package Vorbin. Description of parameters and returns 
    can be found here: https://pypi.org/project/vorbin/#voronoi-2d-binning-function. 

    Parameters
    ----------
    signal_map (2d array): 2d array of signal in each pixel
    noise_map (2d array): 2d array of the noise in each pixel (coming from the error data cube) 
    target_sn (float): the signal to noise threshold one wants to bin to
    plot (Bool): if true, plots the binning results from the voronoi function
    quiet (Bool): if true, prints information about binning 

    Returns
    -------
    bin_num: index for each pixel
    x_good: x pixel coords
    y_good: y pixel coords
    sn (np.array): SNR of each region (or pixel)
    
    """
    y_indices, x_indices = np.indices(signal_map.shape)
    
    x = x_indices.flatten().astype(float)
    y = y_indices.flatten().astype(float)
    signal = signal_map.flatten()
    noise = noise_map.flatten() # noise is based on the error from the error cube

    #our mask is anywhere where the signal is positive
    good = signal > 0

    x_good = x[good]
    y_good = y[good]
    signal_good = signal[good]
    noise_good = noise[good]
    
    try:
        bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale = voronoi_2d_binning(x_good, y_good, signal_good, noise_good, target_sn, plot=plot, quiet=quiet, pixelsize = pixel_size)
    except:
        #if it doesn't bin because SNR is too high in each pixel, just return the array as it is without binning
        return 

    return bin_num, x_good, y_good, sn

def run_voronoi_(signal_map, noise_map, target_sn, pixel_size=None, plot=False, quiet=False):
    """
    Uses voronoi_2d_binning() from the package Vorbin. If all pixels have S/N >= target_sn, 
    returns each pixel as its own bin.
    
    Parameters
    ----------
    signal_map (2d array): 2d array of signal in each pixel
    noise_map (2d array): 2d array of the noise in each pixel
    target_sn (float): the signal to noise threshold one wants to bin to
    plot (Bool): if true, plots the binning results from the voronoi function
    quiet (Bool): if true, suppresses output messages

    Returns
    -------
    bin_num (np.array): array of bin numbers corresponding to good pixels
    x_good (np.array): x coordinates of good pixels
    y_good (np.array): y coordinates of good pixels
    sn (np.array): SNR of each bin (or pixel if unbinned)
    """
    y_indices, x_indices = np.indices(signal_map.shape)

    x = x_indices.flatten().astype(float)
    y = y_indices.flatten().astype(float)
    signal = signal_map.flatten()
    noise = noise_map.flatten()

    #define valid (good) pixels
    good = (signal > 0) & (noise > 0)
    x_good = x[good]
    y_good = y[good]
    signal_good = signal[good]
    noise_good = noise[good]

    #compute S/N for good pixels
    snr_per_pixel = signal_good / noise_good

    if np.all(snr_per_pixel >= target_sn):
        #if all pixels meet the SNR requirement, assign each to its own bin
        #print('All pixels are above the SNR. Assigning each pixel to its own bin.')
        bin_num = np.arange(len(signal_good))
        sn = snr_per_pixel
        return bin_num, x_good, y_good, sn

    #otherwise, proceed with binning
    try:
        bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale = voronoi_2d_binning(
            x_good, y_good, signal_good, noise_good, target_sn, plot=plot, quiet=quiet, pixelsize=pixel_size
        )
        return bin_num, x_good, y_good, sn
    except Exception as e:
        if not quiet:
            print(f"Voronoi binning failed: {e}") #not necessary in streamlit, but I use this in my own code
        return None

def make_bin_map(bin_num, x_good, y_good, shape):
    """
    Takes the outputs of voronoi binning and maps out the pixels and assigns them
    their region number.
    """
    bin_map = -1 * np.ones(shape, dtype=int)
    for i in range(len(bin_num)):
        bin_map[int(y_good[i]), int(x_good[i])] = bin_num[i]
    return bin_map

def extract_bin_arrays(cube, cube_err, bin_map, pixel_area_sr):
    """
    Returns
    -------
    bin_fluxes: fluxes for each pixel at each wavelength
    bin_errors: associated errors
    bin_masks: basically the bin_map thing
    """
    bin_ids = np.unique(bin_map[bin_map >= 0])
    n_bins = len(bin_ids)
    n_spec = cube.shape[0]
    ny, nx = bin_map.shape

    bin_fluxes = np.zeros((n_bins, n_spec))
    bin_errors = np.zeros((n_bins, n_spec))
    bin_masks = np.zeros((n_bins, ny, nx), dtype=bool)

    for i, bin_id in enumerate(bin_ids):
        #create 2D mask for the bin
        mask = (bin_map == bin_id)
        bin_masks[i] = mask

        #get pixel indices for the bin
        y_idx, x_idx = np.where(mask)

        total_flux = np.zeros(n_spec)
        total_var = np.zeros(n_spec)

        for y, x in zip(y_idx, x_idx):
            #this is converting our intensity to flux
            flux = cube[:, y, x].value * 1e6 * pixel_area_sr #in Jy
            err = cube_err[:, y, x].value * 1e6 * pixel_area_sr #in Jy
            total_flux += flux
            total_var += err**2

        bin_fluxes[i] = total_flux
        bin_errors[i] = np.sqrt(total_var)

    return bin_fluxes, bin_errors, bin_masks

def gaussian_with_baseline(x, amp, mean, stddev, slope, intercept):
    """
    Model combining a Gaussian curve and a linear baseline.
    """
    gaussian = amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    baseline = slope * x + intercept
    return gaussian + baseline

def gaussian_fitter(peak_wavelength, fit_width, wavelengths, flux, flux_err, title, truncate_side = None, truncate_percent = 0.0, plot = False):
    """
    Fits a Gaussian on top of a linear baseline at a given peak_wavelength 
    and returns the fit parameters. Allows optional truncation of one side.
    If the fit is bad, it will return NaN vals for the fit parameters.

    Parameters
    ----------
    peak_wavelength (float): wavelength to make the gaussian fit at
    fit_width (float): how far out in um should the Gaussian consider when fitting
    wavelengths (np.array): array of wavelengths from 1D spec file
    flux (np.array): array of fluxes from 1D spec file
    flux_err (np.array): array of errors corresponding to the fluxes
    truncate_side (str or None): 'left', 'right', or None for no truncation. 
                                 Specifies side to exclude from fit.
    truncate_percent (float): how much of the fit_width should we disclude from our truncation?

    Returns
    -------
    popt (list of floats): returns the fitted amplitude, mean, stddev, slope, and intercept
    stddev_fit (float): line width (standard deviation) from gaussian fit
    stddev_uncertainty (float): uncertainty of linewidth from covariance matrix of gaussian fit
    popt_errs (array of floats): uncertainties of popt parameters from covariance matrix of gaussian fit
    """

    #define the fitting range
    fit_mask = (wavelengths > (peak_wavelength - fit_width)) & (wavelengths < (peak_wavelength + fit_width))
    wavelengths_fit = wavelengths[fit_mask]
    flux_fit = flux[fit_mask]
    flux_err_fit = flux_err[fit_mask]

    #estimate a preliminary stddev for Gaussian truncation
    estimated_stddev = truncate_percent * fit_width

    #apply truncation if specified (never specified in Vulcan, but I needed this for my own code)
    if truncate_side == 'left':
        truncation_point = peak_wavelength - estimated_stddev #get the rest of the gaussian but not the local cont.
        mask = wavelengths_fit > truncation_point #create a mask at this point to ignore those values past truncation_point
        wavelengths_fit = wavelengths_fit[mask]
        flux_fit = flux_fit[mask]
        flux_err_fit = flux_err_fit[mask]

    elif truncate_side == 'right':
        truncation_point = peak_wavelength + estimated_stddev
        mask = wavelengths_fit < truncation_point
        wavelengths_fit = wavelengths_fit[mask]
        flux_fit = flux_fit[mask]
        flux_err_fit = flux_err_fit[mask]

    #initial guesses for the parameters
    initial_guess = [np.max(flux_fit) - np.min(flux_fit), peak_wavelength, 0.01, 0, np.min(flux_fit)]

    try:
        popt, pcov = curve_fit(gaussian_with_baseline, wavelengths_fit, flux_fit, sigma=flux_err_fit, p0=initial_guess, maxfev=5000,
                               bounds=([0, peak_wavelength - fit_width, 1e-5, -np.inf, -np.inf], [np.inf, peak_wavelength + fit_width, np.inf, np.inf, np.inf])
                               )
        amp_fit, mean_fit, stddev_fit, slope_fit, intercept_fit = popt
        popt_errs = np.sqrt(np.diag(pcov)) #one standard deviation errors on the parameters

    except (RuntimeError, ValueError):
        #if the fit fails, just return NaN vals
        return [np.nan]*5, np.nan, np.nan, np.array([np.nan]*5)

    #get the uncertainty in stddev_fit from the covariance matrix
    stddev_uncertainty = np.sqrt(pcov[2, 2])

    #now generating errors on model's flux
    #generate fit curve
    fit_vals = gaussian_with_baseline(wavelengths_fit, *popt)

    #propagate uncertainty to each point on the curve using the Jacobian matrix
    J = np.zeros((len(wavelengths_fit), len(popt)))

    for i, x in enumerate(wavelengths_fit):
        #partial derivatives
        d_amp = np.exp(-((x - mean_fit) ** 2) / (2 * stddev_fit ** 2))
        d_mean = amp_fit * d_amp * ((x - mean_fit) / (stddev_fit ** 2))
        d_stddev = amp_fit * d_amp * (((x - mean_fit) ** 2) / (stddev_fit ** 3))
        d_slope = x
        d_intercept = 1
        J[i, :] = [d_amp, d_mean, d_stddev, d_slope, d_intercept]

    #variance at each x: diagonal of J @ pcov @ J.T
    fit_uncertainty = np.sqrt(np.sum(J @ pcov * J, axis=1))

    #plot gaussian fit if the user wants to visualize
    if plot:
        fig = go.Figure()

        #original data
        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=flux,
            mode='lines',
            name='Original Data',
            line=dict(color='blue')
        ))

        #data used for fitting
        fig.add_trace(go.Scatter(
            x=wavelengths_fit,
            y=flux_fit,
            mode='lines',
            name='Data for Fit',
            line=dict(color='green')
        ))

        #fit +/- 1 sigma as a filled error band
        fig.add_trace(go.Scatter(
            x=np.concatenate([wavelengths_fit, wavelengths_fit[::-1]]),
            y=np.concatenate([fit_vals + fit_uncertainty, (fit_vals - fit_uncertainty)[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='Fit ± 1σ'
        ))

        #model line
        fig.add_trace(go.Scatter(
            x=wavelengths_fit,
            y=fit_vals,
            mode='lines',
            name='Fit',
            line=dict(color='red')
        ))

        fig.update_layout(
            title=fr'Gaussian Fit of the {title} line',
            legend=dict(title='Legend', title_font=dict(color='black')),
            xaxis=dict(range=[peak_wavelength - 0.2, peak_wavelength + 0.2], title='Wavelength (μm)', title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title='Flux (Jy per pixel)', title_font=dict(color='black'), tickfont=dict(color='black')),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        #st.markdown("should also return chi^2 for these fits...")

    return popt, stddev_fit, stddev_uncertainty, popt_errs

def extracted_vals_from_gaussian(peak_wavelengths, fit_width, wavelengths, region_flux, region_flux_err, plot = False):
    """
    This function fits a gaussian to every peak wavelength and returns
    the line width, the average rest wavelength (from the fit), and their 
    corresponding uncertainties.
    """
    lws, lws_err, mean_fits, mean_fits_errs = [], [], [], []
    amp_fits, amp_fit_errs = [], []
    
    titles = ['H2(S2)', '[NeII]', '[NeIII]']
    for i in range(len(peak_wavelengths)): #for each peak wavelength...
        popt, lw, lw_err, popt_errs = gaussian_fitter(peak_wavelengths[i], fit_width, wavelengths, region_flux, region_flux_err, titles[i], plot = plot)
        #popt = amp_fit, mean_fit, stddev_fit, slope_fit, intercept_fit
        lws.append(lw)
        lws_err.append(lw_err)
        mean_fits.append(popt[1])
        mean_fits_errs.append(popt_errs[1])
        amp_fits.append(popt[0])
        amp_fit_errs.append(popt_errs[0])

    return lws, lws_err, mean_fits, mean_fits_errs, amp_fits, amp_fit_errs

def impute_with_neighbors(data, quality, required_quality=1, min_neighbors=3):
    """
    Impute NaNs where quality is equal to some required quality using average of at least k valid neighbors.
    Starts with a default value of k=3. Doing more than 3 won't really do anything for this dataset.
    """
    def neighbor_mean_filter(values):
        neighbor_vals = np.delete(values, len(values) // 2) #remove center pixel from calculation of mean
        valid_vals = neighbor_vals[~np.isnan(neighbor_vals)]
        return np.nanmean(valid_vals) if len(valid_vals) >= min_neighbors else np.nan #return the mean if the number of neighbors is >= k

    #only apply to pixels where quality is at some required quality level (in this case 'poor' quality fits)
    mask_to_impute = (np.isnan(data)) & (quality == required_quality)

    imputed_data = data.copy()
    filtered = generic_filter(data, neighbor_mean_filter, size=3, mode='constant', cval=np.nan) #does this for a 3x3 box excluding the center
    imputed_data[mask_to_impute] = filtered[mask_to_impute] #apply our average to imputed data 2d map

    return imputed_data