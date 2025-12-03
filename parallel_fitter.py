# parallel_fitter.py
import numpy as np
from scipy.optimize import curve_fit

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
    
    return popt, stddev_fit, stddev_uncertainty, popt_errs

def extracted_vals_from_gaussian(peak_wavelengths, fit_width, wavelengths, region_flux, region_flux_err):
    """
    This function fits a gaussian to every peak wavelength and returns
    the line width, the average rest wavelength (from the fit), and their 
    corresponding uncertainties.
    """
    lws, lws_err, mean_fits, mean_fits_errs = [], [], [], []
    amp_fits, amp_fit_errs = [], []

    for i in range(len(peak_wavelengths)):
        popt, lw, lw_err, popt_errs = gaussian_fitter(
            peak_wavelengths[i], fit_width,
            wavelengths, region_flux, region_flux_err
        )
        lws.append(lw)
        lws_err.append(lw_err)
        mean_fits.append(popt[1])
        mean_fits_errs.append(popt_errs[1])
        amp_fits.append(popt[0])
        amp_fit_errs.append(popt_errs[0])

    return lws, lws_err, mean_fits, mean_fits_errs, amp_fits, amp_fit_errs
