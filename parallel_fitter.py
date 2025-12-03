# parallel_fitter.py
import numpy as np
from scipy.optimize import curve_fit

# --- COPY THESE FUNCTIONS EXACTLY AS IN YOUR CODE ---
def gaussian_with_baseline(x, amp, mean, stddev, slope, intercept):
    gaussian = amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    baseline = slope * x + intercept
    return gaussian + baseline


def gaussian_fitter(peak_wavelength, fit_width, wavelengths, flux, flux_err):
    # (Same code as before, BUT **remove** all Plotly/Streamlit parts)
    # and remove the "plot" argument
    ...
    return popt, stddev_fit, stddev_uncertainty, popt_errs


def extracted_vals_from_gaussian(peak_wavelengths, fit_width, wavelengths, region_flux, region_flux_err):
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
