# main_page.py
import streamlit as st
import numpy as np
import os
import astropy.units as u
from scipy.signal import find_peaks
from sklearn.preprocessing import OrdinalEncoder

#importing files I wrote
import util as util #useful functions that would otherwise clutter this file up
from sidebar import sidebar_inputs #siderbar values
import plot as plot #visualizations

#start page setup
st.set_page_config(layout="wide")

#banner image
st.image("~/images/banner.jpg", use_container_width=True)

st.set_page_config(page_title="Vulcan", layout="wide")
st.title("VULCAN: Visualization Utility for Luminous Cluster Analysis in NIR/MIR")
st.markdown("""
Welcome to **VULCAN**!

Here you can explore the A2597 dataset from the James Webb Space Telescope and:
- View the full 2D image and spectra
- Subtract the local continuum at determined emission lines
- View spectra at a given pixel
- Fit Gaussian models to emission lines
- Impute based on quality of Gaussian fits
- Make moment maps of emission features (coming soon)
""")

#get sidebar values
uploaded_file, z_input = sidebar_inputs()
#z_input = sidebar_inputs()

#information about this dataset
st.header("Exploring Abell 2597")

#information on A2597
col1, col2 = st.columns([2.5, 1])
with col2:
    st.image("~/images/a2597.jpg", use_container_width=True)
    st.markdown(r"""$\textbf{Figure 1.}$ This image combines X-ray data from NASA's Chandra X-ray Observatory 
                (shown in blue), optical data from the Hubble Space Telescope and the Digitized Sky Survey 
                (in yellow), and hydrogen emission (in red) captured by the Walter Baade Telescope in Chile.
                This picture was taken from the [Chandra Photo Gallery](https://chandra.harvard.edu/photo/2015/a2597/).""")
with col1:
    st.markdown(r"""
**Abell 2597** is a galaxy cluster located approximately 1 billion light-years from Earth in the constellation 
Aquarius. It is one of the most well-studied examples of a cool-core cluster, where the central region 
contains dense, hot gas that cools more rapidly than in other areas of the cluster.

At the center of A2597 is a giant elliptical galaxy, called the Brightest Cluster Galaxy (BCG), hosting a 
supermassive black hole. 
This black hole is notable for displaying a rare "galactic fountain" effect, where it both ejects and 
pulls in massive amounts of cold molecular gas. This cycle fuels both star formation and the black 
hole's activity in a self-regulating feedback loop. To read more about A2597, see
[G. R. Tremblay et al 2018 ApJ 865 13](https://iopscience.iop.org/article/10.3847/1538-4357/aad6dd). Some general
information on this BCG is bulleted below:

- **Right ascension:** 23h 25m 19.70s  
- **Declination:** -12° 07' 27.07"
- **Redshift:** 0.0852  
- **Distance:** 307 Mpc (1,001 Mly)
                
Vulcan provides an interactive environment for analyzing mid-infrared observations captured by JWST's Medium 
Resolution Spectrometer (MRS), using A2597 as a proxy target. By default, it loads Channel 3 data from the MRS, which primarily focuses on 
molecular hydrogen and forbidden neon lines; this dataset was downloaded directly from the Mikulski Archive for 
Space Telescopes (MAST).
""")

st.subheader("IFU Information")
st.markdown("Below includes some general information about the type and shape of our IFU dataset.")
from io import BytesIO

#load primary FITS file
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_to_use = BytesIO(file_bytes)  # wrap in BytesIO for first use
    file_to_use_for_convert = BytesIO(file_bytes)  # new BytesIO for second use
    st.success("Using uploaded FITS file.")
else:
    #default_path = "jw04094-o003_t003_miri_ch3-shortmediumlong_s3d.fits"
    #if os.path.exists(default_path):
    #    file_to_use = default_path
    ##    file_to_use_for_convert = default_path
        #st.info("Using default FITS file.")
    #else:
    st.error("No file uploaded. Go to (link) to test Vulcan on Abell 2597!")
    st.stop()

#print information about the science data cube out
st.write(util.open_file(file_to_use))

if st.checkbox("Show Data Quality Map?"): #if the user wants to view the data quality map produced by JWST pipeline
#to convert to rest frame wavelength
    data_cube, data_cube_err, spectral_axis, header = util.spectral_convert(file_to_use_for_convert, float(z_input), plot = True)
else:
    data_cube, data_cube_err, spectral_axis, header = util.spectral_convert(file_to_use_for_convert, float(z_input), plot = False)

#save to session state
st.session_state.data_cube = data_cube
st.session_state.data_cube_err = data_cube_err
st.session_state.spectral_axis = spectral_axis
st.session_state.header = header
st.session_state.flux = data_cube.sum(axis=(1, 2)).value  #in MJy/sr
st.session_state.flux_err = data_cube_err.sum(axis=(1, 2)).value
st.session_state.wavelengths = spectral_axis.value  #in microns

#to reuse these values later
data_cube = st.session_state.data_cube
data_cube_err = st.session_state.data_cube_err
flux = st.session_state.flux
wavelengths = st.session_state.wavelengths

#find peaks in our spectrum
peaks, _ = find_peaks(flux, height=50000, distance = 200, prominence=10, width = 5, threshold = 700)
peak_wavelengths = wavelengths[peaks]

####################################################
## IMAGE 1: side by side total image and spectrum ##
####################################################

#collapse the cube along the spectral axis
collapsed = data_cube.sum(axis=0).value #shape (y, x)
plot.total_imgs(collapsed, wavelengths, flux, peak_wavelengths, peaks)
#figure caption
st.markdown(r"""$\textbf{Figure 2. (Left)}$ Completely interactive. If you would like to zoom in on a specific region,
            just click and drag to highlight the box to zoom in on. To return to full-view, double click. 
            $\textbf{(Right)}$ The total spectrum of A2597 from Channel 3 of the MRS IFU. The three emission lines that
            are present are H$_2$(S2), [NeII], and [NeIII].

            """)

####################################
## IMAGE 2: local continuum plots ##
####################################

st.header("Spatial Structure of Emission Lines")
st.markdown(r"""This is to show how we background subtract out the local continuum, defined as 
            $\pm 0.1$ um.

            """)

#round the peak wavelengths for display
peak_labels = [fr"H$_2$(S2) at {peak_wavelengths[0]:.3f} μm", f"[NeII] at {peak_wavelengths[1]:.3f} μm", f"[NeIII] at {peak_wavelengths[2]:.3f} μm"]

#let user choose one of the three peak wavelengths
selected_label = st.radio("Choose emission line:", peak_labels)

#get the actual wavelength value from the label
selected_index = peak_labels.index(selected_label)
selected_wavelength = peak_wavelengths[selected_index]

#call isolate_wavelength using the selected wavelength
cont_img, subtract_img, loc_cont_img = util.isolate_wavelength(data_cube, [selected_wavelength - 0.01], [selected_wavelength + 0.01], local=0.1)

#three imagesand their titles that we're going to display next
images = [cont_img, subtract_img, loc_cont_img]
titles = [
    f"Unsubtracted Continuum",
    f"Locally-subtracted Continuum",
    f"Local Continuum"
]

plot.continuum_imgs(images, titles)

#figure caption
st.markdown(r"""$\textbf{Figure 3. (Left)}$ This is the spectral slab of $\pm 0.01$ microns either side 
            the peak wavelength, which includes part of the continuum. $\textbf{(Middle)}$ This is the left plot 
            minus the right plot. It represents the emission line without the local continuum. $\textbf{(Right)}$ 
            This plot is showing the spectral slabs of the local continuum either side the peak wavelength. 
            It's defined as: (peak_wavelength - 0.01) - local, (peak_wavelength - 0.01) for the left side of the 
            emission line, and (peak_wavelength + 0.01), (peak_wavelength + 0.01) + local for the right side. 
            local is defined as 0.1 microns. The red crosshair is showing the center of the BCG as defined in 
            Donahue et al. 2011.

            """)

############################################
## IMAGE 3: side by side image and spaxel ##
############################################

st.header("Viewing Spaxels")
n_spec, y_dim, x_dim = data_cube.shape
x_pixel = st.number_input("X Pixel", min_value=0, max_value=x_dim - 1, value=18) #x pixel coord
y_pixel = st.number_input("Y Pixel", min_value=0, max_value=y_dim - 1, value=15) #y pixel coord

plot.spaxel_imgs(collapsed, data_cube, x_pixel, y_pixel, wavelengths)

st.markdown(r"""$\textbf{Figure 4. (Left)}$ This shows a 2D spatial map of the A2597 BCG summed over the spectral axis. 
            The black box highlights the user's inputted x,y coordinate. The colorbar shows the intensity
            at each pixel. $\textbf{(Right)}$ Remember, the lines shown are H$_2$(S2), [NeII], and [NeIII] from left to right,
            respectively. This displays the spectrum at the inputted pixel, called a spaxel. At the
            default (18, 15) pixel, there is clear broadening effects in the [NeII] and [NeIII] lines. This is most likely 
            due to winds by the AGN. However, we see a lack of broadening in the H$_2$(S2) line. This implies 
            there is some other ionizing source than what is ionizing [NeII] and [NeIII]. 
            """)

#########################
## IMAGE 4: Linewidths ##
#########################

st.header("Calculating Linewidths")
st.markdown("""This next section of code will allow the user to create moment maps that show the dynamics
            of the gas in the BCG. To start, for areas of low signal-to-noise, Vulcan applies the Vorobin method 
            to bin pixels together to achieve some user-inputted threshold. This can be inputted below. 
            """)

#WCS information is needed to convert MJy/sr of each pixel to Jy
pix_scale_ra = data_cube.sum(axis=0).wcs.wcs.cdelt[0]
pix_scale_dec = data_cube.sum(axis=0).wcs.wcs.cdelt[1]
deg_to_rad = np.pi / 180
pixel_area_sr = (np.abs(pix_scale_ra) * deg_to_rad) * (np.abs(pix_scale_dec) * deg_to_rad) #area of pixel in sr

#we're going to base our signal map on the Ne2 line
#0 is H2(S2), 1 is [Neii], 2 is [Neiii] (depends on peak wavelengths)
i = 1
subcube = data_cube.spectral_slab((peak_wavelengths[i]-0.01)*u.um, (peak_wavelengths[i]+0.01)*u.um)
subcube_err = data_cube_err.spectral_slab((peak_wavelengths[i]-0.01)*u.um, (peak_wavelengths[i]+0.01)*u.um)

#computes the signal and noise maps from spectral slabs around an emission line
signal_map, noise_map = util.compute_sn_map(subcube, subcube_err)

snr = st.number_input("SNR threshold:", value=3) #ask the user to input a desired SNR threshold
bin_num, x_good, y_good, sn = util.run_voronoi_(signal_map, noise_map, snr, plot = False, quiet = True)
bin_map = util.make_bin_map(bin_num, x_good, y_good, signal_map.shape)

#check if SNR changed or bin_fluxes not yet in session state
if (
    'bin_fluxes' not in st.session_state
    or 'snr_used' not in st.session_state
    or st.session_state.snr_used != snr
):
    #rerun binning and extract arrays
    bin_num, x_good, y_good, sn = util.run_voronoi_(signal_map, noise_map, snr, plot=False, quiet=True)
    bin_map = util.make_bin_map(bin_num, x_good, y_good, signal_map.shape)
    bin_fluxes, bin_errors, bin_masks = util.extract_bin_arrays(data_cube, data_cube_err, bin_map, pixel_area_sr)

    #cache the results (this function takes many seconds to run)
    st.session_state.snr_used = snr
    st.session_state.bin_fluxes = bin_fluxes
    st.session_state.bin_errors = bin_errors
    st.session_state.bin_masks = bin_masks
    st.session_state.bin_map = bin_map
    st.session_state.bin_num = bin_num

else:
    #if we already ran it and nothing changed, load from session state
    bin_fluxes = st.session_state.bin_fluxes
    bin_errors = st.session_state.bin_errors
    bin_masks = st.session_state.bin_masks
    bin_map = st.session_state.bin_map
    bin_num = st.session_state.bin_num

st.markdown("""Vulcan then fits a simple Gaussian model to the three peak wavelengths at each pixel or 
            region, and returns the linewidth based on the standard deviation of this model. If you would like
            to view the Gaussian fits for a given pixel, check the box below and enter the pixel coords.
            """)

#the following shows Gaussian fits of the three emission lines based on an user-inputted pixel coord
if st.checkbox("Show Gaussian fits?"):
    x_pixel = st.number_input("X Pixel", min_value=0, max_value=x_dim - 1, value=18, key = 'x2')
    y_pixel = st.number_input("Y Pixel", min_value=0, max_value=y_dim - 1, value=15, key = 'y2')
    i = bin_map[y_pixel][x_pixel] #index to grab the spectrum from
    _, _, _, _, _, _ = util.extracted_vals_from_gaussian(peak_wavelengths, 0.1, wavelengths, bin_fluxes[i], bin_errors[i], plot=True)
    st.write("calculate chi^2 for this fit? also, need to have the name of the line show up for this plot. also explain how if one of the lines doesn't show up its because the fit failed for that emission line in that pixel")

#define session key
gauss_key = f"gauss_fit_results_snr_{st.session_state.snr_used}"

#check if we've already computed results for this SNR
if gauss_key not in st.session_state:

    #ChatGPT-4o was used on Oct 11, 2025 to generate the code for the ordinal encoder (because I have never used encoders before)
    #define encoder and fit it on the quality categories
    quality_levels = ['failed', 'poor', 'good', 'excellent']
    encoder = OrdinalEncoder(categories=[quality_levels]) #used to convert quality levels to numbers
    encoder.fit([[q] for q in quality_levels])

    #initialize lists
    quality_labels = []
    quality_encoded = []
    good_quality_labels = [] #only the good and excellent quality fits
    good_quality_encoded = []

    lw = []
    lw_err = []
    mean_fits = []
    mean_fits_errs = []
    amp = []
    amp_err = []

    all_bin_masks = []
    valid_bin_masks = [] #for the pixels that don't have poor or failed fits
    valid_bin_fluxes = []
    valid_bin_errs = []

    for i in range(len(bin_fluxes)):
        all_bin_masks.append(bin_masks[i])

        #fit gaussian model to each pixel, return linewidth and other values from this fit
        lw_, lw_err_, mean_fit, mean_fit_errs, amp_, amp_err_ = util.extracted_vals_from_gaussian(peak_wavelengths, 0.1, wavelengths, bin_fluxes[i], bin_errors[i])
        
        #if any of the fits returned NaN vals, this is obviously a failed fit
        if np.any(np.isnan(lw_err_)) or np.any(np.isnan(lw_)) or np.any(np.isnan(mean_fit)) or np.any(np.isnan(mean_fit_errs)) or np.any(np.isnan(amp_)) or np.any(np.isnan(amp_err_)):
            quality_labels.append('failed')
            quality_encoded.append(encoder.transform([['failed']])[0][0]) #encode the failed keyword as a number
            continue
        
        #if any of the linewidths are returned as negative values, this is also an obvious fail
        if np.any(np.array(lw_) < 0.0):
            quality_labels.append('failed')
            quality_encoded.append(encoder.transform([['failed']])[0][0])
            continue
        
        #if the linewidth error is greater than the linewidth, this is a poor fit.
        if np.any(np.array(lw_err_) > lw_):
            quality_labels.append('poor')
            quality_encoded.append(encoder.transform([['poor']])[0][0])
            continue
        
        #if the amplitude error is greater than a third of the amplitude, this is a poor fit.
        if np.any(np.array(amp_err_) > (1 / 3) * np.array(amp_)):
            quality_labels.append('poor')
            quality_encoded.append(encoder.transform([['poor']])[0][0])
            continue

        #calculate the resolution of our lines
        h2s2_res = (mean_fit[0] / (2 * np.sqrt(2 * np.log(2)) * lw_[0])) 
        ne2_res =  (mean_fit[1] / (2 * np.sqrt(2 * np.log(2)) * lw_[1]))
        ne3_res =  (mean_fit[2] / (2 * np.sqrt(2 * np.log(2)) * lw_[2]))
        
        #if any of our measured resolutions are smaller than the telescope, these are failed fits. 
        #it is not possible to get a better resolution than the actual resolution of the telescope
        if any([
            h2s2_res < (0.1 * 2530), #these values for telescope resolution we're grabbed from the MIRI MRS website
            ne2_res < (0.1 * 2530), #found at this link: https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-observing-modes/miri-medium-resolution-spectroscopy#gsc.tab=0
            ne3_res < (0.1 * 1980),
        ]):
            quality_labels.append('failed')
            quality_encoded.append(encoder.transform([['failed']])[0][0])
            continue

        #if these values passed all checks we now decide between 'good' and 'excellent' fits
        #excellent fits will be defined as when the errors are less than 10% of the actual value
        #otherwise, we'll call the fits good
        if all([
            np.all(np.array(lw_err_) < 0.1 * np.array(lw_)),
            np.all(np.array(amp_err_) < 0.1 * np.array(amp_)),
        ]):
            good_quality_labels.append('excellent')
            good_quality_encoded.append(encoder.transform([['excellent']])[0][0])
            quality_labels.append('excellent')
            quality_encoded.append(encoder.transform([['excellent']])[0][0])
        else:
            quality_labels.append('good')
            quality_encoded.append(encoder.transform([['good']])[0][0])
            good_quality_labels.append('good')
            good_quality_encoded.append(encoder.transform([['good']])[0][0])

        #append all our results
        lw.append(lw_)
        lw_err.append(lw_err_)
        mean_fits.append(mean_fit)
        mean_fits_errs.append(mean_fit_errs)
        amp.append(amp_)
        amp_err.append(amp_err_)
        valid_bin_masks.append(bin_masks[i])
        valid_bin_fluxes.append(bin_fluxes[i])
        valid_bin_errs.append(bin_errors[i])

    #cache all results in session state because this function takes even longer to run
    st.session_state[gauss_key] = {
        "lw": lw,
        "lw_err": lw_err,
        "mean_fits": mean_fits,
        "mean_fits_errs": mean_fits_errs,
        "amp": amp,
        "amp_err": amp_err,
        "valid_bin_masks": valid_bin_masks,
        "valid_bin_fluxes": valid_bin_fluxes,
        "valid_bin_errs": valid_bin_errs,
        "quality_labels": quality_labels,
        "quality_encoded": quality_encoded,
        "all_bin_masks": all_bin_masks,
        "good_quality_labels": good_quality_labels,
        "good_quality_encoded": good_quality_encoded
    }

else:
    #load from session state
    fit_results = st.session_state[gauss_key]
    lw = fit_results["lw"]
    lw_err = fit_results["lw_err"]
    mean_fits = fit_results["mean_fits"]
    mean_fits_errs = fit_results["mean_fits_errs"]
    amp = fit_results["amp"]
    amp_err = fit_results["amp_err"]
    valid_bin_masks = fit_results["valid_bin_masks"]
    valid_bin_fluxes = fit_results["valid_bin_fluxes"]
    valid_bin_errs = fit_results["valid_bin_errs"]
    quality_labels = fit_results["quality_labels"]
    quality_encoded = fit_results["quality_encoded"]
    all_bin_masks = fit_results["all_bin_masks"]
    good_quality_labels = fit_results["good_quality_labels"]
    good_quality_encoded = fit_results["good_quality_encoded"]

#we will begin our plotting of these linewidths now
#compute average linewidths only from good and excellent regions (encoded as 2 or 3)
lw_array = np.array(lw)  #shape: (n_valid_regions, n_lines)
lw_err_array = np.array(lw_err)

quality_array = np.array(good_quality_encoded)
valid_mask = (quality_array >= 2) #select only good (2) and excellent (3) fits
mean_lws = np.nanmean(lw_array[valid_mask], axis=0) #one average per emission line
mean_lw_err = np.nanmean(lw_err_array[valid_mask], axis=0)

#create our 3 separate linewidth maps
image_shape = valid_bin_masks[0].shape
linewidth_maps = [np.full(image_shape, np.nan) for _ in range(lw_array.shape[1])] #just fill in values for now to get the right shape
lw_err_maps = [np.full(image_shape, np.nan) for _ in range(lw_err_array.shape[1])]

#fill in linewidth maps with actual lw values
for i, mask in enumerate(valid_bin_masks):
    for line_idx in range(lw_array.shape[1]):
        linewidth_maps[line_idx][mask] = lw_array[i][line_idx]
        lw_err_maps[line_idx][mask] = lw_err_array[i][line_idx]

peak_labels = [f"{round(w, 3)} μm" for w in peak_wavelengths]

titles2 = [
    "H2(S2) Linewidths",
    "[NeII] Linewidths",
    "[NeIII] Linewidths"
]

#now plot them!
plot.linewidth_plot(titles2, linewidth_maps, lw_err_maps)

st.markdown(r"""$\textbf{Figure 5.}$ This is the map of linewidths for h2s2, [NeII], and neiii. regions of high
            linewidth indicate a broadening effect. we would expect to see this near the center of the BCG due to
            winds by the AGN, as well as other areas with fast-moving gas.
            """)

##########################
## IMAGE 5: Quality Map ##
##########################

#starting imputation section
st.subheader("Imputation")

st.markdown("""Besides the JWST data quality array, we can also create a fit quality check. To view the fit quality
            map, check the box below. We can also attempt
to impute these poor quality fits with our own linewidth choice. Vulcan does this initially in two different ways:
by imputing the mean of the linewidths and by imputing the mean of the spatially k nearest neighbors.
""")

#create an empty 2D image for quality map
image_shape = valid_bin_masks[0].shape
quality_map = np.full(image_shape, np.nan)

#fill in quality map with encoded values
for i, mask in enumerate(all_bin_masks):
    quality_map[mask] = quality_encoded[i]

#checkbox to show/hide quality map
if st.checkbox("Show Fit Quality Map?"):
    plot.quality_map_img(quality_map)

################################
## IMAGE 6: Impute Linewidths ##
################################

#round the peak wavelengths for display
peak_labels = [f"{round(w, 3)} μm" for w in peak_wavelengths]

#let user choose one of the three peak wavelengths
selected_labels = st.radio("Choose an emission line to view imputed linewidths:", peak_labels, key = 'second one')
selected_index = peak_labels.index(selected_labels) #get the index from the label

#impute mean where quality is poor (1)
for line_idx in range(lw_array.shape[1]):
    poor_mask = (quality_map == 1)  # Where the quality is 'poor'
    nan_in_poor = np.isnan(linewidth_maps[line_idx]) & poor_mask
    linewidth_maps[line_idx][nan_in_poor] = mean_lws[line_idx]
    lw_err_maps[line_idx][nan_in_poor] = mean_lw_err[line_idx]

#now we need to impute values in new images based on some number of neighbors
#impute only poor quality (1) pixels using k nearest neighbors
k = st.number_input("k number of neighbors:", value=3, key = 'knn?')

#create our 3 separate linewidth maps
image_shape = valid_bin_masks[0].shape
linewidth_maps_second = [np.full(image_shape, np.nan) for _ in range(lw_array.shape[1])]
lw_err_maps_second = [np.full(image_shape, np.nan) for _ in range(lw_err_array.shape[1])]

#fill in linewidth maps with our actual values
for i, mask in enumerate(valid_bin_masks):
    for line_idx in range(lw_array.shape[1]):
        linewidth_maps_second[line_idx][mask] = lw_array[i][line_idx]
        lw_err_maps_second[line_idx][mask] = lw_err_array[i][line_idx]

#this function imputes based on the average linewidth of the k nearest neighbors and plots the results of both imputations
plot.imputation_imgs(linewidth_maps, lw_err_maps, selected_index, linewidth_maps_second, lw_err_maps_second, quality_map, k)

st.markdown("""Obviously, imputing the mean has no significant value. This tells us nothing about the structure of the 
            gas at those pixels. Our best attempt at recovering some dynamical information is by using the spatial knn 
            method, but even then, we can't impute values if there's nothing else nearby. A future update of Vulcan might
            include different imputation methods to choose from.
""")

###############################
## FINAL PROJECT STUFF BELOW ##
###############################

st.header("Coming Soon in Theaters Near You")
st.markdown("""
- Moment maps of redshift, velocity, and velocity dispersion
- Line ratio maps
- Support for additional datasets            

""")
