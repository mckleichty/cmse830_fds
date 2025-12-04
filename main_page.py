# main_page.py
import streamlit as st
import numpy as np
import os
import astropy.units as u
from scipy.signal import find_peaks
from sklearn.preprocessing import OrdinalEncoder
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

#importing files I wrote
import util as util #useful functions that would otherwise clutter this file up
from sidebar import sidebar_inputs #siderbar values
import plot as plot #visualizations
import parallel_fitter as parallel

#start page setup
st.set_page_config(layout="wide")

#banner image
st.image("https://github.com/mckleichty/cmse830_fds/blob/main/images/banner.jpg?raw=true", use_container_width=True)
st.set_page_config(page_title="Vulcan", layout="wide")

#get sidebar values
uploaded_file, z_input = sidebar_inputs()

#define tabs
tab1, tab2, tab3, tab4 = st.tabs(["Getting Started", "IDA", "EDA and Imputation", "Feature Engineering/ML"])

# --- TAB 1: general information ---
with tab1:
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
    #information about this dataset
    st.header("Exploring Abell 2597")
    
    #information on A2597
    col1, col2 = st.columns([2.5, 1])
    with col2:
        st.image("https://github.com/mckleichty/cmse830_fds/blob/main/images/a2597.jpg?raw=true", use_container_width=True)
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
    Space Telescopes (MAST) and is also available for download from [this Google Drive link](https://drive.google.com/file/d/1UHRmaXy2bDdfFKwCTo-s7IxmROA2eAxV/view?usp=drive_link).
    """)

# --- TAB 2: my IDA ---
with tab2:
    st.subheader("IFU Information")
    st.markdown("Below includes some general information about the type and shape of our IFU dataset.")
    
    #load primary FITS file
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_to_use = BytesIO(file_bytes)  #wrap in BytesIO for first use (because fits was having errors opening this file twice)
        file_to_use_for_convert = BytesIO(file_bytes)  #new BytesIO for second use
        st.success("Using uploaded FITS file.")
    else:
        st.error("No file uploaded.")
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

    #saving to session state
    st.session_state.peak_wavelengths = peak_wavelengths
    peak_wavelengths = st.session_state.peak_wavelengths
    
    ####################################################
    ## IMAGE 1: side by side total image and spectrum ##
    ####################################################
    
    #collapse the cube along the spectral axis
    collapsed = data_cube.sum(axis=0).value #shape (y, x)
    plot.total_imgs(collapsed, wavelengths, flux, peak_wavelengths, peaks)
    #figure caption
    st.markdown(r"""$\textbf{Figure 2. (Left)}$ 2D image of A2597 summed over the spectral axis, in this case, wavelength. The colorbar
                marks the intensity at each pixel in the image.
                $\textbf{(Right)}$ The total spectrum of A2597 from Channel 3 of the MRS IFU. The three emission lines that
                are present are H$_2$(S2), [NeII], and [NeIII].
                """)
    
    ####################################
    ## IMAGE 2: local continuum plots ##
    ####################################
    
    st.header("Spatial Structure of Emission Lines")
    st.markdown(r"""This part of Vulcan subtracts the locally-defined continuum out from the emission lines found in the previous plot.
                The local continuum is defined as $\pm 0.1$ um either side of the peak wavelength.
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
                This plot is showing the spectral slabs of the local continuum either side the peak wavelength, defined as
                0.1 microns out from the center.
                """)
    
    ############################################
    ## IMAGE 3: side by side image and spaxel ##
    ############################################
    
    st.header("Viewing Spaxels")
    n_spec, y_dim, x_dim = data_cube.shape
    x_pixel = st.number_input("X Pixel", min_value=0, max_value=x_dim - 1, value=18) #x pixel coord
    y_pixel = st.number_input("Y Pixel", min_value=0, max_value=y_dim - 1, value=15) #y pixel coord
    
    plot.spaxel_imgs(collapsed, data_cube, x_pixel, y_pixel, wavelengths)
    
    st.markdown(r"""$\textbf{Figure 4. (Left)}$ A 2D spatial map of the A2597 BCG summed over the spectral axis. 
                The black box highlights the user's inputted (x,y) coordinate. The colorbar shows the intensity
                at each pixel. $\textbf{(Right)}$ The lines shown are H$_2$(S2), [NeII], and [NeIII] from left to right,
                respectively. This displays the spectrum at the inputted pixel, called a spaxel. At the
                default (18, 15) pixel, there is clear broadening effects in the [NeII] and [NeIII] lines. This is most likely 
                due to winds by the AGN. However, we see a lack of broadening in the H$_2$(S2) line. This implies 
                there is either some other ionizing source than what is ionizing [NeII] and [NeIII] or a cloud of molecular
                Hydrogen separate from a cloud of Neon gas. 
                """)

# --- TAB 3: my EDA ---
#########################
## IMAGE 4: Linewidths ##
#########################
with tab3:
    st.header("Calculating Linewidths")
    st.markdown("""**This next section of code might take a couple of minutes to run!** This is where Vulcan begins creating moment
                maps that show the dynamics
                of the gas in the BCG. To start, for areas of low signal-to-noise, Vulcan applies the Vorobin method 
                to bin pixels together to achieve some user-inputted S/N threshold. This can be inputted below. 
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
    
    st.markdown("""Vulcan then fits a simple Gaussian model to the peak wavelengths at each pixel or 
                region, and returns the linewidth based on the standard deviation of this model. If you would like
                to view the Gaussian fits for a given pixel, check the box below and enter the pixel coordinates.
                """)
    
    #the following shows Gaussian fits of the three emission lines based on an user-inputted pixel coord
    if st.checkbox("Show Gaussian fits?"):
        x_pixel = st.number_input("X Pixel", min_value=0, max_value=x_dim - 1, value=18, key = 'x2')
        y_pixel = st.number_input("Y Pixel", min_value=0, max_value=y_dim - 1, value=15, key = 'y2')
        i = bin_map[y_pixel][x_pixel] #index to grab the spectrum from
        _, _, _, _, _, _, _ = util.extracted_vals_from_gaussian(peak_wavelengths, 0.1, wavelengths, bin_fluxes[i], bin_errors[i], plot=True)
    
    #define session key
    gauss_key = f"gauss_fit_results_snr_{st.session_state.snr_used}"
    gaussian_results_key = f"gaussian_results_{st.session_state.snr_used}"
    
    #check if we've already computed results for this SNR
    if gauss_key not in st.session_state:
    # Only compute if either key is missing
    #if gauss_key not in st.session_state or gaussian_results_key not in st.session_state:
        
        #number of worker processes to use
        N_WORKERS = min(4, os.cpu_count())
    
        #package arguments for each pixel
        tasks = [(peak_wavelengths, 0.1, wavelengths, bin_fluxes[i], bin_errors[i]) for i in range(len(bin_fluxes))]
    
        #start parallel execution
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            results = list(executor.map(parallel.run_pixel_fit, tasks))
        
        # cache results under snr-specific key
        #st.session_state[gaussian_results_key] = results

        #unpack all results from parallel workers
        lw, lw_err, mean_fits, mean_fits_errs = [], [], [], []
        amp, amp_err = [], []
        chi2_red = []
    
        #ChatGPT-4o was used on Oct 11, 2025 to generate the code for the ordinal encoder (because I have never used encoders before)
        #define encoder and fit it on the quality categories
        quality_levels = ['failed', 'poor', 'good', 'excellent']
        encoder = OrdinalEncoder(categories=[quality_levels]) #used to convert quality levels to numbers
        encoder.fit([[q] for q in quality_levels])
    
        quality_labels = []
        quality_encoded = []
        good_quality_labels = []
        good_quality_encoded = []
    
        all_bin_masks = []
        valid_bin_masks = []
        valid_bin_fluxes = []
        valid_bin_errs = []
    
        # recreate your classification logic using parallel output
        for i, res in enumerate(results):
            all_bin_masks.append(bin_masks[i])
    
            #fit gaussian model to each pixel, return linewidth and other values from this fit
            lw_, lw_err_, mean_fit, mean_fit_errs, amp_, amp_err_, chi2_red_ = res
            #lw_, lw_err_, mean_fit, mean_fit_errs, amp_, amp_err_ = util.extracted_vals_from_gaussian(peak_wavelengths, 0.1, wavelengths, bin_fluxes[i], bin_errors[i])
            
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
            chi2_red.append(chi2_red_)
    
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
            "good_quality_encoded": good_quality_encoded,
            "chi2_red": chi2_red,
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
        chi2_red = fit_results["chi2_red"]
        
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

    #store it
    st.session_state.lw_maps = linewidth_maps
    #linewdith_maps = st.session_state.lw_maps #call it
    
    peak_labels = [f"{round(w, 3)} μm" for w in peak_wavelengths]
    
    titles2 = [
        "H2(S2) Linewidths",
        "[NeII] Linewidths",
        "[NeIII] Linewidths"
    ]
    
    #now plot them!
    plot.linewidth_plot(titles2, linewidth_maps, lw_err_maps)
    
    st.markdown(r"""$\textbf{Figure 5.}$ This is the map of linewidths for H$_2$(S2), [NeII], and [NeIII]. Regions of high
                linewidth indicate a broadening effect. We would expect to see this near the center of the BCG due to
                winds by the AGN, as well as other areas with fast-moving gas.
                """)
    
    ##########################
    ## IMAGE 5: Quality Map ##
    ##########################
    
    #starting imputation section
    st.subheader("Imputation")
    
    st.markdown("""Besides the JWST data quality array, we can also create our own fit quality check. To view the fit quality
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
    
    st.markdown("""Realistically, imputing the average linewidth has no significant value. This tells us nothing about the structure of the 
                gas at those pixels. Our best attempt at recovering some dynamical information is by using the spatial KNN 
                method, but even then, we can't impute values if there's nothing else nearby. A future update of Vulcan might
                include different imputation methods to choose from.
    """)

###############################
## FINAL PROJECT STUFF BELOW ##
###############################

#st.header("Coming Soon in Theaters Near You")
#st.markdown("""
#- Moment maps of redshift, velocity, and velocity dispersion
#- Line ratio maps
#- New imputation methods
#- Support for additional datasets            
#
#""")

def reduced_chi_squared(flux, flux_err, fit_vals, num_params):
    """
    Compute reduced chi-squared.
    """
    residuals = flux - fit_vals
    chi2 = np.sum((residuals / flux_err)**2)
    dof = len(flux) - num_params
    return chi2 / dof


# --- TAB 4: Machine Learning second-component analysis ---
with tab4:
    st.header("ML-Based Second Component Prediction")
    st.markdown("""During the EDA section of this analysis, we saw that for some pixels, there were high $\\tilde{\chi}^2$ values.
    This means that a simple Gaussian model won't work. Instead, we can add a second Gaussian to model the second component of the gas.
    There can be different gas clouds moving in different directions with the same element in them. Vulcan uses a linear regression
    machine learning model to predict which pixels will need a second component fit based on the previous $\\tilde{\chi}^2$ values 
    and the linewdiths. You can select below which emission line to focus on.
    """)

    # Load cached processed Gaussian results from Tab 3
    gauss_key = f"gauss_fit_results_snr_{st.session_state.snr_used}"
    fit_results = st.session_state[gauss_key]

    lw = fit_results["lw"]
    lw_err = fit_results["lw_err"]
    mean_fits = fit_results["mean_fits"]
    mean_fits_errs = fit_results["mean_fits_errs"]
    amp = fit_results["amp"]
    amp_err = fit_results["amp_err"]
    valid_bin_fluxes = fit_results["valid_bin_fluxes"]
    valid_bin_masks = fit_results["valid_bin_masks"]
    chi2_red = fit_results["chi2_red"]
    
    peak_wavelengths = st.session_state.peak_wavelengths
    bin_fluxes = st.session_state.bin_fluxes
    bin_errors = st.session_state.bin_errors
    bin_map = st.session_state.bin_map
    linewdith_maps = st.session_state.lw_maps

    n_lines = len(chi2_red[0])  # e.g., 3 emission lines
    chi2_maps = [np.full(image_shape, np.nan) for _ in range(n_lines)]
    
    for i, mask in enumerate(valid_bin_masks):
        for line_idx in range(n_lines):
            chi2_maps[line_idx][mask] = chi2_red[i][line_idx]


    j = 1 #which emission line to look at
    chi2_threshold = 3.0
    # chi2_map is 2D array of reduced chi^2 for each pixel
    second_component_label = (chi2_maps[j] > chi2_threshold).astype(int)

    # Flatten and stack features: shape = (num_pixels, 2)
    X_raw = np.stack([
        chi2_maps[j].flatten(),   # reduced chi²
        linewdith_maps[j].flatten()      # line width
    ], axis=1)
    chi2_threshold = 3.0
    y_raw = (chi2_maps[j].flatten() > chi2_threshold).astype(int)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # get rid of nans
    mask_valid = ~np.isnan(X_raw).any(axis=1)
    X = X_raw[mask_valid]
    y = y_raw[mask_valid]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    # Optional evaluation
    accuracy = clf.score(X_test, y_test)
    st.write(f"ML model accuracy: {accuracy:.2f}")

    # Initialize map with NaNs
    second_component_pred = np.full(chi2_maps[j].shape, np.nan)
    
    # Flatten the map to 1D
    flat_pred = np.full(X_raw.shape[0], np.nan)
    
    # Assign predictions to valid indices
    flat_pred[mask_valid] = clf.predict(X)
    
    # Reshape back to 2D
    second_component_pred = flat_pred.reshape(chi2_maps[j].shape)
    
    figs = px.imshow(
        second_component_pred,
        #color_continuous_scale='RdBu_r',
        origin='lower',
        labels={'color': 'Second Component'},
        title="ML-predicted Pixels Needing Second Component"
    )
    st.plotly_chart(figs, use_container_width=True)

    #figss = px.imshow(second_component_label, 
    #            color_continuous_scale='RdBu_r', 
    #            origin='lower',
    #            labels={'color':'Second Component'},
    #            title="Pixels needing second component (1 = yes, 0 = no)")
    #st.plotly_chart(figss, use_container_width=True)

    st.markdown("""Based on which pixels are predicted to need a second component fit, Vulcan fits two Gaussian models. We expect to
    see a $\\tilde{\chi}^2$ value closer to 1 if it's fit with a second Gaussian. The fits for a given pixel are shown below.
    """)
    
    x_pixel = st.number_input("X Pixel", min_value=0, max_value=x_dim - 1, value=18, key = 'x3')
    y_pixel = st.number_input("Y Pixel", min_value=0, max_value=y_dim - 1, value=15, key = 'y3')
    i = bin_map[y_pixel][x_pixel] #index to grab the spectrum from
    peak_wavelength = peak_wavelengths[j]
    titles2 = ["H2(S2)", "[NeII]", "[NeIII]"]
    #title = f"{titles2[j]}"
    
    col1, col2 = st.columns([1, 1])
    with col1:
        #_, _, _, _, _, _, _ = util.extracted_vals_from_gaussian(peak_wavelengths, 0.1, wavelengths, bin_fluxes[i], bin_errors[i], plot=True)
        _, _, _, _, _ = util.gaussian_fitter(peak_wavelength, 0.1, wavelengths, bin_fluxes[i], bin_errors[i], titles2[j], truncate_side = None, truncate_percent = 0.0, plot = True)
    with col2:
        _, _, _, _, _ = util.gaussian_fitter_new(peak_wavelength, 0.1, wavelengths, bin_fluxes[i], bin_errors[i], 
                        titles2[j], truncate_side=None, truncate_percent=0.0, plot=True, second_comp_map = bool(second_component_label[y_pixel, x_pixel]))

    
    










