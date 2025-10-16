# VULCAN: Visualization Utility for Luminous Cluster Analysis in NIR/MIR
## *A Streamlit web application to explore IFU spectra in the Infrared*

## Project Overview

VULCAN is a web-based interactive visualization tool designed to support astrophysical research on galaxy clusters using spectral cube data from the James Webb Space Telescope's Medium Resolution Spectrometer (MRS). It focuses on exploring spaxels, gas dynamics, and emission line ratios using Abell 2597 (A2597) as a proxy target.

The A2597 dataset was chosen because it is a well studied Brightest Cluster Galaxy (BCG), and currently has public JWST data out on it. This was used to design and write the code for Vulcan in preparation for future JWST datasets. As of now, Vulcan is not ready to handle other uploaded datasets.

The general features of the app are listed below:
- Interactive web interface built with Streamlit
- Visualize 2D spatial maps and spectra
- View individual pixel (spaxel) spectra
- Subtract local continuum around emission lines
- Fit Gaussian models to spectral lines
- Generate linewidth maps and estimate gas dynamics
- Impute missing/poor-quality fits with SNR-based Voronoi binning and Spatial KNN

Along with future features: 
- Moment maps of velocity and redshift
- Line ratio analysis
- Support for additional datasets

## Getting Started

To use Vulcan, either clone the repository and install the requirements from the `requirements.txt` file, or follow this [link](https://cmse830fds-rxte2xpg3kggapp2vmnqgkk.streamlit.app/)!

If you would like to explore Vulcan's capabilities without your own dataset to upload, feel free to download the A2597 dataset either from the Mikulski Archive for Space Telescopes (MAST) or [this Google Drive link](https://drive.google.com/file/d/1UHRmaXy2bDdfFKwCTo-s7IxmROA2eAxV/view?usp=drive_link), and then upload it into Vulcan!

## App Features

Vulcan includes several key features to support users in analyzing their IFU data. Some of the initial preprocessing steps are:
- Rest-frame wavelength conversion: This step shifts the observed spectra to the rest frame, making it easier to identify emission and absorption lines. While optional, it's a standard step in most astronomical analyses.
- Background subtraction: Many JWST IFU cubes come pre-subtracted, but not all. Vulcan will eventually support user-provided background FITS files for cases where background removal is needed.
- Data quality masking: includes masking of bad pixels based on quality flags in the data cube.
- Dereddening (not currently planned): Although not critical for general exploratory analysis, future updates may include dereddening to correct for dust extinction.

After preprocessing, Vulcan provides interactive 2D spatial maps and spectra visualization. Users can explore spectra at specific pixel locations by inputting pixel coordinates and Vulcan will display the corresponding spectrum alongside the main plot, as shown in the following example.

![spaxel_plot](images/spaxel_plot.jpg)

Another core feature is the visualization of emission lines, including:
- The total continuum of a selected emission line,
- Its locally-defined continuum, and
- The continuum-subtracted map.

This allows users to examine the spatial distribution of ionized gas at specific wavelengths.

For more advanced analysis, Vulcan fits Gaussian models to each emission line at every pixel. If a pixel lacks sufficient signal-to-noise (below a user-defined threshold, default is S/N = 3), Vulcan applies Voronoi binning to group low-S/N pixels into larger regions. This is especially helpful in low-flux areas, such as the outer edges of the field of view or regions far from the BCG center.

Once Gaussian fits are completed, users can inspect individual model fits visually, or rely on automated fit quality flags, which are assigned based on the following rules:

- If any NaN values were returned in the fitting process, this counted as a **failed** fit.
- If any linewidths were negative, this was a **failed** fit.
- If the linewidth error was greater than the actual linewidth, this was a **poor** fit.
- If the amplitude error was more than a third of the actual amplitude, this was a **poor** fit.
- If any linewidths were significantly less than the resolution of the telescope, these were considered **failed** fits.
- If the linewidth and amplitude errors were less than 10% of the actual values, these were considered **excellent** fits.
- Otherwise, the fits were considered **good**.

The fit quality levels are encoded to generate a fit quality map, which highlights which regions of the field have reliable model fits. In general, we would expect pixels closer to the center of the BCG to typically have higher signal-to-noise ratios, resulting in more accurate and higher-quality fits.

Using the Gaussian model parameters, Vulcan also derives the linewidths for each emission line. These linewidths can be used to estimate the line-of-sight velocity and velocity dispersion of the ionized gas within the galaxy. Support for these kinematic analyses is planned as a future feature of Vulcan.

### Imputation
For poor-quality or failed fits, Vulcan offers the option to impute values for affected pixels. However, reconstructing gas dynamics in regions with large gaps is highly unreliable. Simply "filling in" missing data can introduce significant bias, effectively shaping the data to match the user's expectations rather than the true underlying structure.

To mitigate this, Vulcan uses a spatial k-nearest neighbors (KNN) approach. If a missing pixel is surrounded by at least k valid neighbors, Vulcan calculates the average value of those neighbors and imputes it into the pixel. This method can recover some limited information, particularly near the boundaries of high signal-to-noise regions, but its reliability diminishes with distance from those areas.

Ultimately, imputation is offered as a last resort, and users are strongly encouraged to interpret imputed regions with skepticism.

## Streamlit features

Vulcan uses Streamlit’s `session_state` feature to persist uploaded files, user inputs, and intermediate results during reruns caused by widget updates. By storing key data in the session state, Vulcan avoids unnecessary recomputation and ensures that users don’t lose their progress when adjusting parameters or exploring different features.
