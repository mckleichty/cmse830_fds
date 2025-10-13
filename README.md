# VULCAN: Visualization Utility for Luminous Cluster Analysis in NIR/MIR
## *A Streamlit web application to explore IFU spectra in the Infrared*

## Project Overview

VULCAN is a web-based interactive visualization tool designed to support astrophysical research on galaxy clusters using spectral cube data from the James Webb Space Telescope's MIRI MRS instrument. It focuses on exploring gas dynamics, emission lines, and AGN feedback in Abell 2597 (A2597) — a well-known cool-core galaxy cluster.

This project's goal is to allow any user to upload and explore basic analyses' of IFU spectra of a Brightest Cluster Galaxy (BCG). Vulcan is designed to support astrophysical research by making complex datasets intuitive to explore, helping users uncover how gas moves and interacts with active galactic nucleus (AGN) feedback. Our current case study is A2597. It starts by showing a dataset from JWST's MIRI MRS of Abell 2597 and the features that can be applied / how they can be applied to the user's own dataset. As of now, Vulcan is not ready to handle uploaded datasets. 

The A2597 dataset was chosen because it is a well studied cluster galaxy, and currently has public JWST data out on it. This was used to design and write the code for Vulcan in preparation for future JWST datasets. 

Features include:
- Interactive web interface built with Streamlit
- Visualize 2D spatial maps and spectra
- View individual pixel (spaxel) spectra
- Subtract local continuum around emission lines
- Fit Gaussian models to spectral lines
- Generate linewidth maps and estimate gas dynamics
- Impute missing/poor-quality fits with SNR-based Voronoi binning and Spatial KNN

Future features include: 
- Moment maps of velocity and redshift
- Line ratio analysis
- Support for additional datasets

## Getting Started

Clone this repository...
requirements for environment should be listed


## Streamlit features

includes session state, caching, etc.

![Banner](images/spaxel_plot.jpg)



