"""plot.py

This file contains functions needed for plotting in Vulcan.

Created by: McKenna Leichty
Last Updated: Oct 13, 2025

"""
#import necessary modules
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import util as util #my own file

####################################################
## IMAGE 1: side by side total image and spectrum ##
####################################################

def total_imgs(collapsed, wavelengths, flux, peak_wavelengths, peaks):
    """
    This function plots the 2D image of the BCG summed over the spectral axis and the total spectrum
    from this cube file.
    """
    #create two columns in streamlit
    col1, col2 = st.columns([1.75, 2])

    with col1:
        st.subheader("Total 2D Image")
        fig_img = go.Figure(data=go.Heatmap(
            z=collapsed,
            x=np.arange(collapsed.shape[1]), #pixel coordinates
            y=np.arange(collapsed.shape[0]),
            colorscale='Thermal',
            colorbar=dict(title=dict(text="Intensity (MJy/sr)", font=dict(color='black')), tickfont=dict(color='black')),
            hovertemplate="X: %{x}<br>Y: %{y}<br>Intensity: %{z:.2e} MJy/sr<extra></extra>"
        ))
        
        fig_img.update_layout(
            xaxis=dict(title=dict(text='X Pixel', font=dict(color='black')), tickfont=dict(color='black')),
            yaxis=dict(title=dict(text='Y Pixel', font=dict(color='black')), tickfont=dict(color='black')),
            hoverlabel=dict(font=dict(color='black')),
            height=450,
            width=400
        )

        st.plotly_chart(fig_img, width='stretch')

    with col2:
        st.subheader("Total 1D Spectrum")

        #y-axis scale toggle (if the user wants linear or log scale, although most people use linear in mid-infrared)
        scale_type = st.radio("Y-axis Scale", ["Linear", "Log"], horizontal=True, key="plotly_spectrum_scale")
        log_scale = scale_type == "Log"

        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(
            x=wavelengths,
            y=flux, #it says flux but its really intensity
            mode='lines',
            name='Spectrum',
            line=dict(color='#B7410E'),
            hovertemplate="λ: %{x:.2f} μm<br>Intensity: %{y:.2e} MJy/sr<extra></extra>"
        ))

        #add black x markers at peak wavelengths on the spectrum (H2(S2), [NeII], and [NeIII])
        fig_spec.add_trace(go.Scatter(
            x=peak_wavelengths,
            y=flux[peaks], #the intensity at those peaks
            mode='markers',
            marker=dict(color='black', symbol='x', size=10),
            name='Peaks',
            hovertemplate="Peak λ: %{x:.2f} μm<br>Intensity: %{y:.2e} MJy/sr<extra></extra>"
        ))

        #names of the emission lines at each peak wavelength
        peak_labels = ['H2(S2)', '[NeII]', '[NeIII]']

        offset = 10000 #to move our label higher
        annotations = []
        for x, y, label in zip(peak_wavelengths, flux[peaks], peak_labels):
            annotations.append(
                dict(
                    x=x,
                    y=y + offset,
                    xref='x',
                    yref='y',
                    text=label,
                    showarrow=False,
                    font=dict(color='black', size=12),
                )
            )
        fig_spec.update_layout(annotations=annotations)

        fig_spec.update_layout(
            yaxis_type="log" if log_scale else "linear",
            hovermode='x',
            hoverlabel=dict(font=dict(color='black')),
            #this creates a vertical line that follows the user's cursor and reports the wavelength and intensity at 
            # that point
            xaxis=dict(
                color='black',
                title=dict(text='Wavelength (μm)', font=dict(color='black')),
                tickfont=dict(color='black'),
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=True,
                spikethickness=1,
                spikedash='dot',
                spikecolor='black'
            ),
            yaxis=dict(
                color='black',
                title=dict(text='Intensity (MJy/sr)', font=dict(color='black')),
                tickfont=dict(color='black'),
                showspikes=False
            ),
            height=450,
        )

        st.plotly_chart(fig_spec, width='stretch')

####################################
## IMAGE 2: local continuum plots ##
####################################

def continuum_imgs(images, titles):
    """
    This function plots the datacube at a given emission line, the locally defined continuum around that line
    and the subtracted local continuum. This is displayed as 3 plots side-by-side. 
    """
    #three columns for side-by-side display
    cols = st.columns(3)

    for i in range(3): #for each column
        with cols[i]:
            #st.subheader(titles[i]) #given a title (the name or wavelength of the line)

            fig = go.Figure()

            fig.add_trace(go.Heatmap(
                z=images[i].value,
                colorscale='Thermal',
                colorbar=dict(title=dict(text="MJy/sr", font=dict(color='black')), tickfont=dict(color='black')),
                showscale=True,
                hovertemplate="X: %{x}<br>Y: %{y}<br>Intensity: %{z:.2f} MJy/sr<extra></extra>"
            ))

            fig.update_layout(
                title=f'{titles[i]}',
                xaxis=dict(title=dict(text='X Pixel', font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text='Y Pixel', font=dict(color='black')), tickfont=dict(color='black')),
                hoverlabel=dict(font=dict(color='black')),
                height=400
            )

            st.plotly_chart(fig, width='stretch')

############################################
## IMAGE 3: side by side image and spaxel ##
############################################

def spaxel_imgs(collapsed, data_cube, x_pixel, y_pixel, wavelengths):
    """
    This function is the same as total_imgs(), but allows for the user to input pixel coordinates
    to show the extracted spectrum in that region of the IFU. 
    """
    col1, col2 = st.columns([1.75, 2])

    with col1:
        st.subheader("2D Image") #same as before
        fig_imgs = go.Figure(data=go.Heatmap(
            z=collapsed,
            x=np.arange(collapsed.shape[1]),
            y=np.arange(collapsed.shape[0]),
            colorscale='Thermal',
            colorbar=dict(title=dict(text="Intensity (MJy/sr)", font=dict(color='black')), tickfont=dict(color='black')),
            hovertemplate="X: %{x}<br>Y: %{y}<br>Intensity: %{z:.2e} MJy/sr<extra></extra>"
        ))

        fig_imgs.update_layout(
            xaxis=dict(title=dict(text='X Pixel', font=dict(color='black')), tickfont=dict(color='black')),
            yaxis=dict(title=dict(text='Y Pixel', font=dict(color='black')), tickfont=dict(color='black')),
            hoverlabel=dict(font=dict(color='black')),
            height=450,
        )

        #add a black rectangle (1-pixel box) to highlight the selected pixel
        fig_imgs.add_shape(
            type="rect",
            x0=x_pixel - 0.5,
            x1=x_pixel + 0.5,
            y0=y_pixel - 0.5,
            y1=y_pixel + 0.5,
            line=dict(color="black", width=2),
            layer="above"
        )
        st.plotly_chart(fig_imgs, width='stretch', key="plotly_img")

    with col2:
        st.subheader("Selected Spaxel")

        scale_type = st.radio("Y-axis Scale", ["Linear", "Log"], horizontal=True, key="plotly_spectrum_scales")
        log_scale = scale_type == "Log"

        #extract the spectrum for the selected pixel
        pixel_spectrum = data_cube[:, y_pixel, x_pixel]

        fig_specs = go.Figure()
        fig_specs.add_trace(go.Scatter(
            x=wavelengths,
            y=pixel_spectrum,
            mode='lines',
            line=dict(color='#B7410E'),
            hovertemplate="λ: %{x:.2f} μm<br>Intensity: %{y:.2e} MJy/sr<extra></extra>"
        ))
        
        fig_specs.update_layout(
            yaxis_type="log" if log_scale else "linear",
            hovermode='x',
            hoverlabel=dict(font=dict(color='black')),
            xaxis=dict(
                color='black',
                title=dict(text='Wavelength (μm)', font=dict(color='black')),
                tickfont=dict(color='black'),
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=True,
                spikethickness=1,
                spikedash='dot',
                spikecolor='black'
            ),
            yaxis=dict(
                color='black',
                title=dict(text='Intensity (MJy/sr)', font=dict(color='black')),
                tickfont=dict(color='black'),
                showspikes=False
            ),
            height=450,
        )

        st.plotly_chart(fig_specs, width='stretch', key="plotly_spectrum")

#########################
## IMAGE 4: Linewidths ##
#########################

def linewidth_plot(titles, linewidth_maps, lw_err_maps):
    """
    This function plots the linewidth maps for each emission line in our total spectrum (3 for A2597).
    It will also return some statistical values for these linewidths, such as median value, median error,
    average value, and average error.
    """
    cols = st.columns([1, 1, 1])
    for i in range(3):
        with cols[i]:
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=linewidth_maps[i],
                colorscale='blackbody',
                colorbar=dict(title=dict(text='μm', font=dict(color='black')), tickfont=dict(color='black')),
                showscale=True,
                hovertemplate="X: %{x}<br>Y: %{y}<br>Linewidth: %{z:.4f} μm<extra></extra>"
            ))

            fig.update_layout(
                title=f'{titles[i]}',
                xaxis=dict(title=dict(text='X Pixel', font=dict(color='black')), tickfont=dict(color='black')),
                yaxis=dict(title=dict(text='Y Pixel', font=dict(color='black')), tickfont=dict(color='black')),
                hoverlabel=dict(font=dict(color='black')),
                height=375,
            )

            st.plotly_chart(fig, use_container_width=True)
            
            #print statistical values off for each peak wavelength
            linewidth_data = np.array(linewidth_maps[i]).flatten()
            linewidth_err_data = np.array(lw_err_maps[i]).flatten()
            
            #create pandas df for table viewing
            stats_df = pd.DataFrame({
                "Statistic": ["Median", "Average", "Std", "Minimum", "Maximum"],#, "Median Error", "Average Error"],
                "Value (μm)": [
                    np.nanmedian(linewidth_data),
                    np.nanmean(linewidth_data),
                    np.nanstd(linewidth_data),
                    np.nanmin(linewidth_data),
                    np.nanmax(linewidth_data)
                ],
                "Error (μm)": [
                    np.nanmedian(linewidth_err_data),
                    np.nanmean(linewidth_err_data),
                    np.nanstd(linewidth_err_data),
                    np.nanmin(linewidth_err_data),
                    np.nanmax(linewidth_err_data)
                ],
            })

            #format the values to 5 decimal places
            stats_df["Value (μm)"] = stats_df["Value (μm)"].map(lambda x: f"{x:.5f}")
            stats_df["Error (μm)"] = stats_df["Error (μm)"].map(lambda x: f"{x:.5f}")

            st.dataframe(stats_df, width='stretch', height=150)

##########################
## IMAGE 5: Quality Map ##
##########################

def quality_map_img(quality_map):
    """
    Optional function that will plot the quality map based on the Gaussian model fits. There are four 
    defined quality "levels": failed, poor, good, and excellent. Documentation for how I chose these definitions
    can be found in the app where this plot is made (also, technically in the second column of this plot).
    """
    col1, col2 = st.columns(2)
    with col1:
        #define our colormap labels
        quality_labels = ['failed = 0', 'poor = 1', 'good = 2', 'excellent = 3']
        tickvals = list(range(len(quality_labels)))

        fig = px.imshow(
            quality_map,
            color_continuous_scale=["#0d0887", "#d5385b", "#fcbf3f", "#f0f921"],
            origin='lower',
            zmin=0,
            zmax=3,
            title="Fit Quality Map",
        )

        fig.update_layout(
            coloraxis_colorbar=dict(
                title=dict(text="Fit Quality", font=dict(color='black')),
                tickvals=tickvals,
                ticktext=quality_labels,
                tickfont=dict(color='black')
            ),
            xaxis=dict(title='X Pixel', title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title='Y Pixel', title_font=dict(color='black'), tickfont=dict(color='black')),
            hoverlabel=dict(font=dict(color='black')),
            width=600,
            height=500
        )

        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("""
**Fits were determined in quality by the following rules:**

- If any NaN values were returned in the fitting process, this counted as a **failed** fit.  
- If any linewidths were negative, this was a **failed** fit.  
- If the linewidth error was greater than the actual linewidth, this was a **poor** fit.  
- If the amplitude error was more than a third of the actual amplitude, this was a **poor** fit.  
- If any linewidths were significantly less than the resolution of the telescope, these were considered **failed** fits.  
- If the linewidth and amplitude errors were less than **10%** of the actual values, these were considered **excellent** fits.  
- Otherwise, the fits were considered **good**.
                    
In general, we would expect high sinal-to-noise near the center of the BCG. I don't have this currently plotted, but
the center of A2597 is near pixel (19, 15).
""")
        
################################
## IMAGE 6: Impute Linewidths ##
################################

def imputation_imgs(linewidth_maps, lw_err_maps, selected_index, linewidth_maps_second, lw_err_maps_second, quality_map, k):
    """
    This function plots our linewidth maps based on two different imputation styles. The first just imputes the 
    average linewidth in any poor or failed quality pixels. The second imputes the average of the n surrounding pixels.
    There is no great way to impute values with such low signal to noise far from the center.
    """
    col1, col2 = st.columns([1, 1])
    with col2:
        fig = px.imshow(
            linewidth_maps[selected_index],
            color_continuous_scale="blackbody",
            origin='lower',
            title="Linewidth Map (Imputed with Average Linewidth)",
        )
        
        #this sets our hovermode thing
        fig.data[0].hovertemplate = "X: %{x}<br>Y: %{y}<br>Linewidth: %{z:.4f} μm<extra></extra>"

        fig.update_layout(
            coloraxis_colorbar=dict(title=dict(text="μm", font=dict(color='black')), tickfont=dict(color='black')),
            xaxis=dict(title='X Pixel', title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title='Y Pixel', title_font=dict(color='black'), tickfont=dict(color='black')),
            hoverlabel=dict(font=dict(color='black')),
            height=450
        )

        st.plotly_chart(fig, width='stretch')
        
        #print statistical values off for each peak wavelength
        linewidth_data = np.array(linewidth_maps[selected_index]).flatten()
        linewidth_err_data = np.array(lw_err_maps[selected_index]).flatten()
        
        #create pandas df for table viewing
        stats_df = pd.DataFrame({
            "Statistic": ["Median", "Average", "Std", "Minimum", "Maximum"],#, "Median Error", "Average Error"],
            "Value (μm)": [
                np.nanmedian(linewidth_data),
                np.nanmean(linewidth_data),
                np.nanstd(linewidth_data),
                np.nanmin(linewidth_data),
                np.nanmax(linewidth_data)
            ],
            "Error (μm)": [
                np.nanmedian(linewidth_err_data),
                np.nanmean(linewidth_err_data),
                np.nanstd(linewidth_err_data),
                np.nanmin(linewidth_err_data),
                np.nanmax(linewidth_err_data)
            ],
        })

        #format the values to 5 decimal places
        stats_df["Value (μm)"] = stats_df["Value (μm)"].map(lambda x: f"{x:.5f}")
        stats_df["Error (μm)"] = stats_df["Error (μm)"].map(lambda x: f"{x:.5f}")

        st.dataframe(stats_df, width='stretch', height=150)
        
    with col1:
        for line_idx in range(len(linewidth_maps_second)): #going row by row
            linewidth_maps_second[line_idx] = util.impute_with_neighbors(linewidth_maps_second[line_idx], quality_map, required_quality=1, min_neighbors=k)
            lw_err_maps_second[line_idx] = util.impute_with_neighbors(lw_err_maps_second[line_idx], quality_map, required_quality=1, min_neighbors=k)
        
        fig = px.imshow( 
            linewidth_maps_second[selected_index],
            color_continuous_scale="blackbody",
            origin='lower',
            title=f"Linewidth Map (Imputed with {k} Nearest Neighbors)",
        )
        
        #hovermode
        fig.data[0].hovertemplate = "X: %{x}<br>Y: %{y}<br>Linewidth: %{z:.4f} μm<extra></extra>"
        
        fig.update_layout(
            coloraxis_colorbar=dict(title=dict(text="μm", font=dict(color='black')), tickfont=dict(color='black')),
            xaxis=dict(title='X Pixel', title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title='Y Pixel', title_font=dict(color='black'), tickfont=dict(color='black')),
            hoverlabel=dict(font=dict(color='black')),
            height=450
        )

        st.plotly_chart(fig, width='stretch')

        #print statistical values off for each peak wavelength
        linewidth_data = np.array(linewidth_maps_second[selected_index]).flatten()
        linewidth_err_data = np.array(lw_err_maps_second[selected_index]).flatten()
        
        #create pandas df for table viewing
        stats_df = pd.DataFrame({
            "Statistic": ["Median", "Average", "Std", "Minimum", "Maximum"],#, "Median Error", "Average Error"],
            "Value (μm)": [
                np.nanmedian(linewidth_data),
                np.nanmean(linewidth_data),
                np.nanstd(linewidth_data),
                np.nanmin(linewidth_data),
                np.nanmax(linewidth_data)
            ],
            "Error (μm)": [
                np.nanmedian(linewidth_err_data),
                np.nanmean(linewidth_err_data),
                np.nanstd(linewidth_err_data),
                np.nanmin(linewidth_err_data),
                np.nanmax(linewidth_err_data)
            ],
        })

        #format the values to 5 decimal places
        stats_df["Value (μm)"] = stats_df["Value (μm)"].map(lambda x: f"{x:.5f}")
        stats_df["Error (μm)"] = stats_df["Error (μm)"].map(lambda x: f"{x:.5f}")

        st.dataframe(stats_df, width='stretch', height=150)
