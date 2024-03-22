"""
(c) 2024, Jedd Bellamy-Carter

Test streamlit app that plots a scan from an mzML data file.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Range1d
from bokeh.plotting import figure

from pyteomics import mgf, mzml, pylab_aux, mass, parser

## FUNCTIONS ##

def detect_peaks(spectrum, threshold=5, distance=4, prominence=0.8, width=3, centroid=False):
    """Peak picking from a given spectrum using the relative maxima
    algorithm using a window of size order.

    Only peaks above the specified threshold are returned

    Inputs
    ------

    spectrum : spectrum object from pyteomics

    """
    rel_threshold = spectrum['intensity array'].max() * (threshold / 100)
    if centroid:
        peaks = np.where(spectrum['intensity array'] > rel_threshold)[0]
    else:
        peaks, properties = signal.find_peaks(spectrum['intensity array'], height=rel_threshold, prominence=prominence, width=width, distance=distance)

    return peaks


## APP LAYOUT ##

st.sidebar.title("Quick mzML Data Explorer")
st.sidebar.markdown("This is a simple data explorer for mass spectrometry data stored in `.mzmL` data format")

## Import Raw File

raw_file = st.sidebar.file_uploader("Select a file", type = ['mzml'], key="rawfile", help="Select an mzML file to explore.")
if raw_file is not None:
    reader = mzml.read(raw_file, use_index=True)

    scan_filter_list = {'all': []}
    reader.reset()
    for scan in reader:
        idx = scan['index']
        scan_filter_list['all'].append(idx)
        if 'filter string' in scan['scanList']['scan'][0]:
            filter = scan['scanList']['scan'][0]['filter string']
        elif 'spectrum title' in scan:
            filter = scan['spectrum title']
        else:
            continue    
        
        if filter not in scan_filter_list:
            scan_filter_list[filter] = []
        scan_filter_list[filter].append(idx)

spectrum_tab, chromatogram_tab = st.tabs(["Spectrum", "Chromatogram"])

with spectrum_tab:
    st.markdown("Explore spectra, scan by scan.")

    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        if raw_file is not None:
            # PLOT SETTINGS
            st.markdown("### Settings")
            num_scans = reader[-1]['index']
            scan_filter = st.selectbox("Select a scan filter", scan_filter_list, help="Filter scans by spectrum description. `all` shows all scans.")

            if len(scan_filter_list[scan_filter]) > 1:
                scan_number = st.select_slider("Select a scan", scan_filter_list[scan_filter], help="Only scans with matching filter can be selected.")
            else:
                scan_number = scan_filter_list[scan_filter][0]
                st.write("Selected scan: ", scan_number)

            label_threshold = st.number_input("Label Threshold (%)", min_value=0, max_value=100, value=2, help="Label peaks with intensity above threshold% of maximum.")

            labels_on = st.toggle("_m/z_ labels on", help="Display all peak labels on plot.")

            selected_scan = reader[scan_number]

            ## USE settings from col1
            if 'centroid spectrum' in scan:
                st.write("Scan contains centroid data.")
                peaks_ = detect_peaks(selected_scan, threshold = label_threshold, centroid = True)
            else:
                peaks_ = detect_peaks(selected_scan, threshold = label_threshold, centroid = False)
            peaks = ColumnDataSource(data=dict(
                x = selected_scan['m/z array'][peaks_],
                y = selected_scan['intensity array'][peaks_],
                desc = ["%.2f" % x for x in selected_scan['m/z array'][peaks_]]
                ))

            TOOLTIPS = [
                ("m/z", "@x{0.00}")
                ]

            labels = LabelSet(x='x', y='y', text='desc', source=peaks, text_font_size='8pt')


    with col2:
        if raw_file is not None and selected_scan:
            
            if 'filter string' in scan['scanList']['scan'][0]:
                filter = scan['scanList']['scan'][0]['filter string']
            elif 'spectrum title' in scan:
                filter = scan['spectrum title']
            else:
                filter = ""
            spectrum_title = f"#{scan_number}; {filter}"
            
            spectrum_plot = figure(
                                    title=spectrum_title,
                                    x_axis_label='m/z',
                                    y_axis_label='intensity',
                                    tools='pan,box_zoom,xbox_zoom,reset,save',
                                    active_drag='xbox_zoom'
                                    )
            # Format axes
            spectrum_plot.left[0].formatter.use_scientific = True
            spectrum_plot.left[0].formatter.power_limit_high = 0
            spectrum_plot.left[0].formatter.precision = 1
            spectrum_plot.y_range.start = 0
            
            if 'scanWindowList' in selected_scan['scanList']['scan'][0]:
                min_mz = selected_scan['scanList']['scan'][0]['scanWindowList']['scanWindow'][0]['scan window lower limit']
                max_mz = selected_scan['scanList']['scan'][0]['scanWindowList']['scanWindow'][0]['scan window upper limit']
                spectrum_plot.x_range = Range1d(min_mz, max_mz)

            # PLOT SPECTRUM (depends if centroid or profile data)
            if 'centroid spectrum' in selected_scan:
                spectrum_plot.vbar(x=selected_scan['m/z array'], top=selected_scan['intensity array'], width=0.01, color='black')
            else:
                spectrum_plot.line(selected_scan['m/z array'], selected_scan['intensity array'], line_width = 1.5, color='black')
            
            # Set Peak labelling
            r = spectrum_plot.circle('x', 'y', source=peaks, alpha=0.2, size = 8, hover_alpha=0.8)
            
            if labels_on:
                spectrum_plot.add_layout(labels)
            hover = HoverTool(renderers=[r], tooltips=TOOLTIPS)
            spectrum_plot.add_tools(hover)

            st.bokeh_chart(spectrum_plot, use_container_width=True)

with chromatogram_tab:
    st.write("Coming soon!")