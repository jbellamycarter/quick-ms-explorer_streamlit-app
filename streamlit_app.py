"""
(c) 2024, Jedd Bellamy-Carter

Test streamlit app that plots a scan from an mzML data file.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from bokeh.models import ColumnDataSource, LabelSet, HoverTool
from bokeh.plotting import figure

from pyteomics import mgf, mzml, pylab_aux, mass, parser

## FUNCTIONS ##

def detect_peaks(spectrum, threshold=5, distance=4, prominence=0.8, width=3):
    """Peak picking from a given spectrum using the relative maxima
    algorithm using a window of size order.

    Only peaks above the specified threshold are returned

    Inputs
    ------

    spectrum : spectrum object from pyteomics

    """
    rel_threshold = spectrum['intensity array'].max() * (threshold / 100)
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
        filter = scan['scanList']['scan'][0]['filter string']
        if filter not in scan_filter_list:
            scan_filter_list[filter] = []
        scan_filter_list[filter].append(idx)
        scan_filter_list['all'].append(idx)

col1, col2 = st.columns([0.3, 0.7])

with col1:
    if raw_file is not None:
       
        num_scans = reader[-1]['index']
        scan_filter = st.selectbox("Select a scan filter", scan_filter_list)
        scan_number = st.select_slider("Select a scan", scan_filter_list[scan_filter])

        label_threshold = st.number_input("Label Threshold (%)", min_value=0, max_value=100, value=2)

        labels_on = st.toggle("m/z labels on")

        scan = reader[scan_number]
        peaks_ = detect_peaks(scan, threshold = label_threshold)
        peaks = ColumnDataSource(data=dict(
            x = scan['m/z array'][peaks_],
            y = scan['intensity array'][peaks_],
            desc = ["%.2f" % x for x in scan['m/z array'][peaks_]]
            ))

        TOOLTIPS = [
            ("m/z", "@x")
            ]

        labels = LabelSet(x='x', y='y', text='desc', source=peaks)

with col2:
    if raw_file is not None:
        spectrum_plot = figure(
                                title=scan['scanList']['scan'][0]['filter string'],
                                x_axis_label='m/z',
                                y_axis_label='intensity',
                                tools='pan,box_zoom,xbox_zoom,reset,save',
                                active_drag='xbox_zoom'
                                )

        spectrum_plot.line(scan['m/z array'], scan['intensity array'], line_width = 1.5, color='black')
        r = spectrum_plot.circle('x', 'y', source=peaks, alpha=0.7, size = 8)
        if labels_on:
            spectrum_plot.add_layout(labels)
        hover = HoverTool(renderers=[r], tooltips=TOOLTIPS)
        spectrum_plot.add_tools(hover)

        st.markdown("### Spectrum Plot")
        st.bokeh_chart(spectrum_plot, use_container_width=True)
