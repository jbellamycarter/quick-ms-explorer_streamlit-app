"""
(c) 2024, Jedd Bellamy-Carter

Test streamlit app that plots a scan from an mzML data file.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from bokeh.models import ColumnDataSource, LabelSet, HoverTool, Range1d, Toggle
from bokeh.plotting import figure

from pyteomics import mgf, mzml, pylab_aux, mass, parser

## FUNCTIONS ##

def detect_peaks(spectrum, threshold=5, distance=4, prominence=0.8, width=2, centroid=False):
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

def average_spectra(spectra, bin_width=None):
    """Average several spectra into one spectrum. Tolerant to variable m/z bins.
    Assumes spectra are scans from pyteomics reader object."""

    ref_scan = np.unique(spectra[0]['m/z array'])
    if bin_width is None:
        bin_width = np.min(np.diff(ref_scan)) # Determines minimum spacing between m/z for interpolation
    ref_mz = np.arange(ref_scan[0], ref_scan[-1], bin_width)
    merge_int = np.zeros_like(ref_mz)

    for scan in spectra:
        tmp_mz = scan['m/z array']
        tmp_int = scan['intensity array']
        merge_int += np.interp(ref_mz, tmp_mz, tmp_int, left=0, right=0)

    merge_int = merge_int / len(spectra)

    avg_spec = spectra[0].copy()  # Make copy of first spectrum metadata
    avg_spec['m/z array'] = ref_mz
    avg_spec['intensity array'] = merge_int
    avg_spec['scanList']['scan'][0]['filter string'] = "AV: {:.2f} - {:.2f}".format(spectra[0]['scanList']['scan'][0]['scan start time'], spectra[-1]['scanList']['scan'][0]['scan start time'])

    return avg_spec

def get_xic(mz, scans, mz_tol=0.1, ms_level=1):
    xic = []
    rt = []
    for i, scan in enumerate(scans):
        if scan['ms level'] is not ms_level:
            continue
        rt.append(scan['scanList']['scan'][0]['scan start time'])
        idx = np.where(np.abs(scan['m/z array'] - mz) < mz_tol)[0]
        if idx.any():
            scan_int = scan['intensity array'][idx].sum()
        else:
            scan_int = 0
        xic.append(scan_int)
    return {'time array': np.array(rt), 'intensity array': np.array(xic)}

def generate_tic_bpc(_data_reader):
    total_ion_chromatograms = {}
    base_peak_chromatograms = {}
    _data_reader.reset()
    for scan in _data_reader:
        _ms_level = scan['ms level']
        if _ms_level not in total_ion_chromatograms:
            total_ion_chromatograms[_ms_level] = {'time array': [], 'intensity array': []}
            base_peak_chromatograms[_ms_level] = {'time array': [], 'intensity array': []}
        _time = scan['scanList']['scan'][0]['scan start time']
        total_ion_chromatograms[_ms_level]['time array'].append(_time)
        base_peak_chromatograms[_ms_level]['time array'].append(_time)
        total_ion_chromatograms[_ms_level]['intensity array'].append(scan['total ion current'])
        base_peak_chromatograms[_ms_level]['intensity array'].append(scan['base peak intensity'])
    
    for _ms_level in total_ion_chromatograms:
        total_ion_chromatograms[_ms_level]['time array'] = np.array(total_ion_chromatograms[_ms_level]['time array'])
        total_ion_chromatograms[_ms_level]['intensity array'] = np.array(total_ion_chromatograms[_ms_level]['intensity array'])
        base_peak_chromatograms[_ms_level]['time array'] = np.array(base_peak_chromatograms[_ms_level]['time array'])
        base_peak_chromatograms[_ms_level]['intensity array'] = np.array(base_peak_chromatograms[_ms_level]['intensity array'])

    return total_ion_chromatograms, base_peak_chromatograms

## APP LAYOUT ##

st.set_page_config(page_title= "Quick mzML Explorer", layout="wide", menu_items = {'about': "This is a very simple data explorer for mzML mass spectrometry data. Written by Jedd Bellamy-Carter (Loughborough University, UK)."})
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

    # Generate TIC and BPC
    total_ion_chromatograms, base_peak_chromatograms = generate_tic_bpc(reader)

spectrum_tab, chromatogram_tab = st.tabs(["Spectrum", "Chromatogram"])

with spectrum_tab:
    st.markdown("Explore spectra, scan by scan.")

    scol1, scol2 = st.columns([0.3, 0.7])

    with scol1:
        if raw_file is not None:
            # PLOT SETTINGS
            st.markdown("### Settings")
            scan_filter = st.selectbox("Select a scan filter", scan_filter_list, help="Filter scans by spectrum description. `all` shows all scans.")

            if len(scan_filter_list[scan_filter]) > 1:
                tog_avg_scans = st.toggle("Average scans", help="Toggle whether to generate averaged spectrum.")
                if tog_avg_scans:
                    scan_range = st.select_slider("Select scans to average", scan_filter_list[scan_filter], value=(scan_filter_list[scan_filter][0], scan_filter_list[scan_filter][-1]), help="Only scans with matching filter can be selected.")
                    scan_number = "{}-{}".format(*scan_range)
                else:
                    scan_number = st.select_slider("Select a scan to display", scan_filter_list[scan_filter], help="Only scans with matching filter can be selected.")

            else:
                scan_number = scan_filter_list[scan_filter][0]
                st.write("Selected scan: ", scan_number)

            label_threshold = st.number_input("Label Threshold (%)", min_value=0, max_value=100, value=2, help="Label peaks with intensity above threshold% of maximum.")

            labels_on = st.toggle("_m/z_ labels on", help="Display all peak labels on plot.")

            if tog_avg_scans:
                if 'centroid spectrum' in reader[scan_range[0]]:
                    selected_scan = average_spectra(reader[scan_range[0]:scan_range[1]], bin_width=0.5)
                else:
                    selected_scan = average_spectra(reader[scan_range[0]:scan_range[1]])
            else:
                selected_scan = reader[scan_number]
            scan_start_time = selected_scan['scanList']['scan'][0]['scan start time']

            st.write("Scan time: ", scan_start_time, scan_start_time.unit_info)

            ## USE settings from scol1
            if 'centroid spectrum' in selected_scan:
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
                ("m/z", "@x{0.00}"),
                ("int", "@y{0.0}")
                ]

            labels = LabelSet(x='x', y='y', text='desc', source=peaks, text_font_size='8pt', text_color='black')


    with scol2:
        if raw_file is not None and selected_scan:
            
            if 'filter string' in selected_scan['scanList']['scan'][0]:
                filter = selected_scan['scanList']['scan'][0]['filter string']
            elif 'spectrum title' in selected_scan:
                filter = selected_scan['spectrum title']
            else:
                filter = ""
            spectrum_title = f"#{scan_number}; {filter}"
            
            spectrum_plot = figure(
                                    title=raw_file.name + "\n" + spectrum_title,
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
            r = spectrum_plot.circle('x', 'y', source=peaks, alpha=0.2, size = 8, hover_alpha=0.8, color='dodgerblue')
            
            if labels_on:
                spectrum_plot.add_layout(labels)
            spec_hover = HoverTool(renderers=[r], tooltips=TOOLTIPS)
            spectrum_plot.add_tools(spec_hover)

            st.bokeh_chart(spectrum_plot, use_container_width=True)

            if st.button("Show data"):
                st.write(pd.DataFrame({'m/z': selected_scan['m/z array'], 'intensity': selected_scan['intensity array']}))

with chromatogram_tab:
    st.markdown("Explore chromatograms. Generate eXtracted Ion Chromatograms (XIC) for selected ions.")

    ccol1, ccol2 = st.columns([0.3, 0.7])

    with ccol1:
        if raw_file is not None:
            # PLOT SETTINGS
            st.markdown("### Settings")

            chromatogram_type = st.radio("Chromatogram type", ['TIC', 'BPC', 'XIC'], horizontal=True, help="`TIC`: total ion chromatogram.  `BPC`: base peak chromatogram.  `XIC`: extracted ion chromatogram")
            ms_level = st.selectbox("MS Level", [1,2], index=1, help="Level of MS (i.e. `2` for MS/MS).")
            
            if chromatogram_type == 'TIC':
                selected_chromatogram = total_ion_chromatograms[ms_level]
                chromatogram_title = "TIC"
            elif chromatogram_type == 'BPC':
                selected_chromatogram = base_peak_chromatograms[ms_level]
                chromatogram_title = "BPC"
            else:
                selected_mz = st.number_input("Select _m/z_ to extract.")
                mz_tolerance = st.number_input("Window (u)", value=0.1, help="Window (+/-) around selected _m/z_ to generate chromatogram.")
                selected_chromatogram = get_xic(selected_mz, reader[0:-1], mz_tolerance, ms_level)
                chromatogram_title = f"XIC: {selected_mz}, {mz_tolerance}"

            chrom_peaks_ = detect_peaks(selected_chromatogram, threshold = 2)
            chrom_peaks = ColumnDataSource(data=dict(
                x = selected_chromatogram['time array'][chrom_peaks_],
                y = selected_chromatogram['intensity array'][chrom_peaks_],
                desc = ["%.2f" % x for x in selected_chromatogram['time array'][chrom_peaks_]]
                ))

            CHROMTOOLTIPS = [
                ("time", "@x{0.00}"),
                ("int", "@y{0.0}")
                ]

    with ccol2:
        if raw_file is not None:
            
            chromatogram_plot = figure(
                                    title=raw_file.name + "\n" + chromatogram_title,
                                    x_axis_label='time',
                                    y_axis_label='intensity',
                                    tools='pan,box_zoom,xbox_zoom,reset,save',
                                    active_drag='xbox_zoom'
                                    )
            # Format axes
            chromatogram_plot.left[0].formatter.use_scientific = True
            chromatogram_plot.left[0].formatter.power_limit_high = 0
            chromatogram_plot.left[0].formatter.precision = 1
            chromatogram_plot.y_range.start = 0

            # PLOT Chromatogram

            chromatogram_plot.line(selected_chromatogram['time array'], selected_chromatogram['intensity array'], line_width=1.5, color='black')
            cr = chromatogram_plot.circle('x', 'y', source=chrom_peaks, alpha=0.2, size = 8, hover_alpha=0.8, color='dodgerblue')
            chrom_hover = HoverTool(renderers=[cr], tooltips=CHROMTOOLTIPS)
            chromatogram_plot.add_tools(chrom_hover)
            st.bokeh_chart(chromatogram_plot, use_container_width=True)