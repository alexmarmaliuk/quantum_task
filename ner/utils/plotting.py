import numpy as np
import scipy as sp
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

def plot_hist(data, bins=5, column_name='length', xlabel='x', ylabel='y', title='', width=800, height=500, bargap=0.1):
    """
    Plot histogram using Plotly.
    
    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - bins (int): Number of bins for the histogram.
    - column_name (str): The name of the column to plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - title (str): Title of the histogram.
    - width (int): Width of the plot in pixels.
    - height (int): Height of the plot in pixels.
    """
    fig = px.histogram(
        data, 
        x=column_name,  # Correctly refer to the column in the DataFrame
        nbins=bins, 
        title=title,
        labels={column_name: xlabel}
    )
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        bargap=bargap  
    )
    fig.update_traces(marker_color='lightblue')
    fig.show()

def plot_setup(xlabel='X', ylabel='Y', title_base=f'', title_addon=''):
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_base+'. '+title_addon)
    plt.legend()
    plt.show()