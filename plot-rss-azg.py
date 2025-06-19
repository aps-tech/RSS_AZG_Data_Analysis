import numpy as np
import pandas as pd
import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.express as px
import re
import plotly.graph_objs as go
from dash.dash_table.Format import Format, Scheme
import io
import base64
import dash_bootstrap_components as dbc


# --------- Data Loading ---------
def load_data(filepath):
    """
    Loads data from a CSV file and parses the TimeStamp column to datetime.

    Args:
        filepath (str): Path to the data file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_csv(filepath, comment='#')
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    return df
    
def apply_detector_scaling(dff, detectors=2, env_corr=5.89, AZG_factor=0.15):
    try:
        env_corr = float(env_corr)
    except (TypeError, ValueError):
        env_corr = 1.0  # fallback
    
    try:
        AZG_factor = float(AZG_factor)
    except (TypeError, ValueError):
        AZG_factor = 0.15  # fallback
    
    for i in range(1, 25):
        b_col = f'B{i}'
        if b_col in dff.columns:
            dff[b_col] = dff[b_col] * 3 / detectors * env_corr * AZG_factor
    return dff
    #not being used anymore since detector-select rewrite



# --------- Initialize the Dash App ---------
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

# --------- Load Initial Data ---------
df = load_data("Data_example.txt")

# --------- Data Cleaning: Remove Jan 2000 rows ---------
# Handles TimeStamp as either string or datetime
if pd.api.types.is_datetime64_any_dtype(df['TimeStamp']):
    mask = ~df['TimeStamp'].dt.strftime('%Y-%m').eq('2000-01')
else:
    mask = ~df['TimeStamp'].astype(str).str.startswith('2000-01')

df = df[mask].reset_index(drop=True)

# --------- Ensure B1 through B24 Exist ---------
for i in range(1, 25):
    col_name = f'B{i}'
    if col_name not in df.columns:
        df[col_name] = np.nan

# --------- Compute Total Dwell Time (TotDwell) ---------
# This finds all columns named Dwell1..Dwell24 (case-insensitive) and sums them for each row
Dwell_cols = [col for col in df.columns if col.lower().startswith('dwell')]
Dwell_cols = [col for col in Dwell_cols if re.fullmatch(r'Dwell([1-9]|1[0-9]|2[0-4])', col, re.IGNORECASE)]
Dwell_cols = sorted(Dwell_cols, key=lambda x: int(re.findall(r'\d+', x)[0]))
df['TotDwell'] = df[Dwell_cols].sum(axis=1)

# --------- Exclude Some Columns from Display ---------
excluded_cols = [
    'RecordStatus',
    'Status',
    'Excursions',
    'Temperature',
    'TotalDetectorCurrent',
    'DetectorVoltage'
]
excluded_cols_set = {x.lower() for x in excluded_cols}

# --------- Hide Certain Columns (by Prefix or B number) from Table by Default ---------
hide_prefixes = [
    "Dwell", "D1Bin", "D2Bin", "D3Bin", "B"
]
b_regex = re.compile(r"^B([1-9]|1[0-9]|2[0-4])$", re.IGNORECASE)

# ---------Gather all display columns (after exclusions above)--------------------------
display_columns = [col for col in df.columns if col.lower() not in excluded_cols_set]

# ----------------------Find columns to hide by default--------------------------------
hidden_columns = []
for col in display_columns:
    if any(col.startswith(prefix) for prefix in hide_prefixes) or b_regex.match(col):
        hidden_columns.append(col)
visible_columns = [col for col in display_columns if col not in hidden_columns]
checklist_options = [{'label': col, 'value': col} for col in display_columns]

# --------- Find Numeric Columns (for plotting, dropdowns, etc) ---------
numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in excluded_cols]

# --------- Explicit Calculation of B1...B24 columns ---------
# Calculating counts per millisecond (counts/dwell) in each bin from 3 detectors 
# and averaging them out (see note in spec about this not being a valid method)
# Written out instead of using a loop, for manual checking of validity

# B1
df['B1D1_scaled'] = np.where(df['Dwell1'] != 0, df['D1Bin1'] / df['Dwell1'] * 1000, np.nan)
df['B1D2_scaled'] = np.where(df['Dwell9'] != 0, df['D2Bin1'] / df['Dwell9'] * 1000, np.nan)
df['B1D3_scaled'] = np.where(df['Dwell17'] != 0, df['D3Bin1'] / df['Dwell17'] * 1000, np.nan)
df['B1'] = df[['B1D1_scaled', 'B1D2_scaled', 'B1D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B2
df['B2D1_scaled'] = np.where(df['Dwell2']  != 0, df['D1Bin2']  / df['Dwell2'] * 1000,  np.nan)
df['B2D2_scaled'] = np.where(df['Dwell10']  != 0, df['D2Bin2']  / df['Dwell10'] * 1000, np.nan)
df['B2D3_scaled'] = np.where(df['Dwell18'] != 0, df['D3Bin2']  / df['Dwell18'] * 1000,  np.nan)
df['B2'] = df[['B2D1_scaled', 'B2D2_scaled', 'B2D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B3
df['B3D1_scaled'] = np.where(df['Dwell3']  != 0, df['D1Bin3']  / df['Dwell3'] * 1000,  np.nan)
df['B3D2_scaled'] = np.where(df['Dwell11'] != 0, df['D2Bin3']  / df['Dwell11'] * 1000, np.nan)
df['B3D3_scaled'] = np.where(df['Dwell19'] != 0, df['D3Bin3']  / df['Dwell19'] * 1000, np.nan)
df['B3'] = df[['B3D1_scaled', 'B3D2_scaled', 'B3D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B4
df['B4D1_scaled'] = np.where(df['Dwell4']  != 0, df['D1Bin4']  / df['Dwell4'] * 1000,  np.nan)
df['B4D2_scaled'] = np.where(df['Dwell12'] != 0, df['D2Bin4']  / df['Dwell12'] * 1000, np.nan)
df['B4D3_scaled'] = np.where(df['Dwell20'] != 0, df['D3Bin4']  / df['Dwell20'] * 1000, np.nan)
df['B4'] = df[['B4D1_scaled', 'B4D2_scaled', 'B4D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B5
df['B5D1_scaled'] = np.where(df['Dwell5']  != 0, df['D1Bin5']  / df['Dwell5'] * 1000,  np.nan)
df['B5D2_scaled'] = np.where(df['Dwell13'] != 0, df['D2Bin5']  / df['Dwell13'] * 1000, np.nan)
df['B5D3_scaled'] = np.where(df['Dwell21'] != 0, df['D3Bin5']  / df['Dwell21'] * 1000, np.nan)
df['B5'] = df[['B5D1_scaled', 'B5D2_scaled', 'B5D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B6
df['B6D1_scaled'] = np.where(df['Dwell6']  != 0, df['D1Bin6']  / df['Dwell6'] * 1000, np.nan)
df['B6D2_scaled'] = np.where(df['Dwell14'] != 0, df['D2Bin6']  / df['Dwell14'] * 1000, np.nan)
df['B6D3_scaled'] = np.where(df['Dwell22'] != 0, df['D3Bin6']  / df['Dwell22'] * 1000, np.nan)
df['B6'] = df[['B6D1_scaled', 'B6D2_scaled', 'B6D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B7
df['B7D1_scaled'] = np.where(df['Dwell7']  != 0, df['D1Bin7']  / df['Dwell7'] * 1000, np.nan)
df['B7D2_scaled'] = np.where(df['Dwell15'] != 0, df['D2Bin7']  / df['Dwell15'] * 1000, np.nan)
df['B7D3_scaled'] = np.where(df['Dwell23'] != 0, df['D3Bin7']  / df['Dwell23'] * 1000, np.nan)
df['B7'] = df[['B7D1_scaled', 'B7D2_scaled', 'B7D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B8
df['B8D1_scaled'] = np.where(df['Dwell8']  != 0, df['D1Bin8']  / df['Dwell8'] * 1000, np.nan)
df['B8D2_scaled'] = np.where(df['Dwell16'] != 0, df['D2Bin8']  / df['Dwell16'] * 1000, np.nan)
df['B8D3_scaled'] = np.where(df['Dwell24'] != 0, df['D3Bin8']  / df['Dwell24'] * 1000, np.nan)
df['B8'] = df[['B8D1_scaled', 'B8D2_scaled', 'B8D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B9
df['B9D1_scaled'] = np.where(df['Dwell9']  != 0, df['D1Bin9']  / df['Dwell9'] * 1000, np.nan)
df['B9D2_scaled'] = np.where(df['Dwell17'] != 0, df['D2Bin9']  / df['Dwell17'] * 1000, np.nan)
df['B9D3_scaled'] = np.where(df['Dwell1'] != 0, df['D3Bin9']  / df['Dwell1'] * 1000, np.nan)
df['B9'] = df[['B9D1_scaled', 'B9D2_scaled', 'B9D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B10
df['B10D1_scaled'] = np.where(df['Dwell10'] != 0, df['D1Bin10'] / df['Dwell10'] * 1000, np.nan)
df['B10D2_scaled'] = np.where(df['Dwell18'] != 0, df['D2Bin10'] / df['Dwell18'] * 1000, np.nan)
df['B10D3_scaled'] = np.where(df['Dwell2']  != 0, df['D3Bin10'] / df['Dwell2'] * 1000, np.nan)
df['B10'] = df[['B10D1_scaled', 'B10D2_scaled', 'B10D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B11
df['B11D1_scaled'] = np.where(df['Dwell11'] != 0, df['D1Bin11'] / df['Dwell11'] * 1000, np.nan)
df['B11D2_scaled'] = np.where(df['Dwell19'] != 0, df['D2Bin11'] / df['Dwell19'] * 1000, np.nan)
df['B11D3_scaled'] = np.where(df['Dwell3']  != 0, df['D3Bin11'] / df['Dwell3'] * 1000, np.nan)
df['B11'] = df[['B11D1_scaled', 'B11D2_scaled', 'B11D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B12
df['B12D1_scaled'] = np.where(df['Dwell12'] != 0, df['D1Bin12'] / df['Dwell12'] * 1000, np.nan)
df['B12D2_scaled'] = np.where(df['Dwell20'] != 0, df['D2Bin12'] / df['Dwell20'] * 1000, np.nan)
df['B12D3_scaled'] = np.where(df['Dwell4']  != 0, df['D3Bin12'] / df['Dwell4'] * 1000, np.nan)
df['B12'] = df[['B12D1_scaled', 'B12D2_scaled', 'B12D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B13
df['B13D1_scaled'] = np.where(df['Dwell13'] != 0, df['D1Bin13'] / df['Dwell13'] * 1000, np.nan)
df['B13D2_scaled'] = np.where(df['Dwell21'] != 0, df['D2Bin13'] / df['Dwell21'] * 1000, np.nan)
df['B13D3_scaled'] = np.where(df['Dwell5']  != 0, df['D3Bin13'] / df['Dwell5'] * 1000, np.nan)
df['B13'] = df[['B13D1_scaled', 'B13D2_scaled', 'B13D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B14
df['B14D1_scaled'] = np.where(df['Dwell14'] != 0, df['D1Bin14'] / df['Dwell14'] * 1000, np.nan)
df['B14D2_scaled'] = np.where(df['Dwell22'] != 0, df['D2Bin14'] / df['Dwell22'] * 1000, np.nan)
df['B14D3_scaled'] = np.where(df['Dwell6']  != 0, df['D3Bin14'] / df['Dwell6'] * 1000, np.nan)
df['B14'] = df[['B14D1_scaled', 'B14D2_scaled', 'B14D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B15
df['B15D1_scaled'] = np.where(df['Dwell15'] != 0, df['D1Bin15'] / df['Dwell15'] * 1000, np.nan)
df['B15D2_scaled'] = np.where(df['Dwell23'] != 0, df['D2Bin15'] / df['Dwell23'] * 1000, np.nan)
df['B15D3_scaled'] = np.where(df['Dwell7']  != 0, df['D3Bin15'] / df['Dwell7'] * 1000, np.nan)
df['B15'] = df[['B15D1_scaled', 'B15D2_scaled', 'B15D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B16
df['B16D1_scaled'] = np.where(df['Dwell16'] != 0, df['D1Bin16'] / df['Dwell16'] * 1000, np.nan)
df['B16D2_scaled'] = np.where(df['Dwell24'] != 0, df['D2Bin16'] / df['Dwell24'] * 1000, np.nan)
df['B16D3_scaled'] = np.where(df['Dwell8']  != 0, df['D3Bin16'] / df['Dwell8'] * 1000, np.nan)
df['B16'] = df[['B16D1_scaled', 'B16D2_scaled', 'B16D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B17
df['B17D1_scaled'] = np.where(df['Dwell17'] != 0, df['D1Bin17'] / df['Dwell17'] * 1000, np.nan)
df['B17D2_scaled'] = np.where(df['Dwell1'] != 0, df['D2Bin17'] / df['Dwell1'] * 1000, np.nan)
df['B17D3_scaled'] = np.where(df['Dwell9']  != 0, df['D3Bin17'] / df['Dwell9'] * 1000, np.nan)
df['B17'] = df[['B17D1_scaled', 'B17D2_scaled', 'B17D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B18
df['B18D1_scaled'] = np.where(df['Dwell18'] != 0, df['D1Bin18'] / df['Dwell18'] * 1000, np.nan)
df['B18D2_scaled'] = np.where(df['Dwell2']  != 0, df['D2Bin18'] / df['Dwell2'] * 1000, np.nan)
df['B18D3_scaled'] = np.where(df['Dwell10']  != 0, df['D3Bin18'] / df['Dwell10'] * 1000, np.nan)
df['B18'] = df[['B18D1_scaled', 'B18D2_scaled', 'B18D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B19
df['B19D1_scaled'] = np.where(df['Dwell19'] != 0, df['D1Bin19'] / df['Dwell19'] * 1000, np.nan)
df['B19D2_scaled'] = np.where(df['Dwell3']  != 0, df['D2Bin19'] / df['Dwell3'] * 1000, np.nan)
df['B19D3_scaled'] = np.where(df['Dwell11'] != 0, df['D3Bin19'] / df['Dwell11'] * 1000, np.nan)
df['B19'] = df[['B19D1_scaled', 'B19D2_scaled', 'B19D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B20
df['B20D1_scaled'] = np.where(df['Dwell20'] != 0, df['D1Bin20'] / df['Dwell20'] * 1000, np.nan)
df['B20D2_scaled'] = np.where(df['Dwell4']  != 0, df['D2Bin20'] / df['Dwell4'] * 1000, np.nan)
df['B20D3_scaled'] = np.where(df['Dwell12'] != 0, df['D3Bin20'] / df['Dwell12'] * 1000, np.nan)
df['B20'] = df[['B20D1_scaled', 'B20D2_scaled', 'B20D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B21
df['B21D1_scaled'] = np.where(df['Dwell21'] != 0, df['D1Bin21'] / df['Dwell21'] * 1000, np.nan)
df['B21D2_scaled'] = np.where(df['Dwell5']  != 0, df['D2Bin21'] / df['Dwell5'] * 1000, np.nan)
df['B21D3_scaled'] = np.where(df['Dwell13'] != 0, df['D3Bin21'] / df['Dwell13'] * 1000, np.nan)
df['B21'] = df[['B21D1_scaled', 'B21D2_scaled', 'B21D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B22
df['B22D1_scaled'] = np.where(df['Dwell22'] != 0, df['D1Bin22'] / df['Dwell22'] * 1000, np.nan)
df['B22D2_scaled'] = np.where(df['Dwell6']  != 0, df['D2Bin22'] / df['Dwell6'] * 1000, np.nan)
df['B22D3_scaled'] = np.where(df['Dwell14'] != 0, df['D3Bin22'] / df['Dwell14'] * 1000, np.nan)
df['B22'] = df[['B22D1_scaled', 'B22D2_scaled', 'B22D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B23
df['B23D1_scaled'] = np.where(df['Dwell23'] != 0, df['D1Bin23'] / df['Dwell23'] * 1000, np.nan)
df['B23D2_scaled'] = np.where(df['Dwell7']  != 0, df['D2Bin23'] / df['Dwell7'] * 1000, np.nan)
df['B23D3_scaled'] = np.where(df['Dwell15'] != 0, df['D3Bin23'] / df['Dwell15'] * 1000, np.nan)
df['B23'] = df[['B23D1_scaled', 'B23D2_scaled', 'B23D3_scaled']].sum(axis=1, skipna=True, min_count=1)

# B24
df['B24D1_scaled'] = np.where(df['Dwell24'] != 0, df['D1Bin24'] / df['Dwell24'] * 1000, np.nan)
df['B24D2_scaled'] = np.where(df['Dwell8']  != 0, df['D2Bin24'] / df['Dwell8'] * 1000, np.nan)
df['B24D3_scaled'] = np.where(df['Dwell16'] != 0, df['D3Bin24'] / df['Dwell16'] * 1000, np.nan)
df['B24'] = df[['B24D1_scaled', 'B24D2_scaled', 'B24D3_scaled']].sum(axis=1, skipna=True, min_count=1)




# ---------------------------------------------------------
# --------- DASH LAYOUT SECTION (UI elements) -------------
# ---------------------------------------------------------

app.layout = html.Div([
    html.H1("RSS-AZG Data Analysis Tool"),
    
    # --- Record Range Selectors ---
    html.H3("Select range of records to display:"),
    html.Div([
        html.Label("Start Record Number (minimum is 1):"),
        dcc.Input(
            id='start-record',
            type='number',
            min=int(df['RecordNumber'].min()),
            max=int(df['RecordNumber'].max()),
            value=4540,
            step=1,
            style={'width': '120px'}
        ),
        html.Label("End Record Number:"),
        dcc.Input(
            id='end-record',
            type='number',
            min=int(df['RecordNumber'].min()),
            max=int(df['RecordNumber'].max()),
            value=4700,
            step=1,
            style={'width': '120px'}
        ),
        # Shows the maximum record number
        html.Span(
            id='max-record-display',
            children=f"(Max Record number for this file is {int(df['RecordNumber'].max())})",
            style={'margin-left': '16px', 'alignSelf': 'center', 'fontWeight': 'bold', 'color': '#00528C'}
        ),

        
    ], style={'margin-bottom': '20px', 'display': 'flex', 'gap': '20px'}),
    # --- Color scale controls and gamma scale buttons ---
    html.Div([
        html.Label("Min color scale:"),
        dcc.Input(id='manual-min', type='number', value=0, debounce=True, step=0.01, style={'width': '120px'}),
        html.Label("Max color scale:"),
        dcc.Input(id='manual-max', type='number', value=300, debounce=True, step=0.01, style={'width': '120px'}),
        html.Button("Set gamma scale to 0-200", id='set-b-scale', n_clicks=0, style={'margin-left': '20px', 'margin-right': '10px'}),
        html.Button("Auto Color Range (auto-updates with record range change)", id='reset-b-scale', n_clicks=0)
    ], style={'margin-bottom': '20px', 'display': 'flex', 'gap': '20px'}),


   

    

    
    # --- Heatmap/image by URDLR (Shifted by BinRefAtStart) ---
    dcc.Graph(id='b-heatmap-shifted'),
    
    
        html.Div([
        html.Label(
            "Select Detectors to be included in Image:",
            style={
                'fontSize': '1.4em',
                'fontWeight': 'bold',
                'margin-bottom': '6px',
                'display': 'block',
                'textAlign': 'center'
            }
        ),
        dcc.Checklist(
            id='detector-select',
            options=[
                {'label': 'D1', 'value': 'D1'},
                {'label': 'D2', 'value': 'D2'},
                {'label': 'D3', 'value': 'D3'}
            ],
            value=['D1', 'D2', 'D3'],
            labelStyle={
                'display': 'inline-block',
                'margin-right': '32px',
                'fontSize': '1.25em',
                'transform': 'scale(1.4)',
                'verticalAlign': 'middle',
            },
            inputStyle={
                'marginRight': '10px',
                'width': '22px',
                'height': '22px',
                'verticalAlign': 'middle',
            },
            style={
                'justifyContent': 'center',    # centers the checkboxes
                'display': 'flex',
                'gap': '20px'
            }
        )
    ], style={
        'margin-bottom': '24px',
        'margin-top': '14px',
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center'   # centers the children of this Div
    }),
    
    
    # --- RPM filtering controls (applied only to heatmaps) to hide data at connections/overnight/not drilling ---
    html.Div([
        html.Button("Hide data below 30 RPM in heatmap", id='hide-below-30rpm', n_clicks=0, style={'margin-right': '10px'}),
        html.Button("Restore sliding data", id='show-all-rpm', n_clicks=0),
        html.Label("Heatmap Shift Mode:", style={'margin-left': '30px', 'margin-right': '8px'}),
        dcc.RadioItems(
            id='heatmap-shift-mode',
            options=[
                {'label': 'Shift by BinRefAtStart (UP RIGHT DOWN LEFT UP)', 'value': 'shift'},
                {'label': 'No Shift (Raw Bins)', 'value': 'raw'}
            ],
            value='raw',
            labelStyle={'display': 'inline-block', 'margin-right': '16px'}
        )
    ], style={'margin-bottom': '20px', 'margin-top': '10px', 'display': 'flex', 'alignItems': 'center'}),



# --- Field-only plots ---
# --- Field-only plots ---
html.Div([
    dbc.Button(
        "Show Field-Print-Only/Test Facility Plot Controls",
        id="collapse-fieldplots-btn",
        color="primary",
        className="mb-2",
        n_clicks=0,
        style={"margin-bottom": "8px"}
    ),
    dbc.Collapse(
        id="collapse-fieldplots",
        is_open=False,  # Collapsed by default!
        children=[
            html.Div([
                html.H4(
                    "Test facility plot generation, for field logs only, NOT for final logs. No guarantee of accuracy.",
                    style={'margin-bottom': '12px'}
                ),
                # Depth Range Row
                html.Div([
                    html.Label("Start Depth:", style={'margin-right': '8px'}),
                    dcc.Input(id='start-depth', type='number', value=0, style={'margin-right': '24px', 'width': '100px'}),
                    html.Label("End Depth:", style={'margin-right': '8px'}),
                    dcc.Input(id='end-depth', type='number', value=100, style={'margin-right': '24px', 'width': '100px'}),
                    html.Button("Apply EasyDepth range to visible plot only", id='apply-depth', n_clicks=0, style={'margin-left': '12px'}),
                ], style={'margin-bottom': '12px', 'display': 'flex', 'alignItems': 'center'}),
                html.Div(id='depth-apply-status', style={'color': '#00528C', 'margin-bottom': '12px'}),

                # Upload row
                html.Div([
                    html.Span("Apply real Time-Depth file", style={'margin-right': '16px'}),
                    dcc.Upload(
                        id='upload-timedepth',
                        children=html.Button('Load Time Depth file'),
                        accept='.txt,.csv,.tsv,.dat,text/plain',
                        multiple=False,
                        style={'display': 'inline-block'}
                    ),
                ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '10px'}),

                # Environmental Correction Row
                html.Div([
                    html.Label("Environmental Correction:", style={'margin-right': '8px'}),
                    dcc.Input(
                        id='env_corr',
                        type='number',
                        value=5.89,
                        style={'width': '150px'}
                    ),
                ], style={'margin-bottom': '12px', 'display': 'flex', 'alignItems': 'center'}),

                # Detector Calibrations Row
                html.Div([
                    html.Label("Detector 1 calibration:", style={'margin-right': '8px'}),
                    dcc.Input(
                        id='det1-cal',
                        type='number',
                        value=1,
                        style={'margin-right': '24px', 'width': '100px'}
                    ),
                    html.Label("Detector 2 calibration:", style={'margin-right': '8px'}),
                    dcc.Input(
                        id='det2-cal',
                        type='number',
                        value=1,
                        style={'margin-right': '24px', 'width': '100px'}
                    ),
                    html.Label("Detector 3 calibration:", style={'margin-right': '8px'}),
                    dcc.Input(
                        id='det3-cal',
                        type='number',
                        value=1,
                        style={'width': '100px'}
                    ),
                ], style={'margin-bottom': '12px', 'display': 'flex', 'alignItems': 'center'}),

                # AZG Factor Row
                html.Div([
                    html.Label("AZG to standalone factor", style={'margin-right': '8px'}),
                    dcc.Input(
                        id='AZG_factor',
                        type='number',
                        value=0.15,
                        style={'width': '150px'}
                    ),
                ], style={'margin-bottom': '16px', 'display': 'flex', 'alignItems': 'center'}),

                # Tool Configuration Row
                html.Div([
                    html.Label("Select tool configuration:", style={'margin-right': '16px'}),
                    dcc.RadioItems(
                        id='detector-tool-switch',
                        options=[
                            {'label': '2 Detector tool (RSS-AZG)', 'value': '2det'},
                            {'label': '3 Detector tool (AZG)', 'value': '3det'}
                        ],
                        value='2det',
                        labelStyle={'display': 'inline-block', 'margin-right': '32px'}
                    )
                ], style={'margin-bottom': '0', 'margin-top': '8px', 'display': 'flex', 'alignItems': 'center'}),

                # Placeholder Print Button and Message
                html.Div([
                    html.Button("Print", id="print-btn", n_clicks=0, style={'width': '120px'}),
                    html.Div(id="print-placeholder-msg", style={'margin-top': '10px', 'color': '#d9534f', 'fontWeight': 'bold'})
                ], style={'margin-top': '20px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start'}),
            ],
            style={'border': '2px solid #C4C4C4', 'border-radius': '12px', 'padding': '16px', 'margin': '30px 0'}
            )
        ]
    ),
]),


    # --- Time series chart and column selector ---
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in numeric_columns],
        value='RPM' if numeric_columns else None,
        clearable=False,
        style={'width': '50%'}
    ),
    dcc.Graph(id='time-series-chart'),
    html.H3("Toggle Table Columns to Display"),
    
    
    dcc.Checklist(
        id='column-toggle',
        options=checklist_options,
        value=visible_columns,
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    ),
    
    html.Div(
    "Note that TotDwell and B1...B24 are not pulled from memory directly but instead computed here.",
    style={'margin-bottom': '10px', 'fontStyle': 'italic', 'color': '#555'}
    ),
    
    # --- Main Data Table ---
    dash_table.DataTable(
        id='data-table',
        columns=[
            {
                'name': col,
                'id': col,
                'type': 'numeric' if re.fullmatch(r"B([1-9]|1[0-9]|2[0-4])", col) else 'any',
                'format': Format(precision=2, scheme=Scheme.fixed) if re.fullmatch(r"B([1-9]|1[0-9]|2[0-4])", col) else None
            }
            for col in visible_columns
        ],
        data=df[visible_columns].to_dict('records'),
        page_size=15,
        style_table={'overflowX': 'auto'},
        style_cell={'minWidth': '80px', 'maxWidth': '180px', 'whiteSpace': 'normal'},
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        style_header={'fontWeight': 'bold'}
    ),

    # --- Store for heatmap filtering state (RPM filter) ---
    dcc.Store(id='rpm-filtered', data=False),

    # --- Data source file upload UI ---
    html.Div([
        html.Hr(),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Select Different Memory Data Source Text File - not implemented yet - save new file as Data_example.txt for now'),
            accept='.txt,.csv,.tsv,.dat,text/plain',
            multiple=False,
            style={'margin-top': '40px'}
        ),
        html.Div(id='upload-status', style={'margin-top': '10px', 'color': '#00528C'})
    ])
])


# ---------------------------------------------------------
# ----------------- DASH CALLBACKS ------------------------
# ---------------------------------------------------------

# --- Gamma scale control callback: set to 0-200 or auto depending on which button was pressed ---
@app.callback(
    [Output('manual-min', 'value'),
     Output('manual-max', 'value')],
    [Input('set-b-scale', 'n_clicks'),
     Input('reset-b-scale', 'n_clicks')],
    [State('manual-min', 'value'),
     State('manual-max', 'value')]
)
def update_manual_scale(set_clicks, reset_clicks, manual_min, manual_max):
    ctx = dash.callback_context
    if not ctx.triggered:
        # On first page load: set to 0/200
        return 0, 300
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'set-b-scale':
        return 0, 200
    elif button_id == 'reset-b-scale':
        return None, None
    return manual_min, manual_max

# --- Update the time series chart (unfiltered by RPM) ---
@app.callback(
    Output('time-series-chart', 'figure'),
    Input('column-dropdown', 'value'),
    Input('start-record', 'value'),
    Input('end-record', 'value')
#    Input('detector-tool-switch', 'value'),
#    Input('AZG_factor', 'value'),
#    Input('env_corr', 'value')
)
def update_chart(selected_col, start_rec, end_rec):
    """
    Draws a time series plot for the selected column.
    This plot is not filtered by RPM—shows all records in the user-selected range.
    """
#    detectors = 2 if detector_tool_value == '2det' else 3
    dff = df.copy()
#    dff = apply_detector_scaling(dff, detectors, env_corr_value, AZG_factor_value)

    if not selected_col or selected_col not in dff.columns:
        return {}
    # Filter to user-specified range (by RecordNumber)
    if start_rec is not None and end_rec is not None:
        df_filtered = dff[(dff['RecordNumber'] >= start_rec) & (dff['RecordNumber'] <= end_rec)]
    else:
        df_filtered = dff
    fig = px.line(df_filtered, x='RecordNumber', y=selected_col,
                  title=f"Time Series of {selected_col} (Records {start_rec} to {end_rec})",
                  labels={'RecordNumber': 'Record Number', selected_col: selected_col})
    fig.update_layout(transition_duration=500)
    return fig

# --- Update the main data table based on column toggles and record range ---
@app.callback(
    Output('data-table', 'columns'),
    Output('data-table', 'data'),
    Input('column-toggle', 'value'),
    Input('start-record', 'value'),
    Input('end-record', 'value')
#    Input('detector-tool-switch', 'value'),
#    Input('AZG_factor', 'value'),
#    Input('env_corr', 'value')
)
def update_table_columns_and_data(visible_cols, start_rec, end_rec):
    """
    Updates the columns and data in the main table.
    Applies column toggling, record range filtering, and detector scaling.
    """
    if not visible_cols:
        return [], []
    # Determine number of detectors (2 or 3) based on selector input
#    detectors = 2 if detector_tool_value == '2det' else 3

    # Copy the DataFrame for safe modifications
    dff = df.copy()

    # Scale B columns for the detector tool
#    dff = apply_detector_scaling(dff, detectors, env_corr, AZG_factor)


    # Filter dataframe to selected record range by RecordNumber
    if start_rec is not None and end_rec is not None:
        dff = dff[(dff['RecordNumber'] >= start_rec) & (dff['RecordNumber'] <= end_rec)]

    # Set up columns with formatting for B columns
    cols = [
        {
            'name': col,
            'id': col,
            'type': 'numeric' if re.fullmatch(r"B([1-9]|1[0-9]|2[0-4])", col) else 'any',
            'format': Format(precision=2, scheme=Scheme.fixed)
                if re.fullmatch(r"B([1-9]|1[0-9]|2[0-4])", col) else None
        }
        for col in visible_cols
    ]
    data = dff[visible_cols].to_dict('records')
    return cols, data





# --- Upload data file, replace DataFrame with uploaded file ---
#@app.callback(
#    Output('upload-status', 'children'),
#    Input('upload-data', 'contents'),
#    State('upload-data', 'filename')
#)
#def update_data_source(contents, filename):
#    """
#    Handles the file upload. Replaces the DataFrame with the uploaded file.
#    """
#    if contents is None:
#        return ""
#    content_type, content_string = contents.split(',')
#    decoded = base64.b64decode(content_string)
#    try:
#        # Try loading as a text file (CSV/TSV or whitespace delimited)
#        df_new = pd.read_csv(io.StringIO(decoded.decode('utf-8')), comment='#', sep=None, engine='python')
#        global df
#        df = df_new
#        return f"Data source '{filename}' loaded successfully! Please adjust display controls if needed."
#    except Exception as e:
#        return f"Error loading file '{filename}': {e}"

# --- Set RPM filter flag for heatmap filtering (True/False) ---
@app.callback(
    Output('rpm-filtered', 'data'),
    [Input('hide-below-30rpm', 'n_clicks'),
     Input('show-all-rpm', 'n_clicks')],
    prevent_initial_call=True
)
def set_rpm_filter(hide_clicks, show_clicks):
    """
    Sets a boolean store to indicate whether the heatmap should show only data with RPM >= 30.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return False
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'hide-below-30rpm':
        return True
    elif button_id == 'show-all-rpm':
        return False
    return False

# --- Update max-record display when a new file is loaded ---
@app.callback(
    Output('max-record-display', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_max_record(contents, filename):
    """
    Updates the text showing the maximum record number for the currently loaded data file.
    """
    global df
    if contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df_new = pd.read_csv(io.StringIO(decoded.decode('utf-8')), comment='#', sep=None, engine='python')
            if 'RecordNumber' in df_new.columns:
                max_rec = int(df_new['RecordNumber'].max())
                return f"(Max Record number for this file is {max_rec})"
        except Exception:
            return "(Max Record number for this file is ?)"
    # Default (startup or fallback)
    if 'RecordNumber' in df.columns:
        return f"(Max Record number for this file is {int(df['RecordNumber'].max())})"
    else:
        return "(Max Record number for this file is ?)"

# --- Heatmap with B columns circularly shifted left by BinRefAtStart (to show oriented per URDLU) ---
@app.callback(
    Output('b-heatmap-shifted', 'figure'),
    Input('start-record', 'value'),
    Input('end-record', 'value'),
    Input('manual-min', 'value'),
    Input('manual-max', 'value'),
    Input('rpm-filtered', 'data'),
    Input('detector-tool-switch', 'value'),
    Input('env_corr', 'value'),
    Input('AZG_factor', 'value'),
    Input('heatmap-shift-mode', 'value'),
    Input('detector-select', 'value')
)
def update_shifted_heatmap(start_rec, end_rec, manual_min, manual_max, rpm_filtered, detector_tool_value, env_corr_value, AZG_factor_value, shift_mode_value, selected_detectors):
    '''
    Builds the shifted heatmap/image, using the selected detectors,
    and where each row's B columns are circularly shifted left by BinRefAtStart.
    '''
    if not selected_detectors:  # Nothing selected: show empty plot.
        return go.Figure()

    # Copy/scale DataFrame for this view
#    detectors = 2 if detector_tool_value == '2det' else 3
    dff = df.copy()
#    dff = apply_detector_scaling(dff, detectors, env_corr_value, AZG_factor_value)

    # RPM filter if needed
    if rpm_filtered and 'RPM' in dff.columns:
        dff = dff[dff['RPM'] >= 30]

    if start_rec is None or end_rec is None or start_rec > end_rec:
        return go.Figure()
    df_slice = dff[(dff['RecordNumber'] >= start_rec) & (dff['RecordNumber'] <= end_rec)].copy()
    if df_slice.empty:
        return go.Figure()


    detectors = []
    if 'D1' in selected_detectors:
        detectors.append('D1')
    if 'D2' in selected_detectors:
        detectors.append('D2')
    if 'D3' in selected_detectors:
        detectors.append('D3')

    # Build the list of column names accordingly (used only for checking column existence, not for actual data matrix)
    b_cols = []
    for i in range(1, 25):
        for d in detectors:
            b_cols.append(f'B{i}{d}_scaled')




    # For each record, sum the bins for selected detectors
    z_matrix = []
    for _, row in df_slice.iterrows():
        row_bins = []
        for i in range(1, 25):
            values = []
            if 'D1' in selected_detectors:
                values.append(row.get(f'B{i}D1_scaled', np.nan))
            if 'D2' in selected_detectors:
                values.append(row.get(f'B{i}D2_scaled', np.nan))
            if 'D3' in selected_detectors:
                values.append(row.get(f'B{i}D3_scaled', np.nan))
            # sum with min_count=1 for true NaN if all values are nan
            if values:
                s = pd.Series(values).sum(skipna=True, min_count=1)
            else:
                s = np.nan
            row_bins.append(s)
        z_matrix.append(row_bins)
    z_data = np.array(z_matrix, dtype=float)

    y_labels = [
        f"{int(recnum)} | {ts.strftime('%Y-%m-%d %H:%M:%S') if not pd.isnull(ts) else ''}"
        for recnum, ts in zip(df_slice['RecordNumber'], df_slice['TimeStamp'])
    ] if 'RecordNumber' in df_slice.columns and 'TimeStamp' in df_slice.columns else df_slice.index.astype(str)
    x_labels = [f'B{i}' for i in range(1, 25)]

    # Step 4: Circularly shift by BinRefAtStart, if required
    if shift_mode_value == 'shift' and 'BinRefAtStart' in df_slice.columns:
        for idx, row in enumerate(df_slice.itertuples()):
            shift = int(getattr(row, 'BinRefAtStart')) % 24 if not pd.isnull(getattr(row, 'BinRefAtStart')) else 0
            z_data[idx] = np.roll(z_data[idx], -shift)

    all_b_vals = z_data.flatten()
    nonzero_vals = all_b_vals[(all_b_vals != 0) & ~np.isnan(all_b_vals)]
    if nonzero_vals.size > 0:
        auto_min_b = np.min(nonzero_vals)
        auto_max_b = np.max(nonzero_vals)
    else:
        auto_min_b, auto_max_b = 0, 1 # fallback for all-zero/NaN

    min_b = manual_min if manual_min is not None else auto_min_b
    max_b = manual_max if manual_max is not None else auto_max_b

    colorscale = [
        [0.0, 'white'],
        [0.25, 'yellow'],
        [0.5, 'orange'],
        [0.75, 'red'],
        [1.0, 'black']
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        colorbar=dict(title='Gamma'),
        zmin=min_b,
        zmax=max_b,
        hoverongaps=False
    ))
    fig.update_layout(
        title=f"AZG Image (Records {start_rec} to {end_rec})",
        xaxis_title="",
        yaxis_title="Record Number and TimeStamp",
        yaxis_autorange='reversed',
        margin=dict(l=60, r=40, t=40, b=60)
    )
    if shift_mode_value == 'shift':
        fig.update_xaxes(
            tickvals=[-0.5, 5.5, 11.5, 17.5, 23.5],
            ticktext=["Up", "Right", "Down", "Left", "Up"],
            range=[-0.5, 23.5]
        )
    else:
        fig.update_xaxes(
            tickvals=None,
            ticktext=None,
            range=[-0.5, 23.5]
        )
    return fig




# --- Interpolated depth calculator for test facility logs ---
@app.callback(
    Output('depth-apply-status', 'children'),
    Input('apply-depth', 'n_clicks'),
    State('start-depth', 'value'),
    State('end-depth', 'value'),
    State('start-record', 'value'),
    State('end-record', 'value'),
    State('rpm-filtered', 'data')
)
def apply_depths(n_clicks, start_depth, end_depth, start_rec, end_rec, rpm_filtered):
    """
    When operator clicks "Apply" in the Depth section, calculates
    linearly interpolated depths from Start Depth to End Depth, 
    for the records visible in the current image (with RPM filtering if active).
    """
    if n_clicks == 0 or start_depth is None or end_depth is None:
        return ""
    global df
    # Filter as the current heatmap does
    dff = df.copy()
    if rpm_filtered and 'RPM' in dff.columns:
        dff = dff[dff['RPM'] >= 30]
    if start_rec is None or end_rec is None or start_rec > end_rec:
        return "Invalid record range."
    df_slice = dff[(dff['RecordNumber'] >= start_rec) & (dff['RecordNumber'] <= end_rec)]
    n = len(df_slice)
    if n < 2:
        return "Need at least two records to interpolate depth."
    # Generate interpolated depth column
    depth_values = np.linspace(start_depth, end_depth, n)
    # Write to the relevant records in the original df
    df.loc[df_slice.index, 'Depth'] = depth_values
    return f"Depth column applied to {n} records (from {start_depth} to {end_depth} feet)."


@app.callback(
    Output('collapse-fieldplots', 'is_open'),
    Input('collapse-fieldplots-btn', 'n_clicks'),
    State('collapse-fieldplots', 'is_open')
)
def toggle_fieldplots(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("print-placeholder-msg", "children"),
    Input("print-btn", "n_clicks"),
    prevent_initial_call=True
)
def show_print_msg(n):
    if n and n > 0:
        return "This section is still being implemented"
    return ""


# --- Standard Dash main entry point ---
if __name__ == '__main__':
    app.run(debug=True)
