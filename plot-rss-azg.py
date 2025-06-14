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
            dff[b_col] = dff[b_col] * 1000 * 3 / detectors * env_corr * AZG_factor
    return dff



# --------- Initialize the Dash App ---------
app = Dash(__name__)

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

df['B1'] = (
    np.where(df['Dwell1']  != 0, df['D1Bin1']  / df['Dwell1'],  np.nan) +
    np.where(df['Dwell16'] != 0, df['D2Bin1']  / df['Dwell16'], np.nan) +  
    np.where(df['Dwell8']  != 0, df['D3Bin1']  / df['Dwell8'],  np.nan)    
)
df['B2'] = (
    np.where(df['Dwell2']  != 0, df['D1Bin2']  / df['Dwell2'],  np.nan) +
    np.where(df['Dwell17'] != 0, df['D2Bin2']  / df['Dwell17'], np.nan) +  
    np.where(df['Dwell9']  != 0, df['D3Bin2']  / df['Dwell9'],  np.nan)    
)
df['B3'] = (
    np.where(df['Dwell3']  != 0, df['D1Bin3']  / df['Dwell3'],  np.nan) +
    np.where(df['Dwell18'] != 0, df['D2Bin3']  / df['Dwell18'], np.nan) +
    np.where(df['Dwell10'] != 0, df['D3Bin3']  / df['Dwell10'], np.nan)
)
df['B4'] = (
    np.where(df['Dwell4']  != 0, df['D1Bin4']  / df['Dwell4'],  np.nan) +
    np.where(df['Dwell19'] != 0, df['D2Bin4']  / df['Dwell19'], np.nan) +
    np.where(df['Dwell11'] != 0, df['D3Bin4']  / df['Dwell11'], np.nan)
)
df['B5'] = (
    np.where(df['Dwell5']  != 0, df['D1Bin5']  / df['Dwell5'],  np.nan) +
    np.where(df['Dwell20'] != 0, df['D2Bin5']  / df['Dwell20'], np.nan) +
    np.where(df['Dwell12'] != 0, df['D3Bin5']  / df['Dwell12'], np.nan)
)
df['B6'] = (
    np.where(df['Dwell6']  != 0, df['D1Bin6']  / df['Dwell6'],  np.nan) +
    np.where(df['Dwell21'] != 0, df['D2Bin6']  / df['Dwell21'], np.nan) +
    np.where(df['Dwell13'] != 0, df['D3Bin6']  / df['Dwell13'], np.nan)
)
df['B7'] = (
    np.where(df['Dwell7']  != 0, df['D1Bin7']  / df['Dwell7'],  np.nan) +
    np.where(df['Dwell22'] != 0, df['D2Bin7']  / df['Dwell22'], np.nan) +
    np.where(df['Dwell14'] != 0, df['D3Bin7']  / df['Dwell14'], np.nan)
)
df['B8'] = (
    np.where(df['Dwell8']  != 0, df['D1Bin8']  / df['Dwell8'],  np.nan) +
    np.where(df['Dwell23'] != 0, df['D2Bin8']  / df['Dwell23'], np.nan) +
    np.where(df['Dwell15'] != 0, df['D3Bin8']  / df['Dwell15'], np.nan)
)
df['B9'] = (
    np.where(df['Dwell9']  != 0, df['D1Bin9']  / df['Dwell9'],  np.nan) +
    np.where(df['Dwell24'] != 0, df['D2Bin9']  / df['Dwell24'], np.nan) +
    np.where(df['Dwell16'] != 0, df['D3Bin9']  / df['Dwell16'], np.nan)
)
df['B10'] = (
    np.where(df['Dwell10'] != 0, df['D1Bin10'] / df['Dwell10'], np.nan) +
    np.where(df['Dwell1']  != 0, df['D2Bin10'] / df['Dwell1'],  np.nan) +
    np.where(df['Dwell17'] != 0, df['D3Bin10'] / df['Dwell17'], np.nan)
)
df['B11'] = (
    np.where(df['Dwell11'] != 0, df['D1Bin11'] / df['Dwell11'], np.nan) +
    np.where(df['Dwell2']  != 0, df['D2Bin11'] / df['Dwell2'],  np.nan) +
    np.where(df['Dwell18'] != 0, df['D3Bin11'] / df['Dwell18'], np.nan)
)
df['B12'] = (
    np.where(df['Dwell12'] != 0, df['D1Bin12'] / df['Dwell12'], np.nan) +
    np.where(df['Dwell3']  != 0, df['D2Bin12'] / df['Dwell3'],  np.nan) +
    np.where(df['Dwell19'] != 0, df['D3Bin12'] / df['Dwell19'], np.nan)
)
df['B13'] = (
    np.where(df['Dwell13'] != 0, df['D1Bin13'] / df['Dwell13'], np.nan) +
    np.where(df['Dwell4']  != 0, df['D2Bin13'] / df['Dwell4'],  np.nan) +
    np.where(df['Dwell20'] != 0, df['D3Bin13'] / df['Dwell20'], np.nan)
)
df['B14'] = (
    np.where(df['Dwell14'] != 0, df['D1Bin14'] / df['Dwell14'], np.nan) +
    np.where(df['Dwell5']  != 0, df['D2Bin14'] / df['Dwell5'],  np.nan) +
    np.where(df['Dwell21'] != 0, df['D3Bin14'] / df['Dwell21'], np.nan)
)
df['B15'] = (
    np.where(df['Dwell15'] != 0, df['D1Bin15'] / df['Dwell15'], np.nan) +
    np.where(df['Dwell6']  != 0, df['D2Bin15'] / df['Dwell6'],  np.nan) +
    np.where(df['Dwell22'] != 0, df['D3Bin15'] / df['Dwell22'], np.nan)
)
df['B16'] = (
    np.where(df['Dwell16'] != 0, df['D1Bin16'] / df['Dwell16'], np.nan) +
    np.where(df['Dwell7']  != 0, df['D2Bin16'] / df['Dwell7'],  np.nan) +
    np.where(df['Dwell23'] != 0, df['D3Bin16'] / df['Dwell23'], np.nan)
)
df['B17'] = (
    np.where(df['Dwell17'] != 0, df['D1Bin17'] / df['Dwell17'], np.nan) +
    np.where(df['Dwell8']  != 0, df['D2Bin17'] / df['Dwell8'],  np.nan) +
    np.where(df['Dwell24'] != 0, df['D3Bin17'] / df['Dwell24'],  np.nan)
)
df['B18'] = (
    np.where(df['Dwell18'] != 0, df['D1Bin18'] / df['Dwell18'], np.nan) +
    np.where(df['Dwell9']  != 0, df['D2Bin18'] / df['Dwell9'], np.nan) +
    np.where(df['Dwell1']  != 0, df['D3Bin18'] / df['Dwell1'],  np.nan)
)
df['B19'] = (
    np.where(df['Dwell19'] != 0, df['D1Bin19'] / df['Dwell19'], np.nan) +
    np.where(df['Dwell10'] != 0, df['D2Bin19'] / df['Dwell10'], np.nan) +
    np.where(df['Dwell2']  != 0, df['D3Bin19'] / df['Dwell2'],  np.nan)
)
df['B20'] = (
    np.where(df['Dwell20'] != 0, df['D1Bin20'] / df['Dwell20'], np.nan) +
    np.where(df['Dwell11'] != 0, df['D2Bin20'] / df['Dwell11'], np.nan) +
    np.where(df['Dwell3']  != 0, df['D3Bin20'] / df['Dwell3'],  np.nan)
)
df['B21'] = (
    np.where(df['Dwell21'] != 0, df['D1Bin21'] / df['Dwell21'], np.nan) +
    np.where(df['Dwell12'] != 0, df['D2Bin21'] / df['Dwell12'], np.nan) +
    np.where(df['Dwell4']  != 0, df['D3Bin21'] / df['Dwell4'],  np.nan)
)
df['B22'] = (
    np.where(df['Dwell22'] != 0, df['D1Bin22'] / df['Dwell22'], np.nan) +
    np.where(df['Dwell13'] != 0, df['D2Bin22'] / df['Dwell13'], np.nan) +
    np.where(df['Dwell5']  != 0, df['D3Bin22'] / df['Dwell5'],  np.nan)
)
df['B23'] = (
    np.where(df['Dwell23'] != 0, df['D1Bin23'] / df['Dwell23'], np.nan) +
    np.where(df['Dwell14'] != 0, df['D2Bin23'] / df['Dwell14'], np.nan) +
    np.where(df['Dwell6']  != 0, df['D3Bin23'] / df['Dwell6'],  np.nan)
)
df['B24'] = (
    np.where(df['Dwell24'] != 0, df['D1Bin24'] / df['Dwell24'], np.nan) +
    np.where(df['Dwell15'] != 0, df['D2Bin24'] / df['Dwell15'], np.nan) +
    np.where(df['Dwell7']  != 0, df['D3Bin24'] / df['Dwell7'],  np.nan)
)


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
    #------2/3 detector selector to use in bin calculation-----------------
    html.Div([
    html.Label("Select tool configuration:", style={'margin-right': '16px'}),
    dcc.RadioItems(
        id='detector-tool-switch',
        options=[
            {'label': '2 Detector tool (RSS-AZG)', 'value': '2det'},
            {'label': '3 Detector tool (AZG)', 'value': '3det'}
        ],
        value='3det',
        labelStyle={'display': 'inline-block', 'margin-right': '32px'}
    )
], style={'margin-bottom': '16px', 'margin-top': '16px'}),

    

    
    # --- Heatmap/image by URDLR (Shifted by BinRefAtStart) ---
    dcc.Graph(id='b-heatmap-shifted'),
    
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
            value='shift',
            labelStyle={'display': 'inline-block', 'margin-right': '16px'}
        )
    ], style={'margin-bottom': '20px', 'margin-top': '10px', 'display': 'flex', 'alignItems': 'center'}),


    # --- Field-only plots ---
    html.Div([
        html.H4("Test facility plot generation, for field logs only, NOT for final logs. No guarantee of accuracy.", style={'margin-bottom': '12px'}),
        html.Div([
            html.Label("Start Depth:"),
            dcc.Input(id='start-depth', type='number', value=0, style={'margin-right': '16px', 'width': '100px'}),
            html.Label("End Depth:"),
            dcc.Input(id='end-depth', type='number', value=100, style={'margin-right': '16px', 'width': '100px'}),
            html.Button("Apply EasyDepth", id='apply-depth', n_clicks=0, style={'margin-left': '12px'}),
        ], style={'margin-bottom': '8px'}),
        html.Div(id='depth-apply-status', style={'color': '#00528C'}),
        
        
        #---Real time-depth file load button-----
        #############DO NOT FORGET TO IMPLEMENT CALLBACK#######################
        html.Div([
        html.Span("Apply real Time-Depth file", style={'margin-right': '16px'}),
        dcc.Upload(
            id='upload-timedepth',
            children=html.Button('Load Time Depth file'),
            accept='.txt,.csv,.tsv,.dat,text/plain',
            multiple=False,
            style={'display': 'inline-block'}
        ),
        
        html.Div([
            html.Label("Environmental Correction:"), 
            dcc.Input(
                id='env_corr',
                type='number',
                value=5.89,
                style={'margin-left': '8px', 'width': '150px'}
            ),
        ], style={'margin-bottom': '8px'}),


        html.Label("Detector 1 calibration:"), #not used yet
        dcc.Input(
            id='det1-cal',
            type='number',
            value=1,
            style={'margin-right': '16px', 'width': '150px'}
        ),

        html.Label("Detector 2 calibration:"), #not used yet
        dcc.Input(
            id='det2-cal',
            type='number',
            value=1,
            style={'margin-right': '16px', 'width': '150px'}
        ),

        html.Label("Detector 3 calibration:"), #not used yet
        dcc.Input(
            id='det3-cal',
            type='number',
            value=1,
            style={'margin-right': '16px', 'width': '150px'}
        ),

        html.Div([
            html.Label("AZG to standalone factor"), 
            dcc.Input(
                id='AZG_factor',
                type='number',
                value=0.15,
                style={'margin-left': '8px', 'width': '150px'}
            ),
        ], style={'margin-bottom': '8px'}),
        
        ], style={'display': 'flex', 'alignItems': 'center', 'margin-top': '10px', 'margin-bottom': '8px'}),

        
    ], style={'border': '2px solid #C4C4C4', 'border-radius': '12px', 'padding': '16px', 'margin': '30px 0'}),



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
            children=html.Button('Select Different Memory Data Source Text File'),
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
    Input('end-record', 'value'),
    Input('detector-tool-switch', 'value'),
    Input('AZG_factor', 'value'),
    Input('env_corr', 'value')
)
def update_chart(selected_col, start_rec, end_rec, detector_tool_value, AZG_factor_value, env_corr_value):
    """
    Draws a time series plot for the selected column.
    This plot is not filtered by RPM—shows all records in the user-selected range.
    """
    detectors = 2 if detector_tool_value == '2det' else 3
    dff = df.copy()
    dff = apply_detector_scaling(dff, detectors, env_corr_value, AZG_factor_value)

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
    Input('end-record', 'value'),
    Input('detector-tool-switch', 'value'),
    Input('AZG_factor', 'value'),
    Input('env_corr', 'value')
)
def update_table_columns_and_data(visible_cols, start_rec, end_rec, detector_tool_value, env_corr, AZG_factor):
    """
    Updates the columns and data in the main table.
    Applies column toggling, record range filtering, and detector scaling.
    """
    if not visible_cols:
        return [], []
    # Determine number of detectors (2 or 3) based on selector input
    detectors = 2 if detector_tool_value == '2det' else 3

    # Copy the DataFrame for safe modifications
    dff = df.copy()

    # Scale B columns for the detector tool
    dff = apply_detector_scaling(dff, detectors, env_corr, AZG_factor)


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
@app.callback(
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_data_source(contents, filename):
    """
    Handles the file upload. Replaces the DataFrame with the uploaded file.
    """
    if contents is None:
        return ""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Try loading as a text file (CSV/TSV or whitespace delimited)
        df_new = pd.read_csv(io.StringIO(decoded.decode('utf-8')), comment='#', sep=None, engine='python')
        global df
        df = df_new
        return f"Data source '{filename}' loaded successfully! Please adjust display controls if needed."
    except Exception as e:
        return f"Error loading file '{filename}': {e}"

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
    Input('heatmap-shift-mode', 'value')
)
def update_shifted_heatmap(start_rec, end_rec, manual_min, manual_max, rpm_filtered, detector_tool_value, env_corr_value, AZG_factor_value, shift_mode_value):
    '''
    Builds the shifted heatmap/image, where each row's B columns 
    are circularly shifted left by BinRefAtStart.
    '''
    # Determine number of detectors (2 or 3) based on selector input
    detectors = 2 if detector_tool_value == '2det' else 3
    
    # Build the B column list
    b_cols = [f'B{i}' for i in range(1, 25) if f'B{i}' in df.columns]
    dff = df.copy()
    
    # Recompute B columns using detector scaling
    dff = apply_detector_scaling(dff, detectors, env_corr_value, AZG_factor_value)

    
    # Apply RPM filter to the data if needed
    if rpm_filtered and 'RPM' in dff.columns:
        dff = dff[dff['RPM'] >= 30]
        
    if start_rec is None or end_rec is None or start_rec > end_rec:
        return go.Figure()
    df_slice = dff[(dff['RecordNumber'] >= start_rec) & (dff['RecordNumber'] <= end_rec)].copy()
    if df_slice.empty:
        return go.Figure()
    
    # For each row, perform a circular shift by BinRefAtStart
    # Apply shift or leave raw
    shifted_b_matrix = []
    for _, row in df_slice.iterrows():
        b_values = row[b_cols].values
        if shift_mode_value == 'shift':
            shift = int(row['BinRefAtStart']) % 24 if not pd.isnull(row['BinRefAtStart']) else 0
            b_shifted = np.roll(b_values, -shift)
            shifted_b_matrix.append(b_shifted)
        else:
            shifted_b_matrix.append(b_values)
        
    z_data = np.array(shifted_b_matrix, dtype=float)
    y_labels = [
        f"{int(recnum)} | {ts.strftime('%Y-%m-%d %H:%M:%S') if not pd.isnull(ts) else ''}"
        for recnum, ts in zip(df_slice['RecordNumber'], df_slice['TimeStamp'])
    ] if 'RecordNumber' in df_slice.columns and 'TimeStamp' in df_slice.columns else df_slice.index.astype(str)
    x_labels = b_cols
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
    
    # Build the x_labels for the x-axis
    if shift_mode_value == 'shift':
        x_labels = b_cols
    else:
        x_labels = b_cols

        
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
            tickvals=[-0.5, 6.5, 12.5, 18.5, 23.5],
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


# --- Standard Dash main entry point ---
if __name__ == '__main__':
    app.run(debug=True)
