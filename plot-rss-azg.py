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


def load_data(filepath):
    df = pd.read_csv(filepath, comment='#')
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    return df

app = Dash(__name__)

df = load_data("Data_example.txt")

# Ensure TimeStamp is a string for startswith, or use dt.strftime if it's datetime
# Ignore the 2000-01 records 
if pd.api.types.is_datetime64_any_dtype(df['TimeStamp']):
    mask = ~df['TimeStamp'].dt.strftime('%Y-%m').eq('2000-01')
else:
    mask = ~df['TimeStamp'].astype(str).str.startswith('2000-01')

df = df[mask].reset_index(drop=True)

# Add B1 through B24 if not present
for i in range(1, 25):
    col_name = f'B{i}'
    if col_name not in df.columns:
        df[col_name] = np.nan

# Identify all Dwell columns, case-insensitively, to avoid missing one
Dwell_cols = [col for col in df.columns if col.lower().startswith('dwell')]
Dwell_cols = [col for col in Dwell_cols if re.fullmatch(r'Dwell([1-9]|1[0-9]|2[0-4])', col, re.IGNORECASE)]
Dwell_cols = sorted(Dwell_cols, key=lambda x: int(re.findall(r'\d+', x)[0]))
df['TotDwell'] = df[Dwell_cols].sum(axis=1)

# Columns to exclude from display (case-insensitive)
excluded_cols = [
    'RecordStatus',
    'Status',
    'Excursions',
    'Temperature',
    'TotalDetectorCurrent',
    'DetectorVoltage'
]
excluded_cols_set = {x.lower() for x in excluded_cols}

# List the *prefixes* to match columns
hide_prefixes = [
    "Dwell", "D1Bin", "D2Bin", "D3Bin", "B"
]
b_regex = re.compile(r"^B([1-9]|1[0-9]|2[0-4])$", re.IGNORECASE)

# Gather all display columns (after exclusions above)
display_columns = [col for col in df.columns if col.lower() not in excluded_cols_set]

# Find columns to hide by default:
hidden_columns = []
for col in display_columns:
    if any(col.startswith(prefix) for prefix in hide_prefixes) or b_regex.match(col):
        hidden_columns.append(col)
visible_columns = [col for col in display_columns if col not in hidden_columns]
checklist_options = [{'label': col, 'value': col} for col in display_columns]

# Define numeric_columns after creating excluded_cols
numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in excluded_cols]

df['B1'] = (
    np.where(df['Dwell1']  != 0, df['D1Bin1']  / df['Dwell1'],  np.nan) +
    np.where(df['Dwell16'] != 0, df['D2Bin1']  / df['Dwell16'], np.nan) +  
    np.where(df['Dwell8']  != 0, df['D3Bin1']  / df['Dwell8'],  np.nan)    
)
df['B2'] = (
    np.where(df['Dwell2']  != 0, df['D1Bin2']  / df['Dwell2'],  np.nan) +
    np.where(df['Dwell17'] != 0, df['D2Bin2']  / df['Dwell17'], np.nan) +  
    np.where(df['Dwell9'] != 0, df['D3Bin2']  / df['Dwell9'], np.nan)    
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
    np.where(df['Dwell24']  != 0, df['D2Bin9']  / df['Dwell24'],  np.nan) +
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
    np.where(df['Dwell24']  != 0, df['D3Bin17'] / df['Dwell24'],  np.nan)
)
df['B18'] = (
    np.where(df['Dwell18'] != 0, df['D1Bin18'] / df['Dwell18'], np.nan) +
    np.where(df['Dwell9'] != 0, df['D2Bin18'] / df['Dwell9'], np.nan) +
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

#FOR NOW UNTIL CALS 1,2,3 are used
CAL123 = 1

for i in range(1, 25):
    b_col = f'B{i}'
    if b_col in df.columns:
        df[b_col] = df[b_col] * 1000 / CAL123

# Dash Layout with manual scale controls, defaults, and buttons
app.layout = html.Div([
    html.H1("RSS-AZG Data Analysis Tool"),
    
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
        
        html.Span(
            id='max-record-display',
            children=f"(Max Record number for this file is {int(df['RecordNumber'].max())})",
            style={'margin-left': '16px', 'alignSelf': 'center', 'fontWeight': 'bold', 'color': '#00528C'}
        ),

        
    ], style={'margin-bottom': '20px', 'display': 'flex', 'gap': '20px'}),
    
    html.Div([
        html.Label("Min color scale:"),
        dcc.Input(id='manual-min', type='number', value=0, debounce=True, step=0.01, style={'width': '120px'}),
        html.Label("Max color scale:"),
        dcc.Input(id='manual-max', type='number', value=200, debounce=True, step=0.01, style={'width': '120px'}),
        html.Button("Set gamma scale to 0-200", id='set-b-scale', n_clicks=0, style={'margin-left': '20px', 'margin-right': '10px'}),
        html.Button("Auto Color Range (auto-updates with record range change)", id='reset-b-scale', n_clicks=0)
    ], style={'margin-bottom': '20px', 'display': 'flex', 'gap': '20px'}),
    
    dcc.Graph(id='b-heatmap'),
    
    
    dcc.Graph(id='b-heatmap-shifted'),

    
    
    
    html.Div([
    html.Button("Hide data below 30 RPM in heatmap", id='hide-below-30rpm', n_clicks=0, style={'margin-right': '10px'}),
    html.Button("Restore sliding data", id='show-all-rpm', n_clicks=0)
    ], style={'margin-bottom': '20px', 'margin-top': '10px'}),

    html.Div([
        html.H4("Test facility plot generation, for field logs only, NOT for final logs. No guarantee of accuracy.", style={'margin-bottom': '12px'}),
        html.Div([
            html.Label("Start Depth:"),
            dcc.Input(id='start-depth', type='number', value=0, style={'margin-right': '16px', 'width': '100px'}),
            html.Label("End Depth:"),
            dcc.Input(id='end-depth', type='number', value=100, style={'margin-right': '16px', 'width': '100px'}),
            html.Button("Apply", id='apply-depth', n_clicks=0, style={'margin-left': '12px'}),
        ], style={'margin-bottom': '8px'}),
        html.Div(id='depth-apply-status', style={'color': '#00528C'})
    ], style={'border': '2px solid #C4C4C4', 'border-radius': '12px', 'padding': '16px', 'margin': '30px 0'}),

    
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in numeric_columns],
        value=numeric_columns[0] if numeric_columns else None,
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
    "Note that B1...B24 are not pulled from memory directly but instead computed here.",
    style={'margin-bottom': '10px', 'fontStyle': 'italic', 'color': '#555'}
    ),
    
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

dcc.Store(id='rpm-filtered', data=False),


html.Div([
    html.Hr(),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Select Different Data Source Text File'),
        accept='.txt,.csv,.tsv,.dat,text/plain',
        multiple=False,
        style={'margin-top': '40px'}
    ),
    html.Div(id='upload-status', style={'margin-top': '10px', 'color': '#00528C'})
])


])

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
        return 0, 200
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'set-b-scale':
        return 0, 200
    elif button_id == 'reset-b-scale':
        return None, None
    return manual_min, manual_max

@app.callback(
    Output('time-series-chart', 'figure'),
    Input('column-dropdown', 'value'),
    Input('start-record', 'value'),
    Input('end-record', 'value')
)
def update_chart(selected_col, start_rec, end_rec):
    if not selected_col or selected_col not in df.columns:
        return {}
    # Filter to user-specified range (by RecordNumber)
    if start_rec is not None and end_rec is not None:
        df_filtered = df[(df['RecordNumber'] >= start_rec) & (df['RecordNumber'] <= end_rec)]
    else:
        df_filtered = df
    fig = px.line(df_filtered, x='RecordNumber', y=selected_col,
                  title=f"Time Series of {selected_col} (Records {start_rec} to {end_rec})",
                  labels={'RecordNumber': 'Record Number', selected_col: selected_col})
    fig.update_layout(transition_duration=500)
    return fig

@app.callback(
    Output('data-table', 'columns'),
    Output('data-table', 'data'),
    Input('column-toggle', 'value'),
    Input('start-record', 'value'),
    Input('end-record', 'value')
)
def update_table_columns_and_data(visible_cols, start_rec, end_rec):
    if not visible_cols:
        return [], []
    # Filter dataframe to selected record range by RecordNumber
    if start_rec is not None and end_rec is not None:
        df_filtered = df[(df['RecordNumber'] >= start_rec) & (df['RecordNumber'] <= end_rec)]
    else:
        df_filtered = df
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
    data = df_filtered[visible_cols].to_dict('records')
    return cols, data


@app.callback(
    Output('b-heatmap', 'figure'),
    Input('start-record', 'value'),
    Input('end-record', 'value'),
    Input('manual-min', 'value'),
    Input('manual-max', 'value'),
    Input('rpm-filtered', 'data')
)



def update_heatmap(start_rec, end_rec, manual_min, manual_max, rpm_filtered):
    b_cols = [f'B{i}' for i in range(1, 25) if f'B{i}' in df.columns]
    dff = df.copy()
    if rpm_filtered and 'RPM' in dff.columns:
        dff = dff[dff['RPM'] >= 30]
        
    if start_rec is None or end_rec is None or start_rec > end_rec:
        return go.Figure()
    df_slice = dff[(dff['RecordNumber'] >= start_rec) & (dff['RecordNumber'] <= end_rec)]

    
    z_data = df_slice[b_cols].values
    y_labels = df_slice['TimeStamp'].dt.strftime('%Y-%m-%d %H:%M:%S') if 'TimeStamp' in df_slice.columns else df_slice.index.astype(str)
    x_labels = b_cols
    # Exclude zeros from auto color scale computation
    # Use only values in the currently visible heatmap (df_slice)
    all_b_vals = df_slice[b_cols].values.flatten()
    nonzero_vals = all_b_vals[(all_b_vals != 0) & ~np.isnan(all_b_vals)]
    if nonzero_vals.size > 0:
        auto_min_b = np.min(nonzero_vals)
        auto_max_b = np.max(nonzero_vals)
    else:
        auto_min_b, auto_max_b = 0, 1  # fallback for all-zero/NaN
    min_b = manual_min if manual_min is not None else auto_min_b
    max_b = manual_max if manual_max is not None else auto_max_b

    colorscale = [
        [0.0, 'white'],
        [0.25, 'yellow'],
        [0.5, 'orange'],
        [0.75, 'red'],
        [1.0, 'black']
    ]
    
    # Display both RecordNumber and TimeStamp as y-axis labels
    if 'RecordNumber' in df_slice.columns and 'TimeStamp' in df_slice.columns:
        y_labels = [
            f"{int(recnum)} | {ts.strftime('%Y-%m-%d %H:%M:%S') if not pd.isnull(ts) else ''}"
            for recnum, ts in zip(df_slice['RecordNumber'], df_slice['TimeStamp'])
        ]
    elif 'RecordNumber' in df_slice.columns:
        y_labels = [str(int(recnum)) for recnum in df_slice['RecordNumber']]
    elif 'TimeStamp' in df_slice.columns:
        y_labels = df_slice['TimeStamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        y_labels = df_slice.index.astype(str)
    
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
    
    # Overlay RPM curve (scatter), sharing y-axis, with a transparent background.
    if 'RPM' in df_slice.columns:
        fig.add_trace(
            go.Scatter(
                x=df_slice['RPM'],
                y=y_labels,
                mode='lines+markers',
                line=dict(color='blue', width=2),
                name='RPM',
                xaxis='x2',
                yaxis='y',
                marker=dict(opacity=0.1),
                showlegend=True,
            )
        )
        # Add a second x-axis for RPM, at the top, fixed 0–1600.
        fig.update_layout(
            xaxis2=dict(
                title="",
                overlaying='x',
                side='top',
                range=[0, 1600],
                showgrid=True,
                zeroline=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1.0)
        )
        # Set heatmap opacity
        fig.data[0].opacity = 1.0
    
    fig.update_layout(
        title=f"24-Bin Data Image (unshifted) (Records {start_rec} to {end_rec})",
        xaxis_title="Bins 1-24",
        yaxis_title="Record Number and TimeStamp",
        yaxis_autorange='reversed',
        margin=dict(l=60, r=40, t=40, b=60)
    )
    return fig


@app.callback(
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_data_source(contents, filename):
    if contents is None:
        return ""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Try loading as a text file (CSV/TSV or whitespace delimited)
        # You can refine parsing based on your file style
        df_new = pd.read_csv(io.StringIO(decoded.decode('utf-8')), comment='#', sep=None, engine='python')
        global df
        df = df_new
        return f"Data source '{filename}' loaded successfully! Please adjust display controls if needed."
    except Exception as e:
        return f"Error loading file '{filename}': {e}"


@app.callback(
    Output('rpm-filtered', 'data'),
    [Input('hide-below-30rpm', 'n_clicks'),
     Input('show-all-rpm', 'n_clicks')],
    prevent_initial_call=True
)
def set_rpm_filter(hide_clicks, show_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'hide-below-30rpm':
        return True
    elif button_id == 'show-all-rpm':
        return False
    return False

@app.callback(
    Output('max-record-display', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_max_record(contents, filename):
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

@app.callback(
    Output('b-heatmap-shifted', 'figure'),
    Input('start-record', 'value'),
    Input('end-record', 'value'),
    Input('manual-min', 'value'),
    Input('manual-max', 'value'),
    Input('rpm-filtered', 'data')
)
def update_shifted_heatmap(start_rec, end_rec, manual_min, manual_max, rpm_filtered):
    b_cols = [f'B{i}' for i in range(1, 25) if f'B{i}' in df.columns]
    dff = df.copy()
    if rpm_filtered and 'RPM' in dff.columns:
        dff = dff[dff['RPM'] >= 30]
    if start_rec is None or end_rec is None or start_rec > end_rec:
        return go.Figure()
    df_slice = dff[(dff['RecordNumber'] >= start_rec) & (dff['RecordNumber'] <= end_rec)].copy()
    if df_slice.empty:
        return go.Figure()
    
    # Now perform circular shift for each row by BinRefAtStart
    shifted_b_matrix = []
    for _, row in df_slice.iterrows():
        shift = int(row['BinRefAtStart']) % 24 if not pd.isnull(row['BinRefAtStart']) else 0
        b_values = row[b_cols].values
        b_shifted = np.roll(b_values, -shift)
        shifted_b_matrix.append(b_shifted)
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
        auto_min_b, auto_max_b = 0, 1
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
        title=f"Dynamically shifted by BinRefAtStart 24-Bin Image (Records {start_rec} to {end_rec})",
        xaxis_title="",
        yaxis_title="Record Number and TimeStamp",
        yaxis_autorange='reversed',
        margin=dict(l=60, r=40, t=40, b=60)
    )
    
    fig.update_xaxes(
    tickvals=[-0.5, 6.5, 12.5, 18.5, 23.5],
    ticktext=["Up", "Right", "Down", "Left", "Up"],
    range=[-0.5, 23.5]
    )

    return fig


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



if __name__ == '__main__':
    app.run(debug=True)
