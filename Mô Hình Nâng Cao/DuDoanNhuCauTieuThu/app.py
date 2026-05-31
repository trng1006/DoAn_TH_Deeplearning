import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import io, base64, os

# --- 1. CONFIGURATION ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
look_back = 24  # Matching the training sequence length
model_path = 'pjme_lstm_model.h5'

# Load model
model = None
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"WARNING: {model_path} not found!")

scaler = MinMaxScaler(feature_range=(0, 1))

# --- 2. UI LAYOUT ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Hệ Thống Dự Báo Tiêu Thụ Điện Năng AI", 
                        className="text-center my-4 text-primary font-weight-bold"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Bảng Điều Khiển Dữ Liệu", className="bg-primary text-white"),
                dbc.CardBody([
                    html.Label("1. Tải lên dữ liệu tiêu thụ điện (CSV):"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Kéo thả hoặc ', html.A('Chọn File')]),
                        style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                               'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '20px'}
                    ),
                    
                    html.Label("2. Biên độ nhạy dự báo (%):"),
                    dcc.Slider(0, 20, 1, value=0, id='buffer-slider', marks={i: f'{i}%' for i in range(0, 21, 5)}),
                    
                    html.Hr(),
                    html.Label("3. Ngưỡng cảnh báo hệ thống:"),
                    dbc.Input(id='peak-threshold', placeholder='Ngưỡng tải đỉnh (MW)...', type='number', value=40000, className="mb-2"),
                    html.Small("Hệ thống sẽ cảnh báo nếu dự báo vượt ngưỡng này.", className="text-muted")
                ])
            ], className="shadow mb-4"),
            
            html.Div(id='load-alerts')
        ], width=12, lg=3),
        
        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Dự báo phụ tải giờ tới", className="card-subtitle text-muted"),
                        html.H2(id="pred-load", className="text-success font-weight-bold"),
                    ])
                ], className="text-center shadow-sm border-success"), width=6),
                
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Trạng thái Mô hình", className="card-subtitle text-muted"),
                        html.H2("ĐANG CHẠY" if model else "NGOẠI TUYẾN", 
                                id="model-status",
                                className="text-info font-weight-bold" if model else "text-danger"),
                    ])
                ], className="text-center shadow-sm"), width=6),
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='main-graph')
                ])
            ], className="shadow mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Phân Tích Tiêu Thụ Điện Năng", className="bg-info text-white"),
                dbc.CardBody(id='analysis-summary')
            ], className="shadow")
        ], width=12, lg=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Phân Bổ Phụ Tải Theo Thời Gian (Trung bình theo Giờ)", className="bg-dark text-white"),
                dbc.CardBody([dcc.Graph(id='hourly-graph')])
            ], className="shadow my-4")
        ], width=12)
    ])
], fluid=True)

# --- 3. LOGIC ---
@app.callback(
    [Output('main-graph', 'figure'), 
     Output('pred-load', 'children'),
     Output('analysis-summary', 'children'),
     Output('load-alerts', 'children'),
     Output('hourly-graph', 'figure')],
    [Input('upload-data', 'contents'),
     Input('buffer-slider', 'value'),
     Input('peak-threshold', 'value')],
    [State('upload-data', 'filename')]
)
def update_system(contents, buffer, threshold, filename):
    # Initialize default outputs
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_white")
    
    if not contents:
        return empty_fig, "--- MW", "Vui lòng tải lên file dữ liệu (CSV) để bắt đầu.", "", empty_fig
    
    try:
        # 3.1 Load & Process Data
        print(f"Processing file: {filename}")
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df.columns = [c.strip() for c in df.columns]
        
        # Robust column detection
        date_candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        date_col = date_candidates[0] if date_candidates else df.columns[0]
        
        val_candidates = [c for c in df.columns if '_MW' in c or 'load' in c.lower() or 'consumption' in c.lower()]
        if not val_candidates:
            # Fallback to the second column if no name matches
            val_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        else:
            val_col = val_candidates[0]
            
        print(f"Using columns: Date='{date_col}', Value='{val_col}'")
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.dropna(subset=[val_col])
        df = df.sort_values(by=date_col)
        
        # 3.2 Prediction
        final_pred = 0
        if model is not None:
            if len(df) >= look_back:
                # Feature Engineering to match training
                df['Hour'] = df[date_col].dt.hour
                df['Month'] = df[date_col].dt.month
                df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 23)
                df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 23)
                df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
                
                features_list = [val_col, 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
                
                # Convert to numeric and drop NaNs
                for col in features_list:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                data_values = df[features_list].dropna().values
                
                if len(data_values) >= look_back:
                    # Scaling - Note: Ideally, use a pre-fitted scaler from training
                    scaled_data = scaler.fit_transform(data_values)
                    
                    # Prepare last window: (1, look_back, 5)
                    last_window = scaled_data[-look_back:].reshape(1, look_back, 5)
                    pred_scaled = model.predict(last_window, verbose=0)
                    
                    # Inverse transform
                    # To inverse transform, we need a dummy array with 5 columns
                    dummy = np.zeros((1, 5))
                    dummy[0, 0] = pred_scaled[0, 0]
                    base_pred = scaler.inverse_transform(dummy)[0, 0]
                    
                    # Apply buffer factor
                    buffer_val = buffer if buffer is not None else 0
                    final_pred = base_pred * (1 + buffer_val / 100)
                else:
                    return empty_fig, "Lỗi dữ liệu", "Dữ liệu không đủ giá trị số (cần 24h).", "", empty_fig
            else:
                return empty_fig, "Thiếu dữ liệu", f"Cần ít nhất {look_back} dòng dữ liệu để dự báo.", "", empty_fig
        else:
            print("Model is not loaded.")
        
        # 3.3 Analysis & Alerts
        analysis = []
        alerts = []
        safe_threshold = threshold if threshold is not None else 40000
        
        if final_pred > 0:
            avg_load = df[val_col].mean()
            max_load = df[val_col].max()
            analysis.append(html.P(f"• Phụ tải trung bình: {avg_load:.2f} MW"))
            analysis.append(html.P(f"• Phụ tải cao nhất (Max): {max_load:.2f} MW"))
            analysis.append(html.P(f"• Dự báo thay đổi: {((final_pred - avg_load)/avg_load)*100:.1f}% so với trung bình"))
            
            if final_pred > safe_threshold:
                alerts.append(dbc.Alert(f"NGUY HIỂM: Dự báo ({final_pred:.0f} MW) vượt ngưỡng cho phép!", color="danger"))
            elif final_pred > safe_threshold * 0.9:
                alerts.append(dbc.Alert(f"CẢNH BÁO: Dự báo đang tiến gần ngưỡng quá tải.", color="warning"))
        else:
            analysis = "Không thể thực hiện dự báo với dữ liệu này."

        # 3.4 Graphs
        # Hourly Distribution
        df['Hour'] = df[date_col].dt.hour
        hourly_avg = df.groupby('Hour')[val_col].mean().reset_index()
        fig_hourly = px.line(hourly_avg, x='Hour', y=val_col, title="Phụ tải trung bình theo Giờ trong ngày", markers=True)
        fig_hourly.update_layout(template="plotly_white")

        # Main Graph (Recent History + Prediction)
        recent_df = df.tail(168) # Last week
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent_df[date_col], y=recent_df[val_col], name='Thực tế', line=dict(color='#2c3e50')))
        
        if final_pred > 0:
            future_date = df[date_col].iloc[-1] + pd.Timedelta(hours=1)
            fig.add_trace(go.Scatter(x=[future_date], y=[final_pred], name='AI Dự báo', 
                                     marker=dict(size=12, color='#e74c3c', symbol='star')))
        
        fig.update_layout(title=f"Xu hướng phụ tải gần đây & Dự báo AI ({filename})", 
                          template="plotly_white", 
                          margin=dict(l=10, r=10, t=40, b=10), height=400)

        return fig, f"{final_pred:.0f} MW" if final_pred > 0 else "N/A", analysis, alerts, fig_hourly

    except Exception as e:
        print(f"Critical error in callback: {e}")
        import traceback
        traceback.print_exc()
        return empty_fig, "Lỗi hệ thống", f"Đã xảy ra lỗi: {str(e)}", [dbc.Alert(f"Lỗi: {str(e)}", color="danger")], empty_fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)
