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

# --- 1. CẤU HÌNH ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
look_back = 30
model_path = 'bakery_model_local.h5'

# Kiểm tra sự tồn tại của mô hình
model = None
if os.path.exists(model_path):
    try:
        # Sửa lỗi không nạp được hàm 'mse' bằng cách định nghĩa lại trong custom_objects
        model = load_model(model_path, custom_objects={'mse': 'mean_squared_error'})
        print("Mô hình đã được nạp thành công.")
    except Exception as e:
        print(f"Lỗi khi nạp mô hình: {e}")
else:
    print(f"CẢNH BÁO: Không tìm thấy file {model_path}! Hãy chạy file Notebook trước để sinh ra file này.")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

RECIPE = {
    'Bột mì (kg)': 0.2,
    'Đường (kg)': 0.05,
    'Trứng (quả)': 1.0,
    'Sữa (lít)': 0.1
}

# --- 2. GIAO DIỆN ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("AI-Powered Bakery Management System", 
                        className="text-center my-4 text-primary font-weight-bold"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Dữ liệu & Giả lập", className="bg-primary text-white"),
                dbc.CardBody([
                    html.Label("1. Tải lịch sử bán hàng (CSV):"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Kéo thả hoặc ', html.A('Chọn file')]),
                        style={'width': '100%', 'height': '50px', 'lineHeight': '50px', 'borderWidth': '1px',
                               'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '20px'}
                    ),
                    
                    html.Label("2. Giả lập kịch bản Giảm giá (%):"),
                    dcc.Slider(0, 50, 5, value=0, id='discount-slider', marks={i: f'{i}%' for i in range(0, 51, 10)}),
                    
                    html.Hr(),
                    html.Label("3. Tồn kho hiện tại:"),
                    dbc.Input(id='stock-flour', placeholder='Bột mì (kg)...', type='number', value=50, className="mb-2"),
                    dbc.Input(id='stock-eggs', placeholder='Trứng (quả)...', type='number', value=200, className="mb-2"),
                ])
            ], className="shadow mb-4"),
            
            html.Div(id='inventory-alerts')
        ], width=12, lg=3),
        
        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Dự báo đơn hàng ngày mai", className="card-subtitle text-muted"),
                        html.H2(id="pred-sales", className="text-success font-weight-bold"),
                    ])
                ], className="text-center shadow-sm border-success"), width=6),
                
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Trạng thái Mô hình", className="card-subtitle text-muted"),
                        html.H2("Active" if model else "OFFLINE", 
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
                dbc.CardHeader("Nguyên vật liệu cần chuẩn bị", className="bg-info text-white"),
                dbc.CardBody(id='ingredient-table')
            ], className="shadow")
        ], width=12, lg=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Phân tích chu kỳ (Average Sales by Day)", className="bg-dark text-white"),
                dbc.CardBody([dcc.Graph(id='heatmap-graph')])
            ], className="shadow my-4")
        ], width=12)
    ])
], fluid=True)

# --- 3. LOGIC ---
@app.callback(
    [Output('main-graph', 'figure'), 
     Output('pred-sales', 'children'),
     Output('ingredient-table', 'children'),
     Output('inventory-alerts', 'children'),
     Output('heatmap-graph', 'figure')],
    [Input('upload-data', 'contents'),
     Input('discount-slider', 'value'),
     Input('stock-flour', 'value'),
     Input('stock-eggs', 'value')],
    [State('upload-data', 'filename')]
)
def update_system(contents, discount, stock_flour, stock_eggs, filename):
    if not contents:
        return go.Figure(), "---", "Vui lòng upload dữ liệu", "", go.Figure()
    
    # 3.1 Load & Process Data
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    df[date_col] = pd.to_datetime(df[date_col])
    daily = df.groupby(date_col).size().reset_index(name='Sales').set_index(date_col)
    
    # 3.2 Advanced Feature Engineering (Phải khớp với Notebook)
    daily_feat = daily.copy()
    daily_feat['DayOfWeek'] = daily_feat.index.dayofweek
    daily_feat['Month'] = daily_feat.index.month
    
    daily_feat['day_sin'] = np.sin(2 * np.pi * daily_feat['DayOfWeek'] / 7)
    daily_feat['day_cos'] = np.cos(2 * np.pi * daily_feat['DayOfWeek'] / 7)
    daily_feat['month_sin'] = np.sin(2 * np.pi * daily_feat['Month'] / 12)
    daily_feat['month_cos'] = np.cos(2 * np.pi * daily_feat['Month'] / 12)
    daily_feat['rolling_mean_7'] = daily_feat['Sales'].shift(1).rolling(window=7).mean()
    daily_feat.dropna(inplace=True)
    
    # Các cột dùng cho model: Sales, day_sin, day_cos, month_sin, month_cos, rolling_mean_7
    features = ['Sales', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'rolling_mean_7']
    model_data = daily_feat[features]
    
    # 3.3 Prediction
    final_pred = 0
    if model is not None:
        scaler_X.fit(model_data.values)
        scaler_y.fit(model_data[['Sales']].values)
        
        last_30 = scaler_X.transform(model_data.values[-look_back:]).reshape(1, look_back, len(features))
        pred_scaled = model.predict(last_30)
        base_pred = scaler_y.inverse_transform(pred_scaled)[0,0]
        
        boost_factor = 1 + (discount / 100) * 0.5
        final_pred = base_pred * boost_factor
    
    # 3.4 Ingredients & Alerts
    ing_list = []
    alerts = []
    if final_pred > 0:
        for ing, amount in RECIPE.items():
            needed = final_pred * amount
            ing_list.append(html.P(f"• {ing}: {needed:.2f}"))
            if 'Bột mì' in ing and stock_flour < needed:
                alerts.append(dbc.Alert(f"CẢNH BÁO: Thiếu Bột mì!", color="danger"))
            if 'Trứng' in ing and stock_eggs < needed:
                alerts.append(dbc.Alert(f"CẢNH BÁO: Thiếu Trứng!", color="warning"))
    else:
        ing_list = "Mô hình chưa sẵn sàng."

    # 3.5 Graphs
    heatmap_data = daily.copy()
    heatmap_data['DayName'] = heatmap_data.index.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig_heatmap = px.bar(heatmap_data.groupby('DayName')['Sales'].mean().reindex(order).reset_index(), 
                         x='DayName', y='Sales', color='Sales', template="plotly_white")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily['Sales'], name='Thực tế', opacity=0.3))
    fig.add_trace(go.Scatter(x=daily.index, y=daily['Sales'].rolling(7).mean(), name='Xu hướng', line=dict(color='red')))
    if final_pred > 0:
        future_date = daily.index[-1] + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(x=[future_date], y=[final_pred], name='Dự báo', marker=dict(size=15, color='green', symbol='star')))
    
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=30, b=10), height=400)

    return fig, f"{final_pred:.1f}" if final_pred > 0 else "N/A", ing_list, alerts, fig_heatmap

if __name__ == '__main__':
    app.run(debug=True, port=8050)
