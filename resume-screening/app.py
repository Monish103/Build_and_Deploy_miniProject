import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import base64, os
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# ------------------------------#
#   Load Semantic Model
# ------------------------------#
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(resume_text, job_desc):
    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(job_desc, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item() * 100
    return round(score, 2)


# ------------------------------#
#   App & UI Configuration
# ------------------------------#
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "AI Resume Screening System"

UPLOAD_DIR = "./data/resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CARD_STYLE = {
    "padding": "28px",
    "boxShadow": "0px 3px 12px rgba(0,0,0,0.15)",
    "borderRadius": "14px",
    "backgroundColor": "#FFFFFF",
    "marginBottom": "25px",
    "transition": "0.3s"
}

app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="AI Resume Screening Suite",
        color="primary",
        dark=True,
        className="mb-4 p-3 shadow-sm"
    ),

    dbc.Row([
        # Left Input Section
        dbc.Col([
            dbc.Card([
                html.H5("Upload Candidate Resumes", className="fw-bold"),
                html.P("Supported format: PDF", style={"color": "#6c757d"}),
                dcc.Upload(
                    id='upload-resume',
                    children=html.Div([
                        html.I(className="bi bi-file-earmark-arrow-up", style={"fontSize": "28px"}),
                        html.Br(),
                        html.Span("Drag & Drop or Click to Upload", className="fw-semibold")
                    ]),
                    multiple=True,
                    style={
                        'width': '100%', 'height': '140px',
                        'borderWidth': '2px', 'borderStyle': 'dashed',
                        'borderRadius': '12px', 'textAlign': 'center',
                        'paddingTop': '35px', 'cursor': 'pointer',
                        'background': '#f8f9fa'
                    }
                )
            ], style=CARD_STYLE),

            dbc.Card([
                html.H5("Paste Job Description", className="fw-bold"),
                dcc.Textarea(
                    id='job-desc',
                    placeholder='Enter key responsibilities, required skills, experience…',
                    style={
                        'width': '100%', 'height': 220,
                        'borderRadius': '10px', 'border': '1px solid #D0D7DE',
                        'padding': '12px'
                    }
                )
            ], style=CARD_STYLE),

            html.Div(className="text-center", children=[
                dbc.Button(
                    "Analyze",
                    id='analyze-btn',
                    n_clicks=0,
                    size="lg",
                    color="success",
                    className="fw-bold px-4 py-2 shadow-sm"
                )
            ]),
        ], width=5),

        # Right Output Section
        dbc.Col([
            dcc.Loading(
                id="loading-results",
                type="circle",
                color="#007bff",
                children=html.Div(id="results")
            )
        ], width=7),
    ]),

    html.Br()
], fluid=True)


# ------------------------------#
#   PDF Text Extraction Helper
# ------------------------------#
def extract_text_from_pdf(file_path):
    try:
        import pdfplumber
    except:
        raise Exception("Install pdfplumber: pip install pdfplumber")

    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()


# ------------------------------#
#   Callback
# ------------------------------#
@app.callback(
    Output('results', 'children'),
    Input('analyze-btn', 'n_clicks'),
    State('upload-resume', 'contents'),
    State('upload-resume', 'filename'),
    State('job-desc', 'value')
)
def analyze_resumes(n_clicks, contents, filenames, job_desc):
    if n_clicks == 0:
        return ""

    if not contents or not job_desc:
        return dbc.Alert("Please upload resumes and provide a job description.",
                         color="danger", className="fw-bold text-center")

    results = []
    for content, filename in zip(contents, filenames):
        try:
            _, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            file_path = os.path.join(UPLOAD_DIR, filename)

            with open(file_path, "wb") as f:
                f.write(decoded)

            resume_text = extract_text_from_pdf(file_path)
            score = compute_similarity(resume_text, job_desc)

            recommendation = (
                "Strong Fit ✅" if score >= 75 else
                "Moderate Fit ⚠️" if score >= 50 else
                "Poor Fit ❌"
            )

            results.append({"Resume": filename,
                            "Match Score (%)": score,
                            "Recommendation": recommendation})
        except Exception as e:
            results.append({"Resume": filename,
                            "Match Score (%)": 0,
                            "Recommendation": f"Error: {e}"})

    df = pd.DataFrame(results).sort_values("Match Score (%)", ascending=False)

    top_score = df.iloc[0]["Match Score (%)"]
    count = len(df)

    return [
        dbc.Card([
            dbc.Row([
                dbc.Col(html.H4(f"Top Match: {top_score}%", className="text-success fw-bold"), width=6),
                dbc.Col(html.H5(f"Total Candidates: {count}", className="fw-semibold text-secondary"), width=6)
            ])
        ], className="p-3 mb-3 border-0 shadow-sm rounded"),

        dash_table.DataTable(
            df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            style_header={'backgroundColor': '#f1f3f5', 'fontWeight': 'bold'},
            style_cell={'padding': '12px', 'textAlign': 'center', 'fontSize': '15px'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Match Score (%)} >= 75'},
                    'backgroundColor': '#d4edda', 'color': 'black'
                },
                {
                    'if': {'filter_query': '{Match Score (%)} >= 50 && {Match Score (%)} < 75'},
                    'backgroundColor': '#fff3cd', 'color': 'black'
                },
                {
                    'if': {'filter_query': '{Match Score (%)} < 50'},
                    'backgroundColor': '#f8d7da', 'color': 'black'
                }
            ],
            page_size=5,
            sort_action="native"
        )
    ]


# ------------------------------#
# Run App
# ------------------------------#
if __name__ == "__main__": 
    app.run(debug=False, host="0.0.0.0", port=8050)



fine tune and hyperparameter tune
