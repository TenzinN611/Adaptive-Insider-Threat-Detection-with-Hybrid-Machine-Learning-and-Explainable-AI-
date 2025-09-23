# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import numpy as np
import math
import json
import time
from datetime import datetime, timedelta

# ML and Explanation Libraries
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
import shap
import plotly.express as px
import plotly.graph_objects as go

# Reporting
from fpdf import FPDF
import io

# External Service Libraries (ensure they are installed: pip install elasticsearch google-generativeai)
try:
    from elasticsearch import Elasticsearch
except Exception:
    Elasticsearch = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="SOC Dashboard",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and SIEM-like styling
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #1E1E1E; color: #E0E0E0; }
    .stMetric { background-color: #2C2C2C; border: 1px solid #4A4A4A; border-radius: 8px; padding: 15px; }
    .stDataFrame { border: 1px solid #4A4A4A; }
    .stAlert { background-color: #333333; }
    h1, h2, h3 { color: #FFFFFF; }
    .stButton > button { background-color: #4A90E2; color: white; }
</style>
""", unsafe_allow_html=True)

# --- Default Values ---
RANDOM_STATE = 42
DEFAULT_PCT_THRESHOLD = 99.0
DEFAULT_W1 = 0.70
MIN_USER_SAMPLES_FOR_PER_USER = 30
TOP_N_FEATURES_SHAP = 5
DEFAULT_RULE_WEIGHTS = {
    "wikileaks_flag": 1.0,
    "offhour_usb_flag": 0.7,
    "offhour_http_flag": 0.3,
    "offhour_logon_flag": 0.3,
}

# --- Session State Initialization ---
if 'run_complete' not in st.session_state:
    st.session_state['run_complete'] = False
if 'last_investigation' not in st.session_state:
    st.session_state['last_investigation'] = None

# --- Core Data Processing & ML ---

# Helper functions
def _to_numeric_fill0(df):
    return df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

def _build_X(df, cols):
    return _to_numeric_fill0(df[cols])

def _month_start(ts):
    return ts.to_period('M').to_timestamp()

def _month_end_exclusive(ts):
    return (_month_start(ts) + pd.offsets.MonthBegin(1))

def compute_metrics_from_cm(cm):
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else float('nan')
    pre = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1  = (2 * pre * rec / (pre + rec)) if (pre + rec) > 0 else 0.0
    return {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1, 'tn':tn,'fp':fp,'fn':fn,'tp':tp}

def safe_anomaly_scores_from_clf(clf, X):
    # REMOVED the line that converts X to a NumPy array (X.values)
    # The DataFrame X is now passed directly to the model's methods.
    try:
        # Pass the DataFrame 'X' directly
        scores = clf.decision_function(X)
        return -np.asarray(scores)
    except Exception:
        try:
            # Pass the DataFrame 'X' directly
            scores = clf.score_samples(X)
            return -np.asarray(scores)
        except Exception:
            raise RuntimeError("Could not obtain anomaly scores from classifier.")

def compute_rule_score_row(row, rule_weights):
    score = 0.0
    if int(row.get("wikileaks_flag", 0)) == 1: score += rule_weights["wikileaks_flag"]
    trio = {
        "offhour_usb_flag": int(row.get("offhour_usb_flag", 0)),
        "offhour_http_flag": int(row.get("offhour_http_flag", 0)),
        "offhour_logon_flag": int(row.get("offhour_logon_flag", 0)),
    }
    if trio["offhour_usb_flag"] == 1 or sum(trio.values()) >= 2:
        score += sum(rule_weights[k] for k, v in trio.items() if v == 1)
    return score

def per_user_thresholds_from_benign(clf, X_train_pool_df, train_pool_df, pct, min_samples):
    train_scores = safe_anomaly_scores_from_clf(clf, X_train_pool_df)
    train_pool_df_local = train_pool_df.copy().reset_index(drop=True)
    train_pool_df_local['anomaly_score'] = train_scores
    global_threshold = np.percentile(train_scores, pct)
    pupct = pct / 100.0
    grouped = train_pool_df_local.groupby('user')['anomaly_score']
    counts = grouped.count()
    per_user_thresh = grouped.quantile(pupct)
    per_user_thresh = per_user_thresh.where(counts >= min_samples)
    return per_user_thresh, global_threshold

def predict_using_thresholds(scores_array, test_df_local, per_user_thresh_series, global_threshold):
    tmp = test_df_local.copy().reset_index(drop=True)
    tmp['anomaly_score'] = scores_array
    tmp['anomaly_threshold'] = tmp['user'].map(per_user_thresh_series).fillna(global_threshold)
    tmp['is_model_anomaly'] = (tmp['anomaly_score'] >= tmp['anomaly_threshold']).astype(int)
    return tmp

def _attach_monthly_shap_and_rule(clf, X_train_pool, test_month_scored, feature_cols, rule_weights, w1, score_min_max):
    out_df = test_month_scored.copy().reset_index(drop=True)
    alerts_mask = out_df['is_model_anomaly'].astype(int) == 1
    alerts_df = out_df.loc[alerts_mask].copy()

    if alerts_df.empty:
        out_df['shap_explanation'] = ''
        out_df['rule_score'] = 0.0
        out_df['hybrid_score'] = 0.0
        return out_df

    # Rule Score
    alerts_df['rule_score'] = alerts_df.apply(lambda row: compute_rule_score_row(row, rule_weights), axis=1)
    max_possible_rule = sum(w for w in rule_weights.values() if w > 0) or 1.0
    alerts_df['rule_norm'] = (alerts_df['rule_score'] / max_possible_rule).clip(0.0, 1.0)

    # Hybrid Score
    score_min, score_max = score_min_max
    anomaly_norm = ((alerts_df['anomaly_score'] - score_min) / (score_max - score_min)).clip(0, 1)
    alerts_df['hybrid_score'] = (w1 * anomaly_norm) + ((1.0-w1) * alerts_df['rule_norm'])

    # SHAP
    X_alerts = _build_X(alerts_df, feature_cols)
    explainer = shap.TreeExplainer(clf, X_train_pool.sample(min(len(X_train_pool), 100)))
    shap_values = explainer.shap_values(X_alerts)
    shap_arr = np.asarray(shap_values)

    explanations = []
    for i in range(len(X_alerts)):
        shap_row = shap_arr[i]

        # --- MODIFIED LOGIC STARTS HERE ---
        
        # 1. Get the indices of all features with a positive SHAP value.
        positive_indices = np.where(shap_row > 0)[0]
        
        # 2. If there are positive features, get their corresponding SHAP values.
        if len(positive_indices) > 0:
            positive_shap_values = shap_row[positive_indices]
            
            # 3. Sort these positive features by their magnitude (highest first) and get their original indices.
            sorted_positive_indices = positive_indices[np.argsort(positive_shap_values)[::-1]]
            
            # 4. Take the top N (e.g., top 5) from this sorted list.
            top_idx = sorted_positive_indices[:TOP_N_FEATURES_SHAP]
        else:
            # If no features had a positive impact, there are no top positive features to show.
            top_idx = np.array([], dtype=int)
            
        # --- MODIFIED LOGIC ENDS HERE ---

        exp_list = [{
            "feature": feature_cols[idx],
            "value": round(float(X_alerts.iloc[i, idx]), 4),
            "shap_value": round(float(shap_row[idx]), 4)
        } for idx in top_idx] # No need to reverse; it's already sorted from highest to lowest.
        
        explanations.append(json.dumps(exp_list))

    alerts_df['shap_explanation'] = explanations

    # Merge results back
    ID_COLS = ['user', 'day']
    merge_cols = ID_COLS + ['rule_score', 'hybrid_score', 'shap_explanation']
    out = out_df.merge(alerts_df[merge_cols], on=ID_COLS, how='left')
    out['shap_explanation'] = out['shap_explanation'].fillna('')
    out['rule_score'] = out['rule_score'].fillna(0.0)
    out['hybrid_score'] = out['hybrid_score'].fillna(0.0)
    return out

@st.cache_data
def load_and_prep_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    required = {"user", "day", "is_malicious"}
    if not required.issubset(df.columns):
        return None, f"CSV must contain: {required}", None, None

    df['user'] = df['user'].astype(str)
    df['day'] = pd.to_datetime(df['day']).dt.floor('D')
    df.drop_duplicates(subset=['user', 'day'], inplace=True)

    # Time Split to get test_df for metrics
    available_years = sorted(df['day'].dt.year.unique())
    split_year = None
    for year in available_years:
        split_date = pd.Timestamp(f"{year}-06-30")
        if (df['day'] <= split_date).any() and (df['day'] > split_date).any():
            split_year = year
            break
    if split_year is None:
         return df, "Could not find a valid time-split year.", None, None

    split_date = pd.Timestamp(f"{split_year}-06-30")
    train_df = df[df['day'] <= split_date].copy()
    test_df  = df[df['day'] > split_date].copy()

    return df, None, train_df, test_df

@st.cache_data
def run_full_pipeline(df, pct_threshold, rule_weights, w1):
    """
    Main function to run both holdout and FP-only scenarios.
    Returns metrics and the final ranked alerts dataframe.
    """
    # --- 1. Setup and Split ---
    ID_COLS = ['user', 'day']
    LABEL_COLS = ['is_malicious']
    RULE_FLAG_NAMES = list(DEFAULT_RULE_WEIGHTS.keys())
    for flag in RULE_FLAG_NAMES:
        if flag not in df.columns:
            df[flag] = 0

    per_user_z_cols = [c for c in df.columns if '_user_z' in c]
    if not per_user_z_cols:
        return None, "No per-user feature columns ('_user_z') found.", None

    # Time Split
    available_years = sorted(df['day'].dt.year.unique())
    split_year = None
    for year in available_years:
        split_date = pd.Timestamp(f"{year}-06-30")
        if (df['day'] <= split_date).any() and (df['day'] > split_date).any():
            split_year = year
            break
    if split_year is None:
        return None, "Could not find a valid time-split year.", None
    
    split_date = pd.Timestamp(f"{split_year}-06-30")
    train_df = df[df['day'] <= split_date].copy()
    test_df  = df[df['day'] > split_date].copy()
    total_tp_test = int(test_df['is_malicious'].sum())

    init_train_ben = train_df[train_df['is_malicious'] == 0].copy()
    if init_train_ben.empty:
        return None, "Initial benign training set is empty.", None

    X_init_train_pool = _build_X(init_train_ben, per_user_z_cols)
    train_pool_df_init = init_train_ben[ID_COLS + LABEL_COLS + RULE_FLAG_NAMES].copy().reset_index(drop=True)

    def _init_if_and_pool():
        clf = IsolationForest(n_estimators=500, contamination='auto', max_features=0.5, random_state=42, n_jobs=-1)
        clf.fit(X_init_train_pool)
        return clf, X_init_train_pool.copy(), train_pool_df_init.copy()

    all_month_starts = sorted({_month_start(d) for d in test_df['day']})
    
    # --- 2. Run Holdout (One-Time Train) Scenario ---
    clf_holdout, X_train_pool_holdout, train_pool_df_holdout = _init_if_and_pool()
    cm_holdout = np.array([[0,0],[0,0]], dtype=int)
    for m_start in all_month_starts:
        m_end = _month_end_exclusive(m_start)
        month_mask = (test_df['day'] >= m_start) & (test_df['day'] < m_end)
        test_month_df = test_df.loc[month_mask]
        if test_month_df.empty: continue

        X_test_month = _build_X(test_month_df, per_user_z_cols)
        test_scores = safe_anomaly_scores_from_clf(clf_holdout, X_test_month)
        
        per_user_thresh_series, global_thr = per_user_thresholds_from_benign(
            clf_holdout, X_train_pool_holdout, train_pool_df_holdout, pct=pct_threshold, min_samples=MIN_USER_SAMPLES_FOR_PER_USER
        )
        test_month_scored = predict_using_thresholds(test_scores, test_month_df, per_user_thresh_series, global_thr)
        cm_holdout += confusion_matrix(test_month_scored['is_malicious'], test_month_scored['is_model_anomaly'], labels=[0,1])
    
    metrics_holdout = compute_metrics_from_cm(cm_holdout)
    
    # --- 3. Run Expand FP-Only Scenario (WITH DYNAMIC NORMALIZATION) ---
    clf_fp, X_train_pool_fp, train_pool_df_fp = _init_if_and_pool()
    cm_fp_only = np.array([[0,0],[0,0]], dtype=int)
    all_results_fp = []

    for m_start in all_month_starts:
        m_end = _month_end_exclusive(m_start)
        month_mask = (test_df['day'] >= m_start) & (test_df['day'] < m_end)
        test_month_df = test_df.loc[month_mask].copy()
        if test_month_df.empty: continue

        # Scoring & Thresholding
        X_test_month = _build_X(test_month_df, per_user_z_cols)
        test_scores = safe_anomaly_scores_from_clf(clf_fp, X_test_month)
        per_user_thresh_series, global_thr = per_user_thresholds_from_benign(
            clf_fp, X_train_pool_fp, train_pool_df_fp, pct=pct_threshold, min_samples=MIN_USER_SAMPLES_FOR_PER_USER
        )
        test_month_scored = predict_using_thresholds(test_scores, test_month_df, per_user_thresh_series, global_thr)
        
        # --- MODIFIED LOGIC: Pass dynamic score_min_max to the helper function ---
        # 1. Recalculate min/max anomaly scores based on the CURRENT training pool
        current_train_scores = safe_anomaly_scores_from_clf(clf_fp, X_train_pool_fp)
        score_min_max = (current_train_scores.min(), current_train_scores.max())

        # 2. Attach SHAP, Rules, and Hybrid Score using the newly calculated min/max
        test_month_enriched = _attach_monthly_shap_and_rule(
            clf_fp, X_train_pool_fp, test_month_scored, per_user_z_cols, rule_weights, w1, score_min_max
        )
        
        cm_fp_only += confusion_matrix(test_month_enriched['is_malicious'], test_month_enriched['is_model_anomaly'], labels=[0,1])
        all_results_fp.append(test_month_enriched)

        # Retraining Logic
        mask_FP = (test_month_enriched['is_malicious'] == 0) & (test_month_enriched['is_model_anomaly'] == 1)
        retrain_add_df = test_month_enriched.loc[mask_FP, ID_COLS + LABEL_COLS + RULE_FLAG_NAMES].copy()
        if not retrain_add_df.empty:
            train_pool_df_fp = pd.concat([train_pool_df_fp, retrain_add_df], ignore_index=True)
            train_pool_df_fp.drop_duplicates(subset=ID_COLS, keep='last', inplace=True)
            
            cur_train_merge = train_pool_df_fp.merge(df[ID_COLS + per_user_z_cols], on=ID_COLS, how='left').dropna(subset=per_user_z_cols)
            if not cur_train_merge.empty:
                X_train_pool_fp = _build_X(cur_train_merge, per_user_z_cols)
                train_pool_df_fp = cur_train_merge[ID_COLS + LABEL_COLS + RULE_FLAG_NAMES].copy().reset_index(drop=True)
                clf_fp = IsolationForest(n_estimators=500, contamination='auto', max_features=0.5, random_state=42, n_jobs=-1)
                clf_fp.fit(X_train_pool_fp)

    metrics_fp_only = compute_metrics_from_cm(cm_fp_only)
    final_results_df = pd.concat(all_results_fp, ignore_index=True) if all_results_fp else pd.DataFrame()
    
    # Filter for anomalies and sort by hybrid score for the final output
    final_alerts = final_results_df[final_results_df['is_model_anomaly'] == 1].sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    
    # Also create model-ranked version for comparison
    model_ranked_alerts = final_results_df[final_results_df['is_model_anomaly'] == 1].sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    pipeline_results = {
        "metrics_holdout": metrics_holdout,
        "metrics_fp_only": metrics_fp_only,
        "hybrid_ranked_alerts": final_alerts,
        "model_ranked_alerts": model_ranked_alerts,
        "total_tp_in_test": total_tp_test
    }
    
    return pipeline_results, f"✅ Pipeline complete. Generated {len(final_alerts)} alerts.", None

# --- UI Helper Functions ---
def compute_budget_metrics(ranked_df, percents, total_tp):
    rows = []
    for p in percents:
        k = max(1, int(math.floor(p/100.0 * len(ranked_df))))
        head = ranked_df.head(k)
        tp_in_budget = int(head["is_malicious"].sum())
        precision = (tp_in_budget / k * 100.0) if k > 0 else 0.0
        coverage = (tp_in_budget / total_tp * 100.0) if total_tp > 0 else 0.0
        rows.append({"Budget Top %": p, "Top N Alerts": k, "TPs in Budget": tp_in_budget, "Precision (%)": precision, "Coverage of Total TPs (%)": coverage})
    return pd.DataFrame(rows)

# --- Investigation Functions (Full Versions) ---
@st.cache_resource
def make_es_client():
    """Create an ES client using env vars (mirrors your ingestion script)."""
    if Elasticsearch is None:
        return None
    ES_URL = os.environ.get("ES_URL", "https://localhost:9200")
    ES_USER = os.environ.get("ES_USER", "elastic")
    ES_PASSWORD = os.environ.get("ES_PASSWORD", "V29-CjdrKXy+qDBDms+D")
    ES_CA_CERT = os.environ.get(
    "ES_CA_CERT",
    "C:/Users/namse/Downloads/elasticsearch-9.1.3-windows-x86_64/elasticsearch-9.1.3/config/certs/http_ca.crt"
)
    kwargs = {"hosts": [ES_URL], "request_timeout": 60}
    if ES_USER and ES_PASSWORD:
        # elasticsearch-py v8 uses basic_auth
        kwargs["basic_auth"] = (ES_USER, ES_PASSWORD)
    if ES_CA_CERT:
        kwargs["ca_certs"] = ES_CA_CERT
    try:
        client = Elasticsearch(**kwargs)
        # optional: ping to check connectivity (don't raise)
        try:
            client.ping()
        except Exception:
            # ignore ping errors here; caller will handle lack of results
            pass
        return client
    except Exception as e:
        st.warning(f"Elasticsearch client init failed: {e}")
        return None

@st.cache_data
def es_query_user_day(user, day_iso, indices=None, size=5000):
    """
    Query ES for events for `user` on `day_iso` (YYYY-MM-DD). Returns dict with index->DataFrame.
    Cached per user+day to speed repeated investigations.
    """
    client = make_es_client()
    out = {}
    if client is None:
        # no ES available: return empty dict so caller can fallback to local data if available
        return out

    dt = pd.to_datetime(day_iso).floor("D")
    start = dt.isoformat()
    end = (dt + timedelta(days=1)).isoformat()
    # build index list - default to expected cert indices (adjust if your indices differ)
    idx_list = indices or ["cert-logon", "cert-device", "cert-http", "cert-email", "cert-file", "certp-psychometric", "certl-ldap"]
    for idx in idx_list:
        try:
            q = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user": user}}
                        ],
                        "filter": [
                            {"range": {"date": {"gte": start, "lt": end}}}
                        ]
                    }
                },
                "size": size,
                "sort": [{"date": {"order": "asc"}}]
            }
            # psychometric or LDAP may use different fields; accept imperfect matches, caller handles empty frames
            if idx == "certp-psychometric":
                q = {
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"user_id": user}},
                                {"term": {"user": user}}
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    "size": size,
                    "sort": []
                }
            if idx.startswith("certl-ldap"):
                q = {
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"user": user}},
                                {"term": {"user_id": user}}
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    "size": 200
                }

            # Use client.search; ensure we pass the body and size appropriately for your ES client version
            resp = client.search(index=idx, body=q)
            hits = [h["_source"] for h in resp.get("hits", {}).get("hits", [])]
            if hits:
                # Convert to DataFrame - DataFrames are serializable by st.cache_data
                out[idx] = pd.DataFrame(hits)
        except Exception as e:
            # store error info frame so UI can show troubleshooting details
            out[idx] = pd.DataFrame([{"_es_error": str(e)}])
    return out

def heuristic_summary(events_by_index, user, day):
    """Simple fallback summarizer - counts by index and top activities."""
    parts = []
    total = 0
    for idx, df in events_by_index.items():
        if df is None or df.empty:
            continue
        cnt = len(df)
        total += cnt
        keycols = ["activity", "url", "filename", "to", "from", "pc"]
        found = []
        for c in keycols:
            if c in df.columns:
                vals = df[c].dropna().apply(str).unique()[:3]
                if vals.size > 0:
                    found.append(f"{c}:{', '.join(vals)}")
        parts.append(f"{idx}={cnt} ({'; '.join(found[:2])})")
    if total == 0:
        return f"No indexed events found for user {user} on {day}."
    return f"Found {total} events for user {user} on {day}: " + "; ".join(parts)

import google.generativeai as genai

@st.cache_data
def summarise_events_with_ai(_events_by_index, user, day):
    """Attempt to summarise events using the Gemini API."""
    if genai is None:
        st.error("`google-generativeai` library not installed. Please run `pip install google-generativeai`.")
        return "AI library not available."

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBSZaQotSdsh0Fwc_NwrWyoVlL1O4Cp2rA")
    if not GEMINI_API_KEY:
        st.warning("GEMINI_API_KEY environment variable not found. Skipping AI summary.")
        return "AI summarization failed: API key not configured."
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    context_lines, activity_lines = [], []
    for idx, df in _events_by_index.items():
        if df is None or df.empty or "_es_error" in df.columns:
            continue
        sample = df.head(5).to_dict(orient="records")
        if "ldap" in idx or "psychometric" in idx:
            context_lines.extend(f"{idx}: {json.dumps(r)}" for r in sample)
        else:
            activity_lines.extend(
                f"{idx} | {r.get('date', '')} | {r.get('activity', r.get('url', ''))}" for r in sample
            )

    prompt = f"""You are a SOC analyst assistant. Summarize the following user activity for an investigation. Offhour is from 6pm to 8am. mention time always

# Contextual Information about User '{user}'
{''.join(context_lines) or '[No static data available]'}

# User Activity on {day}
{''.join(activity_lines) or '[No activity logs available]'}

# Your Analysis
Based on the data, provide a concise one-paragraph summary of the events for user {user} on {day}. Then, list the most notable findings as bullet points.
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.warning(f"Gemini API call failed: {e}")
        return f"(AI summarization failed: {e})"

def create_shap_viz(shap_json_str):
    if not shap_json_str or pd.isna(shap_json_str):
        return None
    try:
        shap_data = json.loads(shap_json_str)
        if not shap_data: return None
        
        df = pd.DataFrame(shap_data)
        df['color'] = np.where(df['shap_value'] < 0, '#4A90E2', '#D0021B')
        # Sorting ascending looks better for horizontal bar charts
        df = df.sort_values('shap_value', ascending=True)

        # --- FIX STARTS HERE ---
        # 1. Combine both values into a single text label for each bar.
        #    We use zip() to iterate through both columns at the same time.
        text_labels = [f"Feature Value: {v:.3f}<br>SHAP Value: {s:.3f}" 
                       for v, s in zip(df['value'], df['shap_value'])]

        # 2. Pass the combined labels to the 'text' argument.
        #    The invalid 'marker_color1' and 'text1' arguments are removed.
        fig = go.Figure(go.Bar(
            x=df['shap_value'], y=df['feature'], orientation='h',
            marker_color=df['color'], 
            text=text_labels
        ))
        
        # Optional: Improve text placement and angle
        fig.update_traces(textposition='outside', textangle=0)
        # --- FIX ENDS HERE ---

        fig.update_layout(
            title_text="Feature Contributions to Anomaly Score (SHAP)", template="plotly_dark",
            xaxis_title="SHAP Value", yaxis_title="Feature", margin=dict(l=10, r=10, t=40, b=10), height=300
        )
        return fig
    except (json.JSONDecodeError, TypeError) as e:
        st.warning(f"Could not parse SHAP data: {e}")
        return None

import plotly.graph_objects as go

def create_event_plotly_graph(events_by_index, max_nodes=50, aggregation_window_minutes=5):
    """
    Merges, aggregates, and formats events into an interactive Plotly graph
    with detailed hover data for each node.
    """
    color_map = {
        'cert-logon': '#4A90E2', 'cert-http': '#50E3C2', 'cert-device': '#F5A623',
        'cert-email': '#9013FE', 'cert-file': '#D0021B', 'default': '#777777'
    }
    
    # 1. Gather all events
    all_events = []
    for source, df in events_by_index.items():
        if 'ldap' in source or 'psychometric' in source: continue
        if df is None or df.empty or "_es_error" in df.columns: continue
        for i, row in df.iterrows():
            timestamp = pd.to_datetime(row.get("date", row.get("timestamp")), errors='coerce', utc=True)
            if pd.isna(timestamp): continue
            all_events.append({"timestamp": timestamp, "source": source, "data": row.to_dict()})

    if not all_events: return None
    sorted_events = sorted(all_events, key=lambda x: x['timestamp'])

    # 2. Aggregate similar consecutive events
    if not sorted_events: return None
    aggregated_events = []
    current_group = [sorted_events[0]]
    for i in range(1, len(sorted_events)):
        current_event, last_event_in_group = sorted_events[i], current_group[-1]
        time_diff = (current_event['timestamp'] - last_event_in_group['timestamp']).total_seconds() / 60
        if current_event['source'] == last_event_in_group['source'] and time_diff < aggregation_window_minutes:
            current_group.append(current_event)
        else:
            aggregated_events.append(current_group)
            current_group = [current_event]
    aggregated_events.append(current_group)

    # 3. Create node labels AND hover text from aggregated groups
    final_nodes = []
    for group in aggregated_events:
        first_event, last_event = group[0], group[-1]
        source = first_event['source']
        label_parts = [f"<b>{source.replace('cert-', '')}</b>"]
        hover_parts = []
        
        if len(group) == 1:
            row = first_event['data']
            activity = row.get('activity')
            if activity and pd.notna(activity): label_parts.append(str(activity))
            if source == 'cert-http' and pd.notna(row.get('url')):
                url = row.get('url', ''); domain = url.split('/')[2].replace('www.', '') if '/' in url else url
                label_parts.append(f"to: {domain[:30]}")
            elif source == 'cert-file' and pd.notna(row.get('filename')):
                label_parts.append(row.get('filename'))
            elif source == 'cert-email' and row.get('to'):
                to_field = str(row.get('to', '')); label_parts.append(f"to: {to_field[:30]}")
            
            hover_parts.append("<b>--- Event Details ---</b>")
            for key, val in row.items():
                is_meaningful = False
                if isinstance(val, list):
                    if val: is_meaningful = True
                elif pd.notna(val):
                    is_meaningful = True
                
                if is_meaningful:
                     hover_parts.append(f"<b>{key}:</b> {str(val)[:100]}")
        else:
            label_parts.append(f"({len(group)} events)")
            hover_parts.append("<b>--- Aggregated Event ---</b>")
            hover_parts.append(f"<b>Source:</b> {source.replace('cert-', '')}")
            hover_parts.append(f"<b>Count:</b> {len(group)} events")
            hover_parts.append(f"<b>From:</b> {first_event['timestamp'].strftime('%H:%M:%S UTC')}")
            hover_parts.append(f"<b>To:</b> {last_event['timestamp'].strftime('%H:%M:%S UTC')}")
            
        final_nodes.append({
            "timestamp": first_event['timestamp'],
            "label": "<br>".join(label_parts),
            "hover_text": "<br>".join(hover_parts),
            "source": source
        })

    limited_events = final_nodes[:max_nodes]
    
    # 4. Plotly Graph Generation
    node_x, node_y, node_text, node_hover_text, node_colors, text_positions = [], [], [], [], [], []
    for i, event in enumerate(limited_events):
        y_pos = 0.2 if i % 2 == 0 else -0.2
        text_pos = 'bottom center' if i % 2 == 0 else 'top center'
        
        node_x.append(i); node_y.append(y_pos)
        time_str = event['timestamp'].strftime('%H:%M:%S UTC')
        node_text.append(f"<b>{time_str}</b><br>{event['label']}")
        node_hover_text.append(event['hover_text'])
        node_colors.append(color_map.get(event['source'], color_map['default']))
        text_positions.append(text_pos)

    # --- FIX: Re-added the missing block to define the graph edges ---
    edge_x, edge_y = [], []
    for i in range(len(limited_events) - 1):
        edge_x.extend([node_x[i], node_x[i+1], None])
        edge_y.extend([node_y[i], node_y[i+1], None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        hovertext=node_hover_text,
        hovertemplate='<b>%{text}</b><br><br>%{hovertext}<extra></extra>',
        textposition=text_positions,
        marker=dict(size=35, color=node_colors, line_width=2)
    )

    legend_traces = []
    sources_in_data = sorted(list(set(event['source'] for event in limited_events)))
    for source in sources_in_data:
        legend_traces.append(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color_map.get(source, color_map['default'])),
            name=source.replace('cert-', '').capitalize()
        ))

    fig = go.Figure(data=[edge_trace, node_trace] + legend_traces,
             layout=go.Layout(
                template="plotly_dark", hovermode='closest', showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 0.5])
    ))
    
    for i in range(len(limited_events) - 1):
        fig.add_annotation(
              x=node_x[i+1], y=node_y[i+1], ax=node_x[i], ay=node_y[i],
              xref='x', yref='y', axref='x', ayref='y',
              showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=1, arrowcolor="#888"
        )
    return fig

def run_investigation(alert_row):
    user = str(alert_row["user"])
    day = pd.to_datetime(alert_row["day"]).strftime("%Y-%m-%d")
    with st.spinner(f"Querying Elasticsearch for {user} on {day}..."):
        events = es_query_user_day(user, day)
    with st.spinner("Generating AI summary..."):
        summary = summarise_events_with_ai(events, user, day)
    
    investigation = {
        "user": user, "day": day, "alert_row": alert_row.to_dict(), "events": events, "summary": summary,
        "triggered_rules": [flag for flag in DEFAULT_RULE_WEIGHTS if alert_row.get(flag, 0) == 1],
        "ldap": next((df for name, df in events.items() if 'ldap' in name), None),
        "psychometric": events.get('certp-psychometric')
    }
    st.session_state["last_investigation"] = investigation
    return investigation

def show_investigation_ui(inv):
    if not inv:
        st.info("Click 'Investigate' on an alert to see details here.")
        return

    st.markdown(f"### 🔎 Alert Investigation: {inv['user']} on {inv['day']}")
    st.markdown("---")
    alert = inv.get('alert_row', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Alert Details")
        st.markdown(f"**User:** `{inv.get('user', 'N/A')}`")
        st.markdown(f"**Hybrid Score:** `{alert.get('hybrid_score', 0):.4f}`")
        st.markdown(f"**Triggered Rules:** `{', '.join(inv.get('triggered_rules', [])) or 'None'}`")

    with col2:
        st.subheader("User Info (LDAP)")
        ldap_df = inv.get('ldap')
        if ldap_df is not None and not ldap_df.empty:
            ldap_info = ldap_df.iloc[0]
            # --- ADDED: Display Employee Name ---
            st.markdown(f"**Name:** `{ldap_info.get('employee_name', 'N/A')}`")
            st.markdown(f"**Role:** `{ldap_info.get('role', 'N/A')}`")
            st.markdown(f"**Department:** `{ldap_info.get('department', 'N/A')}`")
            st.markdown(f"**Supervisor:** `{ldap_info.get('supervisor', 'N/A')}`")
        else:
            st.markdown("_No LDAP data found for the prior month._")
        
    with col3:
        st.subheader("User Psychometric")
        psych_df = inv.get('psychometric')
        if psych_df is not None and not psych_df.empty:
            psych_info = psych_df.iloc[0]
            # --- ADDED: Display all 5 psychometric scores ---
            st.markdown(f"**Openness (O):** `{psych_info.get('O', 'N/A')}`")
            st.markdown(f"**Conscientiousness (C):** `{psych_info.get('C', 'N/A')}`")
            st.markdown(f"**Extraversion (E):** `{psych_info.get('E', 'N/A')}`")
            st.markdown(f"**Agreeableness (A):** `{psych_info.get('A', 'N/A')}`")
            st.markdown(f"**Neuroticism (N):** `{psych_info.get('N', 'N/A')}`")
        else:
            st.markdown("_No psychometric data found._")

    st.markdown("---")
    st.subheader("AI Summary")
    st.write(inv.get("summary", "No summary available."))
    
    st.markdown("---")
    st.subheader("Model Explanation (SHAP)")
    shap_fig = create_shap_viz(alert.get('shap_explanation'))
    if shap_fig: st.plotly_chart(shap_fig, use_container_width=True)
    else: st.info("No SHAP explanation available.")

    st.markdown("---")
    # --- Interactive Event Sequence ---
    st.subheader("Event Sequence")
    fig = create_event_plotly_graph(inv['events'])
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time-series events found to generate a sequence graph.")
    
    st.markdown("---")

    # --- Collapsible Data Sections ---
    st.subheader("Evidence & Raw Data")

    with st.expander("Raw Event Logs by Source"):
        # This section now only shows the dynamic event logs
        event_logs = {k: v for k, v in inv["events"].items() if 'ldap' not in k and 'psychometric' not in k}
        if not event_logs:
            st.info("No dynamic event logs found for this investigation.")
        else:
            for idx, df in event_logs.items():
                st.markdown(f"**{idx}** ({len(df)} rows)")
                st.dataframe(df.head(200), use_container_width=True)
                st.markdown("---")

    with st.expander("Full LDAP / Role Information (From Prior Month)"):
        ldap_df = inv.get('ldap')
        if ldap_df is not None and not ldap_df.empty:
            try:
                # Ensure the 'date' column is a proper datetime type for accurate sorting
                ldap_df['date'] = pd.to_datetime(ldap_df['snapshot_date'], errors='coerce')
                
                # Sort the dataframe by date (newest first) and select only the top row
                latest_ldap_record = ldap_df.sort_values(by='date', ascending=False).head(1)
                
                # Display the single, most recent record
                st.dataframe(latest_ldap_record)
                
            except Exception as e:
                st.error(f"Error processing LDAP data for display: {e}")
                st.dataframe(ldap_df) # Show raw data on error
        else:
            st.info("No LDAP snapshot found for this user in the prior month.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload Features CSV", type=["csv"])
    st.markdown("---")
    st.subheader("1. Anomaly Detection")
    pct_threshold = st.slider("Percentile Threshold", 90.0, 99.9, DEFAULT_PCT_THRESHOLD, 0.1, key="pct")
    st.subheader("2. Hybrid Reranking")
    w1 = st.slider("Hybrid Weight (w1 for Anomaly Score)", 0.0, 1.0, DEFAULT_W1, 0.05, key="w1")
    st.caption(f"Hybrid Score = **{w1:.2f}** * Anomaly + **{1-w1:.2f}** * Rule")
    with st.expander("Tune Rule Weights"):
        rule_weights = { flag: st.number_input(flag, value=weight, step=0.1) 
                         for flag, weight in DEFAULT_RULE_WEIGHTS.items() }
    st.markdown("---")
    run_button = st.button("▶️ Run Pipeline", use_container_width=True)

# --- MAIN APP LOGIC ---
if not uploaded_file:
    st.info("Upload a features CSV, adjust controls in the sidebar, and click **Run Pipeline**.")
    st.stop()

df_loaded, error, train_df, test_df = load_and_prep_data(uploaded_file)
if error:
    st.error(error)
    st.stop()

if run_button:
    st.session_state.run_complete = True
    st.cache_data.clear()
    st.cache_resource.clear()

if st.session_state.run_complete:
    with st.spinner("Running monthly retraining pipeline... This may take a minute."):
        pipeline_results, toast_msg, error_msg = run_full_pipeline(df_loaded, pct_threshold, rule_weights, w1)

    if error_msg:
        st.error(error_msg)
        st.stop()
    if toast_msg:
        st.toast(toast_msg)

    # Unpack results
    metrics_holdout, metrics_fp_only = pipeline_results["metrics_holdout"], pipeline_results["metrics_fp_only"]
    hybrid_ranked_df, model_ranked_df = pipeline_results["hybrid_ranked_alerts"], pipeline_results["model_ranked_alerts"]
    total_tp = pipeline_results["total_tp_in_test"]

    # --- UI TABS ---
    st.title("🛡️ SIEM-Style SOC Alert Triage Dashboard")
    st.markdown("##### Monthly Adaptive Anomaly Detection with Rule-Based Reranking")
    tab1, tab2, tab3 = st.tabs(["Overview", "Alert Triage", "Investigations"])

    with tab1:
        st.header("📊 Performance Metrics Comparison")
        st.info("Compares a static model against an adaptive model that retrains monthly on new False Positives.")
        
        st.subheader("Baseline Model (One-Time Train)")
        holdout_alerts, holdout_tp = metrics_holdout.get('tp',0)+metrics_holdout.get('fp',0), metrics_holdout.get('tp',0)
        holdout_coverage = (holdout_tp / total_tp * 100.0) if total_tp > 0 else 0.0
        
        r1c1_h, r1c2_h, r1c3_h = st.columns(3)
        r1c1_h.metric("Total Malicious Events in Test Set", f"{total_tp:,}")
        r1c2_h.metric("Total Model-Detected Alerts", f"{holdout_alerts:,}")
        r1c3_h.metric("TPs Found by Model (Coverage)", f"{holdout_coverage:.2f}%")
        
        st.markdown("")
        test_time_months_str = "Unknown"
        if test_df is not None and not test_df.empty:
            months_span = (test_df['day'].max() - test_df['day'].min()).days / 30.44
            test_time_months_str = f"{months_span:.1f} months"
            
        r2c1_h, r2c2_h, r2c3_h = st.columns(3)
        r2c1_h.metric("Test Time Period", test_time_months_str)
        r2c2_h.metric("Data Points in Test Set", f"{len(test_df):,}" if test_df is not None else "0")
        r2c3_h.metric("Total Users in Dataset", f"{df_loaded['user'].nunique():,}")

        st.markdown("---")

        st.subheader("Adaptive Model (FP-Only Retraining)")
        fp_alerts, fp_tp = metrics_fp_only.get('tp',0)+metrics_fp_only.get('fp',0), metrics_fp_only.get('tp',0)
        fp_coverage = (fp_tp / total_tp * 100.0) if total_tp > 0 else 0.0

        r1c1_fp, r1c2_fp, r1c3_fp = st.columns(3)
        r1c1_fp.metric("Total Malicious Events in Test Set", f"{total_tp:,}")
        r1c2_fp.metric("Total Model-Detected Alerts", f"{fp_alerts:,}")
        r1c3_fp.metric("TPs Found by Model (Coverage)", f"{fp_coverage:.2f}%")

        st.markdown("")
        r2c1_fp, r2c2_fp, r2c3_fp = st.columns(3)
        r2c1_fp.metric("Test Time Period", test_time_months_str)
        r2c2_fp.metric("Data Points in Test Set", f"{len(test_df):,}" if test_df is not None else "0")
        r2c3_fp.metric("Total Users in Dataset", f"{df_loaded['user'].nunique():,}")

    with tab2:
        st.header("🚨 Top 10 Reranked Alerts")
        if not hybrid_ranked_df.empty:
            display_cols = ["user", "day", "hybrid_score", "anomaly_score", "rule_score", "is_malicious"]
            st.dataframe(hybrid_ranked_df[display_cols].head(10).style.format(
                {'hybrid_score':'{:.4f}', 'anomaly_score':'{:.4f}', 'rule_score':'{:.2f}'}
            ), use_container_width=True)
        else: st.info("No alerts to display.")
        
        st.markdown("---")
        st.header("📈 Investigation Budget & Model Comparison")
        st.write("Performance of the **adaptive model** at different investigation budget levels.")
        
        col1_eval, col2_eval = st.columns(2, gap="large")
        with col1_eval:
            st.subheader("Model Score Ranking")
            st.dataframe(compute_budget_metrics(model_ranked_df, [1,2,5,10], total_tp).style.format(
                {'Precision (%)':'{:.2f}', 'Coverage of Total TPs (%)':'{:.2f}'}
            ), use_container_width=True)
        with col2_eval:
            st.subheader("Hybrid Score Reranking")
            st.dataframe(compute_budget_metrics(hybrid_ranked_df, [1,2,5,10], total_tp).style.format(
                {'Precision (%)':'{:.2f}', 'Coverage of Total TPs (%)':'{:.2f}'}
            ), use_container_width=True)

        st.markdown("---")
        st.header("📋 Alert Triage Queue (Hybrid-Ranked)")
        if hybrid_ranked_df.empty:
            st.warning("No alerts generated by the model with the current settings.")
        else:
            num_to_display = st.number_input("Number of alerts to display", 1, len(hybrid_ranked_df), min(10, len(hybrid_ranked_df)), 5)
            for i, alert_row in hybrid_ranked_df.head(num_to_display).iterrows():
                with st.container(border=True):
                    c1, c2, c3 = st.columns([0.4, 0.4, 0.2])
                    c1.markdown(f"**🏅 Rank #{i+1}** | **User:** `{alert_row['user']}` | **Date:** {pd.to_datetime(alert_row['day']).strftime('%Y-%m-%d')}")
                    c2.metric("Hybrid Score", f"{alert_row['hybrid_score']:.4f}")
                    if alert_row['is_malicious']: c3.error("**Malicious**", icon="🔥")
                    else: c3.success("**Benign**", icon="✅")
                    if c3.button("🔎 Investigate", key=f"investigate_{i}"):
                        run_investigation(alert_row)
                        st.toast(f"Investigation ready in 'Investigations' tab.", icon="✅")
                        
    with tab3:
        show_investigation_ui(st.session_state.get("last_investigation"))