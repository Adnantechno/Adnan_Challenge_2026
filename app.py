"""
=============================================================================
  SENTINEL — Web Attack Detection Console
  CIC-IDS-2017 | Semi-Supervised IDS (XGBoost + Isolation Forest + SHAP)
=============================================================================
HOW TO RUN
----------
1. Put these files in one folder:
     app.py
     model_xgboost.joblib
     model_isoforest.joblib
     scaler.joblib
     feature_names.joblib
   (the four .joblib files are produced by the Colab notebook, Step 14)

2. Install dependencies:
     pip install streamlit xgboost scikit-learn imbalanced-learn shap joblib pandas numpy matplotlib

3. Launch:
     streamlit run app.py

   The console opens in your browser at http://localhost:8501
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Sentinel — Web Attack Detection",
                   page_icon="◆", layout="wide")

# ----------------------------------------------------------------------------
# DESIGN SYSTEM  (security-console aesthetic: deep slate + signal amber/red/green)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp { background: #0d1117; color: #c9d1d9; }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:#e6edf3 !important; letter-spacing:-0.3px;}
p, label, .stMarkdown { font-family:'IBM Plex Mono',monospace; }

.masthead {
  border:1px solid #21262d; border-left:3px solid #d29922;
  background:linear-gradient(180deg,#161b22,#0d1117);
  padding:22px 26px; border-radius:6px; margin-bottom:8px;
}
.masthead .eyebrow{font-family:'IBM Plex Mono';color:#d29922;font-size:11px;
  letter-spacing:3px;text-transform:uppercase;}
.masthead h1{margin:6px 0 2px 0;font-size:30px;}
.masthead .sub{color:#8b949e;font-size:13px;}

.verdict{border-radius:8px;padding:26px 28px;margin:10px 0;border:1px solid;}
.verdict h2{margin:0;font-size:26px;}
.v-attack{background:rgba(248,81,73,.08);border-color:#f85149;}
.v-attack h2{color:#ff7b72 !important;}
.v-clean{background:rgba(63,185,80,.08);border-color:#3fb950;}
.v-clean h2{color:#56d364 !important;}

.metric-card{background:#161b22;border:1px solid #21262d;border-radius:6px;
  padding:16px 18px;text-align:center;}
.metric-card .num{font-family:'Space Grotesk';font-size:26px;font-weight:700;color:#e6edf3;}
.metric-card .lbl{font-family:'IBM Plex Mono';font-size:11px;color:#8b949e;
  text-transform:uppercase;letter-spacing:1px;}

.stButton>button{background:#d29922;color:#0d1117;border:none;border-radius:5px;
  font-family:'Space Grotesk';font-weight:700;letter-spacing:.5px;padding:10px 22px;}
.stButton>button:hover{background:#e3b341;color:#0d1117;}
section[data-testid="stSidebar"]{background:#0b0f14;border-right:1px solid #21262d;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# MASTHEAD
# ----------------------------------------------------------------------------
st.markdown("""
<div class="masthead">
  <div class="eyebrow">CIC-IDS-2017 · Semi-Supervised IDS</div>
  <h1>◆ SENTINEL — Web Attack Detection Console</h1>
  <div class="sub">XGBoost (supervised) + Isolation Forest (anomaly) · explained with SHAP</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# LOAD MODELS (cached so it loads once)
# ----------------------------------------------------------------------------
import os

@st.cache_resource
def load_artifacts():
    # --- XGBoost: prefer the portable .json format (version-independent) ---
    # Falls back to the old .joblib if that's what you have.
    from xgboost import XGBClassifier
    if os.path.exists('model_xgboost.json'):
        xgb = XGBClassifier()
        xgb.load_model('model_xgboost.json')
    else:
        xgb = joblib.load('model_xgboost.joblib')

    iso    = joblib.load('model_isoforest.joblib')
    scaler = joblib.load('scaler.joblib')
    feats  = joblib.load('feature_names.joblib')
    return xgb, iso, scaler, feats

try:
    xgb, iso, scaler, feature_names = load_artifacts()
    models_ready = True
except Exception as e:
    models_ready = False
    # Show the REAL error so problems are easy to diagnose
    st.error("Could not load the models. Details below.")
    st.exception(e)
    st.info("Common cause: version mismatch. In your Anaconda Prompt run:\n\n"
            "pip install --upgrade xgboost scikit-learn\n\n"
            "Or re-save XGBoost in Colab with xgb.save_model('model_xgboost.json') "
            "and place that file next to app.py.")
    st.stop()

# --- Optional CRF sequence model (loaded only if the files are present) ---
@st.cache_resource
def load_crf():
    import os
    if os.path.exists('model_crf.joblib') and os.path.exists('crf_top_idx.joblib'):
        crf   = joblib.load('model_crf.joblib')
        binner= joblib.load('crf_binner.joblib')
        topix = joblib.load('crf_top_idx.joblib')
        # the CRF's own scaler, if the notebook saved it (best accuracy)
        cscaler = joblib.load('crf_scaler.joblib') if os.path.exists('crf_scaler.joblib') else None
        return crf, binner, topix, cscaler
    return None, None, None, None

crf_model, crf_binner, crf_topix, crf_scaler = load_crf()
crf_ready = crf_model is not None

# ----------------------------------------------------------------------------
# PREDICTION FUNCTION  (the core detection logic — mirrors the notebook)
# ----------------------------------------------------------------------------
def detect(row_df):
    """row_df: single-row DataFrame with the feature columns."""
    X = row_df[feature_names].values.astype(float)
    Xs = scaler.transform(X)

    # Supervised: XGBoost
    xgb_pred  = int(xgb.predict(Xs)[0])
    xgb_proba = float(xgb.predict_proba(Xs)[0, 1])

    # Anomaly: Isolation Forest (+1 normal, -1 anomaly)
    iso_pred  = 1 if iso.predict(Xs)[0] == -1 else 0

    # Combined semi-supervised rule (OR)
    combined  = 1 if (xgb_pred == 1 or iso_pred == 1) else 0
    return xgb_pred, xgb_proba, iso_pred, combined, Xs

# ----------------------------------------------------------------------------
# SIDEBAR — how it works
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### How detection works")
    st.markdown("""
**Stage 1 — XGBoost**
Supervised model trained on labeled web attacks. High precision on known patterns.

**Stage 2 — Isolation Forest**
Unsupervised anomaly detector. Flags flows that don't look like normal traffic — a
safety net for novel attacks.

**Combined = semi-supervised.**
A flow is flagged if *either* stage flags it (maximizes recall).
""")
    st.markdown("---")
    st.markdown("### Detection threshold")
    thresh = st.slider("XGBoost attack probability", 0.0, 1.0, 0.50, 0.05,
                       help="Lower = catch more attacks (higher recall, more false alarms)")

# ----------------------------------------------------------------------------
# TABS: single-flow analysis  |  batch CSV  |  about
# ----------------------------------------------------------------------------
tab1, tab2, tab4, tab3 = st.tabs(["◈ Analyze a flow", "◈ Batch scan (CSV)",
                                  "◈ Sequence scan (CRF)", "◈ About"])

# ---------------- TAB 1: single flow ----------------
with tab1:
    st.markdown("#### Upload a single flow to analyze")
    st.caption("Upload a one-row CSV (a single network flow with the CICFlowMeter "
               "feature columns). The console runs both detectors and shows a verdict.")

    single = st.file_uploader("Single-flow CSV", type=['csv'], key='single')

    if single is not None:
        row = pd.read_csv(single)
        row.columns = row.columns.str.strip()
        missing = [f for f in feature_names if f not in row.columns]
        if missing:
            st.error(f"CSV is missing {len(missing)} required feature columns "
                     f"(e.g. {missing[:3]}).")
        else:
            xgb_pred, xgb_proba, iso_pred, combined, Xs = detect(row.iloc[[0]])
            attack = (xgb_proba >= thresh) or (iso_pred == 1)

            if attack:
                st.markdown('<div class="verdict v-attack"><h2>⚠ WEB ATTACK DETECTED</h2>'
                            '<p>At least one detector flagged this flow as malicious.</p></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict v-clean"><h2>✓ BENIGN — no threat</h2>'
                            '<p>Both detectors consider this flow normal.</p></div>',
                            unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><div class="num">{xgb_proba*100:.1f}%</div>'
                        f'<div class="lbl">XGBoost attack prob.</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="num">'
                        f'{"ANOMALY" if iso_pred else "NORMAL"}</div>'
                        f'<div class="lbl">Isolation Forest</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="num">'
                        f'{"ATTACK" if attack else "BENIGN"}</div>'
                        f'<div class="lbl">Combined verdict</div></div>', unsafe_allow_html=True)

            # SHAP local explanation
            st.markdown("#### Why this verdict? (SHAP)")
            try:
                import shap
                explainer = shap.TreeExplainer(xgb)
                sv = explainer.shap_values(Xs)
                contrib = pd.DataFrame({'feature': feature_names,
                                        'impact': sv[0]})
                contrib['abs'] = contrib['impact'].abs()
                top = contrib.sort_values('abs', ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#f85149' if v > 0 else '#3fb950' for v in top['impact']]
                ax.barh(top['feature'][::-1], top['impact'][::-1], color=colors[::-1])
                ax.set_xlabel('← benign      SHAP impact      attack →')
                ax.set_title('Top features driving this decision')
                fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
                ax.tick_params(colors='#c9d1d9'); ax.xaxis.label.set_color('#c9d1d9')
                ax.title.set_color('#e6edf3')
                for s in ax.spines.values(): s.set_color('#21262d')
                st.pyplot(fig)
                st.caption("Red pushes toward ATTACK, green toward BENIGN.")
            except Exception as ex:
                st.info(f"SHAP explanation unavailable: {ex}")

# ---------------- TAB 2: batch ----------------
with tab2:
    st.markdown("#### Scan many flows at once")
    st.caption("Upload a CSV with multiple flows. The console classifies each row and "
               "summarizes how many web attacks were found.")
    batch = st.file_uploader("Batch CSV", type=['csv'], key='batch')

    if batch is not None:
        data = pd.read_csv(batch)
        data.columns = data.columns.str.strip()
        missing = [f for f in feature_names if f not in data.columns]
        if missing:
            st.error(f"CSV missing {len(missing)} feature columns.")
        else:
            data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_names)
            Xs = scaler.transform(data[feature_names].values.astype(float))
            proba = xgb.predict_proba(Xs)[:, 1]
            iso_p = np.where(iso.predict(Xs) == -1, 1, 0)
            verdict = np.where((proba >= thresh) | (iso_p == 1), 'WEB ATTACK', 'BENIGN')

            out = data.copy()
            out['XGB_attack_prob'] = proba.round(3)
            out['IsolationForest'] = np.where(iso_p == 1, 'anomaly', 'normal')
            out['VERDICT'] = verdict

            n_attack = int((verdict == 'WEB ATTACK').sum())
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><div class="num">{len(out)}</div>'
                        f'<div class="lbl">Flows scanned</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="num">{n_attack}</div>'
                        f'<div class="lbl">Attacks flagged</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="num">'
                        f'{100*n_attack/max(len(out),1):.1f}%</div>'
                        f'<div class="lbl">Attack rate</div></div>', unsafe_allow_html=True)

            st.dataframe(out[['XGB_attack_prob','IsolationForest','VERDICT']].head(200),
                         use_container_width=True)
            st.download_button("Download full results",
                               out.to_csv(index=False).encode(),
                               "detection_results.csv", "text/csv")

# ---------------- TAB 4: CRF sequence scan ----------------
with tab4:
    st.markdown("#### Sequence detection with the CRF (temporal model)")
    st.caption("The CRF is a sequence model: it needs MANY flows in time order and labels "
               "each flow using its neighbours. Upload a batch CSV (e.g. demo_batch_NEW.csv) "
               "with the feature columns, in capture order.")

    if not crf_ready:
        st.warning("CRF files not found. To enable this tab, place model_crf.joblib, "
                   "crf_binner.joblib and crf_top_idx.joblib (from the CRF notebook) "
                   "next to app.py.")
    else:
        seqfile = st.file_uploader("Sequence CSV (many flows, in order)", type=['csv'], key='seq')
        if seqfile is not None:
            sdata = pd.read_csv(seqfile)
            sdata.columns = sdata.columns.str.strip()
            missing = [f for f in feature_names if f not in sdata.columns]
            if missing:
                st.error(f"CSV missing {len(missing)} feature columns.")
            else:
                sdata = sdata.replace([np.inf,-np.inf], np.nan).dropna(subset=feature_names)
                from sklearn.preprocessing import StandardScaler
                # reproduce the CRF preprocessing: scale -> top features -> bin -> symbols
                Xvals = sdata[feature_names].values.astype(float)
                if crf_scaler is not None:
                    Xs = crf_scaler.transform(Xvals)      # exact scaler from training (accurate)
                else:
                    st.warning("Using an approximate scaler (crf_scaler.joblib not found). "
                               "For accurate CRF results, save crf_scaler in the notebook and "
                               "place crf_scaler.joblib next to app.py.")
                    Xs = StandardScaler().fit_transform(Xvals)
                Xtop = Xs[:, crf_topix]
                Xbin = crf_binner.transform(Xtop).astype(int)
                nf = Xbin.shape[1]

                # build ONE sequence with neighbour context (whole uploaded batch)
                seq = []
                L = len(Xbin)
                for t in range(L):
                    d = {f'f{j}': str(Xbin[t, j]) for j in range(nf)}
                    if t > 0:
                        for j in range(nf): d[f'-1f{j}'] = str(Xbin[t-1, j])
                    if t < L-1:
                        for j in range(nf): d[f'+1f{j}'] = str(Xbin[t+1, j])
                    seq.append(d)

                pred = crf_model.predict([seq])[0]
                pred = np.array([int(v) for v in pred])
                n_attack = int(pred.sum())

                c1, c2 = st.columns(2)
                c1.markdown(f'<div class="metric-card"><div class="num">{L}</div>'
                            f'<div class="lbl">Flows in sequence</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="num">{n_attack}</div>'
                            f'<div class="lbl">Flagged by CRF</div></div>', unsafe_allow_html=True)

                # timeline plot
                fig, ax = plt.subplots(figsize=(10, 2.5))
                ax.plot(pred, color='#199e70', linewidth=1)
                ax.set_title('CRF-detected attacks across the uploaded sequence')
                ax.set_xlabel('Flow position (time order)'); ax.set_ylabel('1 = attack')
                fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
                ax.tick_params(colors='#c9d1d9'); ax.xaxis.label.set_color('#c9d1d9')
                ax.yaxis.label.set_color('#c9d1d9'); ax.title.set_color('#e6edf3')
                for s in ax.spines.values(): s.set_color('#21262d')
                st.pyplot(fig)

                out = sdata.copy()
                out['CRF_verdict'] = np.where(pred==1, 'WEB ATTACK', 'BENIGN')
                st.download_button("Download CRF results",
                                   out.to_csv(index=False).encode(),
                                   "crf_sequence_results.csv", "text/csv")

# ---------------- TAB 3: about ----------------
with tab3:
    st.markdown("""
### About this console
This is the deployment layer of a semi-supervised intrusion detection system built on the
**CIC-IDS-2017** dataset (`Thursday-WorkingHours-Morning-WebAttacks` subset).

**Pipeline**
1. Clean flows (remove NaN/inf) → scale features
2. **XGBoost** — supervised detector for known web attacks (Brute Force, XSS, SQL Injection)
3. **Isolation Forest** — unsupervised anomaly detector for novel threats
4. **OR-fusion** into a single verdict (semi-supervised)
5. **SHAP** — per-flow explanation of every decision

**Metrics that matter here:** because ~98.7% of traffic is benign, we evaluate on
**Precision / Recall / F1**, not accuracy.
""")
