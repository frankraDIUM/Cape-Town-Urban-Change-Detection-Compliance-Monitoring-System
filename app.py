# Cape Town Urban Compliance System


import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime
import os
import sqlite3
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Cape Town Urban Compliance", layout="wide")
st.title("🏙️ Cape Town Urban Compliance Monitoring System")

# ====================== DATABASE SETUP ======================
DB_FILE = "compliance.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            structure_id TEXT NOT NULL,
            model_risk_score REAL,
            model_status TEXT,
            human_decision TEXT,
            case_status TEXT,
            confidence INTEGER,
            comment TEXT,
            time_spent_sec INTEGER,
            timestamp TEXT,
            model_version TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_decision(data):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO review_log
        (structure_id, model_risk_score, model_status, human_decision,
         case_status, confidence, comment, time_spent_sec, timestamp, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["structure_id"], data["model_risk_score"], data["model_status"],
        data["human_decision"], data["case_status"], data["confidence"],
        data["comment"], data["time_spent_sec"], data["timestamp"],
        st.session_state.get("model_version", "v1")
    ))
    conn.commit()
    conn.close()
    # Trigger retraining after new label
    st.session_state["trigger_retrain"] = True

def load_log():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM review_log ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def delete_last_record():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM review_log WHERE id = (SELECT MAX(id) FROM review_log)")
    conn.commit()
    conn.close()

def clear_log():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM review_log")
    conn.commit()
    conn.close()

# ====================== UNCERTAINTY ======================
def compute_uncertainty(score):
    return 1 - abs(score - 0.5) * 2

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    df = gpd.read_file("compliance_results_ultimate_final.geojson")
    df["structure_id"] = "S" + df.index.astype(str).str.zfill(4)
    return df

df = load_data()

# ====================== SESSION STATE ======================
for key in ["selected_id", "auto_mode", "active_learning", "show_satellite", "prediction_success", "start_time", "trigger_retrain"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key == "selected_id" else False

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["Review System", "ML Lab", "Evaluation", "Inspection Priority"])

# ====================== TAB 1: REVIEW SYSTEM ======================
with tab1:
    st.markdown("**Human-in-the-Loop Review Tool**")
    if "ml_predictions" in st.session_state:
        st.success("ML predictions ACTIVE on map")
    else:
        st.info("ML not applied yet")

    # Sidebar (unchanged)
    st.sidebar.header("Filters")
    risk_filter = st.sidebar.multiselect("Risk Category", options=df["risk_category"].unique(), default=df["risk_category"].unique())
    growth_filter = st.sidebar.multiselect("Growth Type", options=df["growth_type"].unique(), default=df["growth_type"].unique())
    st.session_state["show_satellite"] = st.sidebar.checkbox("Show Satellite Imagery", st.session_state.get("show_satellite", False))
    st.session_state["auto_mode"] = st.sidebar.checkbox("Auto-Review Mode", st.session_state["auto_mode"])
    st.session_state["active_learning"] = st.sidebar.checkbox("Active Learning Mode", st.session_state["active_learning"])
    order_option = st.sidebar.selectbox("Base Order", ["Risk High→Low", "Risk Low→High", "Random"])

    # Filter data
    filtered_df = df[(df["risk_category"].isin(risk_filter)) & (df["growth_type"].isin(growth_filter))].copy()

    # === AUTO RETRAINING LOGIC (with visible feedback) ===
    if st.session_state.get("trigger_retrain", False) and len(load_log()) >= 10:
        with st.spinner("🔄 Retraining active learning model with new labels..."):
            # Reuse your existing training logic here (minimal duplication)
            log = load_log()
            log_latest = log.sort_values("timestamp").drop_duplicates("structure_id", keep="last")
            df_ml = df.merge(log_latest[["structure_id", "human_decision"]], on="structure_id", how="left")
            df_ml["label"] = df_ml["human_decision"].map({"Valid": 0, "Illegal": 1, "Uncertain": np.nan})
            df_ml = df_ml.dropna(subset=["label"])

            le = LabelEncoder()
            df_ml["growth_encoded"] = le.fit_transform(df_ml["growth_type"])

            feature_cols = ["final_risk_score", "area_m2", "dist_to_road_m", "dist_to_existing_m",
                            "cluster_size", "growth_encoded"]

            X = df_ml[feature_cols].fillna(0)
            y = df_ml["label"]

            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X, y)   # retrain on all available labeled data

            st.session_state["ml_model"] = model
            st.session_state["model_version"] = datetime.now().strftime("%Y%m%d_%H%M")
            st.session_state["trigger_retrain"] = False

            # Apply predictions again automatically
            df_full = df.copy()
            default_class = le.classes_[0]
            df_full["growth_encoded"] = le.transform(
                df_full["growth_type"].apply(lambda x: x if x in le.classes_ else default_class)
            )
            X_full = df_full[feature_cols].fillna(0)
            st.session_state["ml_predictions"] = model.predict_proba(X_full)[:, 1]
            st.session_state["ml_ids"] = df["structure_id"].tolist()

        st.success("Model retrained successfully with new labels!")

    # Active Learning sorting (unchanged from your last version)
    if st.session_state["active_learning"] and "ml_predictions" in st.session_state and "ml_ids" in st.session_state:
        ml_dict = dict(zip(st.session_state["ml_ids"], st.session_state["ml_predictions"]))
        filtered_df["ml_uncertainty"] = filtered_df["structure_id"].map(
            lambda x: abs(ml_dict.get(x, 0.5) - 0.5)
        )
        filtered_df = filtered_df.sort_values("ml_uncertainty", ascending=True)
    else:
        filtered_df["uncertainty"] = filtered_df["final_risk_score"].apply(compute_uncertainty)
        if order_option == "Risk High→Low":
            filtered_df = filtered_df.sort_values("final_risk_score", ascending=False)
        elif order_option == "Risk Low→High":
            filtered_df = filtered_df.sort_values("final_risk_score", ascending=True)
        else:
            filtered_df = filtered_df.sample(frac=1, random_state=42)

    options = [""] + list(filtered_df["structure_id"])
    st.write(f"Showing **{len(filtered_df)}** structures")


    def get_next_id(current_id):
        if current_id not in options:
            return options[1] if len(options) > 1 else ""
        idx = options.index(current_id)
        return options[idx + 1] if idx < len(options) - 1 else ""

    def auto_select_next():
        st.session_state["selected_id"] = get_next_id(st.session_state["selected_id"])

    # Layout
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Interactive Map")
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" if st.session_state["show_satellite"] else "OpenStreetMap"
        attr = "Esri World Imagery" if st.session_state["show_satellite"] else "OpenStreetMap"

        map_center = [-34.04, 18.68]
        zoom_level = 14
        if st.session_state.get("selected_id"):
            sel_df = df[df["structure_id"] == st.session_state["selected_id"]]
            if not sel_df.empty:
                sel = sel_df.to_crs("EPSG:4326").iloc[0]
                centroid = sel.geometry.centroid
                map_center = [centroid.y, centroid.x]
                zoom_level = 18

        m = folium.Map(location=map_center, zoom_start=zoom_level, tiles=tiles, attr=attr)

        has_ml = "ml_predictions" in st.session_state and "ml_ids" in st.session_state

        filtered_df_4326 = filtered_df.to_crs("EPSG:4326")

        for _, row in filtered_df_4326.iterrows():
            sid = row["structure_id"]
            base_color = {"Critical": "red", "High": "orange", "Moderate": "yellow", "Low": "green"}.get(row["risk_category"], "blue")

            if has_ml:
                try:
                    idx = st.session_state["ml_ids"].index(sid)
                    prob = st.session_state["ml_predictions"][idx]
                    opacity = 0.3 + prob * 0.7
                    fill_color = "#8B0000" if prob > 0.75 else "#ff0000" if prob > 0.5 else base_color
                except:
                    opacity = 0.7
                    fill_color = base_color
            else:
                opacity = 0.7
                fill_color = base_color

            highlight = (sid == st.session_state.get("selected_id"))

            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, fc=fill_color, op=opacity, h=highlight: {
                    "fillColor": fc, "color": "cyan" if h else "black",
                    "weight": 5 if h else 1.5, "fillOpacity": op
                },
                tooltip=f"ID: {sid} | Risk: {row['final_risk_score']:.2f}"
            ).add_to(m)

        map_data = st_folium(m, width=1100, height=650)

        if map_data and map_data.get("last_clicked"):
            clicked = map_data["last_clicked"]
            gdf = df.to_crs("EPSG:4326").copy()
            click_pt = gpd.points_from_xy([clicked["lng"]], [clicked["lat"]])[0]
            gdf_proj = gdf.to_crs(epsg=3857)
            click_proj = gpd.GeoSeries([click_pt], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
            gdf_proj["dist"] = gdf_proj.geometry.centroid.distance(click_proj)
            nearest_idx = gdf_proj["dist"].idxmin()
            st.session_state["selected_id"] = gdf.iloc[nearest_idx]["structure_id"]
            st.rerun()

    with col2:
        st.subheader("Review Panel")
        selected_id = st.selectbox("Structure ID", options=options,
                                   index=options.index(st.session_state["selected_id"]) if st.session_state["selected_id"] in options else 0)

        if selected_id != st.session_state["selected_id"]:
            st.session_state["selected_id"] = selected_id
            st.session_state["start_time"] = datetime.now()
            st.rerun()

        if selected_id:
            row = df[df["structure_id"] == selected_id].iloc[0]
            score = float(row["final_risk_score"])
            uncertainty = compute_uncertainty(score)
            st.metric("Risk Score", f"{score:.2f}")
            st.metric("Uncertainty", f"{uncertainty:.2f}")

            if "ml_predictions" in st.session_state and "ml_ids" in st.session_state:
                try:
                    idx = st.session_state["ml_ids"].index(selected_id)
                    ml_prob = st.session_state["ml_predictions"][idx]
                    st.metric("ML Prob (Illegal)", f"{ml_prob:.3f}")
                    if ml_prob > 0.7: st.error("🚨 High ML Risk")
                    elif ml_prob > 0.4: st.warning("⚠️ Medium ML Risk")
                    else: st.success("🟢 Low ML Risk")
                except:
                    pass

            status = st.selectbox("Case Status", ["New", "Under Review", "Approved", "Rejected", "Escalated"])
            decision = st.radio("Decision", ["Valid", "Illegal", "Uncertain"], horizontal=True)
            confidence = st.slider("Confidence (%)", 0, 100, 70)
            comment = st.text_area("Comment")

            if st.button("Save Decision", type="primary"):
                time_spent = (datetime.now() - st.session_state.get("start_time", datetime.now())).seconds
                save_decision({
                    "structure_id": selected_id,
                    "model_risk_score": score,
                    "model_status": str(row.get("compliance_status", "Unknown")),
                    "human_decision": decision,
                    "case_status": status,
                    "confidence": confidence,
                    "comment": comment,
                    "time_spent_sec": time_spent,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Decision saved!")
                if st.session_state["auto_mode"] or st.session_state["active_learning"]:
                    auto_select_next()
                    st.rerun()

    # Review Log Section (only in Tab 1)
    log = load_log()
    st.subheader("Review Log")
    if not log.empty:
        st.dataframe(log, use_container_width=True)
        if st.button("↩️ Undo Last Decision"):
            delete_last_record()
            st.warning("Last decision undone")
            st.rerun()
    else:
        st.info("No review logs yet.")

    if st.button("Clear All Logs"):
        clear_log()
        st.success("All logs cleared")
        st.rerun()

# ====================== TAB 2: ML LAB ======================
with tab2:
    st.subheader("Machine Learning Lab")
    log = load_log()
    if len(log) < 15:
        st.warning("Need at least 15 labeled samples.")
    else:
        log_latest = log.sort_values("timestamp").drop_duplicates("structure_id", keep="last")
        df_ml = df.merge(log_latest[["structure_id", "human_decision"]], on="structure_id", how="left")
        df_ml["label"] = df_ml["human_decision"].map({"Valid": 0, "Illegal": 1, "Uncertain": np.nan})
        df_ml = df_ml.dropna(subset=["label"])

        le = LabelEncoder()
        df_ml["growth_encoded"] = le.fit_transform(df_ml["growth_type"])

        feature_cols = ["final_risk_score", "area_m2", "dist_to_road_m", "dist_to_existing_m",
                        "cluster_size", "growth_encoded"]

        X = df_ml[feature_cols].fillna(0)
        y = df_ml["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        st.session_state["model_version"] = datetime.now().strftime("%Y%m%d_%H%M")
        st.caption(f"Model Version: {st.session_state['model_version']}")

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.metric("Model Accuracy", f"{acc:.3f}")

        st.subheader("Feature Importance")
        fi = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.dataframe(fi, use_container_width=True)

        st.session_state["ml_model"] = model

        if st.button("Apply Model Predictions to All Structures"):
            df_full = df.copy()
            default_class = le.classes_[0]
            df_full["growth_encoded"] = le.transform(
                df_full["growth_type"].apply(lambda x: x if x in le.classes_ else default_class)
            )
            X_full = df_full[feature_cols].fillna(0)
            st.session_state["ml_predictions"] = model.predict_proba(X_full)[:, 1]
            st.session_state["ml_ids"] = df["structure_id"].tolist()
            st.session_state["prediction_success"] = True
            st.rerun()

        if st.session_state.get("prediction_success", False):
            st.success("Model Applied.")
            st.session_state["prediction_success"] = False

# ====================== TAB 3: EVALUATION ======================
with tab3:
    st.subheader("Evaluation Dashboard")
    log = load_log()
    if len(log) < 10:
        st.warning("Need at least 10 labels for metrics.")
    else:
        log_latest = log.sort_values("timestamp").drop_duplicates("structure_id", keep="last")
        eval_df = df.merge(log_latest[["structure_id", "human_decision"]], on="structure_id", how="inner")

        if "ml_predictions" in st.session_state and "ml_ids" in st.session_state:
            prob_dict = dict(zip(st.session_state["ml_ids"], st.session_state["ml_predictions"]))
            eval_df["ml_prob_illegal"] = eval_df["structure_id"].map(prob_dict)
            eval_df["ml_pred"] = (eval_df["ml_prob_illegal"] > 0.5).astype(int)
            eval_df["human_label"] = eval_df["human_decision"].map({"Valid": 0, "Illegal": 1, "Uncertain": 0})

            st.subheader("Reviewer Performance")
            avg_time = log["time_spent_sec"].mean()
            st.metric("Average Review Time (seconds)", f"{avg_time:.1f}" if not pd.isna(avg_time) else "N/A")
            fast = len(log[log["time_spent_sec"] < 5])
            st.metric("Very Fast Decisions (< 5s)", fast)

            precision = precision_score(eval_df["human_label"], eval_df["ml_pred"], zero_division=0)
            recall = recall_score(eval_df["human_label"], eval_df["ml_pred"], zero_division=0)
            acc = accuracy_score(eval_df["human_label"], eval_df["ml_pred"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.3f}")
            col2.metric("Precision (Illegal)", f"{precision:.3f}")
            col3.metric("Recall (Illegal)", f"{recall:.3f}")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(eval_df["human_label"], eval_df["ml_pred"])
            st.write(pd.DataFrame(cm, index=["Actual Valid", "Actual Illegal"], columns=["Predicted Valid", "Predicted Illegal"]))

            st.subheader("Sample of Predictions vs Human Labels")
            st.dataframe(eval_df[["structure_id", "ml_prob_illegal", "ml_pred", "human_decision"]].head(20))
        else:
            st.info("Apply model predictions in the ML Lab tab first.")

# ====================== TAB 4: INSPECTION PRIORITY ======================
with tab4:
    st.subheader("Inspection Priority")
    base_df = df[
        (df["risk_category"].isin(risk_filter)) &
        (df["growth_type"].isin(growth_filter))
    ].copy()

    ml_dict = {}
    if "ml_predictions" in st.session_state and "ml_ids" in st.session_state:
        ml_dict = dict(zip(st.session_state["ml_ids"], st.session_state["ml_predictions"]))

    def compute_priority(row, ml_dict=None):
        risk = row["final_risk_score"]
        uncertainty = 1 - abs(risk - 0.5) * 2
        ml_score = ml_dict.get(row["structure_id"], 0) if ml_dict else 0
        ml_weight = 0.3 if ml_dict else 0
        priority = (0.5 * risk + ml_weight * ml_score + 0.2 * uncertainty)
        return priority

    base_df["priority_score"] = base_df.apply(lambda row: compute_priority(row, ml_dict), axis=1)

    base_df["risk_component"] = 0.5 * base_df["final_risk_score"]
    base_df["ml_component"] = base_df.apply(lambda row: 0.3 * ml_dict.get(row["structure_id"], 0) if ml_dict else 0, axis=1)
    base_df["uncertainty_component"] = 0.2 * (1 - np.abs(base_df["final_risk_score"] - 0.5) * 2)

    def priority_class(row, ml_dict=None):
        risk = row["final_risk_score"]
        ml_score = ml_dict.get(row["structure_id"], 0) if ml_dict else 0
        if risk > 0.8 and ml_score > 0.7:
            return "🚨 CRITICAL"
        elif risk > 0.7:
            return "⚠️ HIGH"
        elif ml_score > 0.7:
            return "MODEL FLAGGED"
        elif (1 - abs(risk - 0.5) * 2) > 0.6:
            return "❓ UNCERTAIN CASE"
        else:
            return "🟢 LOW"

    base_df["priority_class"] = base_df.apply(lambda row: priority_class(row, ml_dict), axis=1)

    priority_queue = base_df.sort_values("priority_score", ascending=False)

    st.dataframe(
        priority_queue[[
            "structure_id",
            "priority_class",
            "priority_score",
            "final_risk_score",
            "risk_component",
            "ml_component",
            "uncertainty_component",
            "growth_type",
            "risk_category"
        ]].head(15),
        use_container_width=True,
        hide_index=True
    )
    st.caption("Higher priority score = higher urgency for inspection. Breakdown shows contribution of each factor.")

st.caption("System Ready")
st.caption("Tip: Train model → Apply Predictions → Use Inspection Priority tab to plan field visits.")
