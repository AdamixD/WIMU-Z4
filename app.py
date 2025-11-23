import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from mer.config import MODELS_DIR, SAMPLE_RATE
from mer.modeling.embeddings import extract_embeddings, load_audio_mono
from mer.modeling.predict import load_model, DEFAULT_DEVICE, analyze_audio
from mer.heads import BiGRUHead, BiGRUClassificationHead

st.set_page_config(
    page_title="Music Emotion Recognition",
    page_icon="🎵",
    layout="wide"
)

if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "audio_duration" not in st.session_state:
    st.session_state.audio_duration = 0.0
if "sample_rate" not in st.session_state:
    st.session_state.sample_rate = SAMPLE_RATE
if "model" not in st.session_state:
    st.session_state.model = None
if "model_path" not in st.session_state:
    st.session_state.model_path = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "current_time" not in st.session_state:
    st.session_state.current_time = 0.0
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None
if "audio_file_name" not in st.session_state:
    st.session_state.audio_file_name = None
if "scale_mode" not in st.session_state:
    st.session_state.scale_mode = "norm"  # "norm" or "19"
if "prediction_mode" not in st.session_state:
    st.session_state.prediction_mode = None  # "VA" or "Russell4Q"
if "model_2" not in st.session_state:
    st.session_state.model_2 = None
if "model_path_2" not in st.session_state:
    st.session_state.model_path_2 = None
if "prediction_mode_2" not in st.session_state:
    st.session_state.prediction_mode_2 = None
if "predictions_2" not in st.session_state:
    st.session_state.predictions_2 = None
if "comparison_mode" not in st.session_state:
    st.session_state.comparison_mode = False


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def detect_model_type(model):
    """Detect if model is for VA regression or Russell 4Q classification"""
    if isinstance(model, BiGRUClassificationHead):
        return "Russell4Q"
    elif isinstance(model, BiGRUHead):
        return "VA"
    else:
        # Try to infer from output layer
        if hasattr(model, 'out'):
            out_features = model.out.out_features
            if out_features == 2:
                return "VA"
            elif out_features == 4:
                return "Russell4Q"
    return "VA"  # Default to VA


def analyze_audio_classification(song_embds, model, device=DEFAULT_DEVICE):
    """Analyze audio with classification model (Russell 4Q)"""
    model.eval()
    model = model.to(device)
    
    X = torch.from_numpy(song_embds).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(X)
        predictions = torch.argmax(logits, dim=-1)
    
    predictions = predictions.squeeze().cpu().numpy()
    
    # Map class indices to quadrant names and descriptions
    quadrant_map = {
        0: "Q1 (Happy/Excited)",
        1: "Q2 (Angry/Tense)",
        2: "Q3 (Sad/Depressed)",
        3: "Q4 (Calm/Relaxed)"
    }
    
    # Create dataframe with predictions
    df = pd.DataFrame({
        "frame": np.arange(len(predictions)),
        "time_seconds": np.arange(len(predictions)),
        "russell4q_class": predictions,
        "russell4q_name": [quadrant_map[p] for p in predictions],
    })
    
    return df


st.title("Music Emotion Recognition")
st.markdown("Load an audio file and model to analyze music emotion (Valence/Arousal)")

with st.sidebar:
    st.header("File Selection")
    
    st.subheader("Audio File")
    audio_file = st.file_uploader(
        "Select audio file",
        type=["mp3", "wav", "flac", "m4a", "ogg"],
        key="audio_uploader"
    )
    
    if audio_file is not None:
        audio_bytes = audio_file.read()
        st.session_state.audio_bytes = audio_bytes
        st.session_state.audio_file_name = audio_file.name
        
        try:
            with st.spinner("Loading audio..."):
                audio_data = load_audio_mono(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
                st.session_state.audio_data = audio_data
                st.session_state.audio_duration = len(audio_data) / SAMPLE_RATE
                st.session_state.sample_rate = SAMPLE_RATE
            st.success(f"✅ Audio loaded: {audio_file.name}")
            st.info(f"Duration: {format_time(st.session_state.audio_duration)}")
        except Exception as e:
            st.error(f"Error loading audio: {str(e)}")
            st.session_state.audio_data = None
    
    st.divider()
    
    st.subheader("Model File")
    model_file = st.file_uploader(
        "Select model file",
        type=["pth", "pt"],
        key="model_uploader"
    )
    
    if model_file is not None:
        try:
            model_bytes = model_file.read()
            model_path = Path(f"/tmp/{model_file.name}")
            model_path.write_bytes(model_bytes)
            
            with st.spinner("Loading model..."):
                model = load_model(model_path)
                if model is not None:
                    new_mode = detect_model_type(model)
                    # Clear predictions if model type changed
                    if st.session_state.prediction_mode != new_mode:
                        st.session_state.predictions = None
                    st.session_state.model = model
                    st.session_state.model_path = model_path
                    st.session_state.prediction_mode = new_mode
                    st.success(f"✅ Model loaded: {model_file.name}")
                    st.info(f"Device: {DEFAULT_DEVICE}")
                    st.info(f"Prediction mode: {st.session_state.prediction_mode}")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.session_state.model = None
    
    st.divider()
    
    st.subheader("Load from Models Folder")
    models_dir = MODELS_DIR
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
        if model_files:
            selected_model = st.selectbox(
                "Select model from models folder",
                options=[None] + model_files,
                format_func=lambda x: x.name if x else "None"
            )
            if selected_model:
                with st.spinner("Loading model..."):
                    model = load_model(selected_model)
                    if model is not None:
                        new_mode = detect_model_type(model)
                        # Clear predictions if model type changed
                        if st.session_state.prediction_mode != new_mode:
                            st.session_state.predictions = None
                        st.session_state.model = model
                        st.session_state.model_path = selected_model
                        st.session_state.prediction_mode = new_mode
                        st.success(f"✅ Model loaded: {selected_model.name}")
                        st.info(f"Device: {DEFAULT_DEVICE}")
                        st.info(f"Prediction mode: {st.session_state.prediction_mode}")
    
    st.divider()
    
    # Comparison Mode Section
    st.subheader("Model Comparison")
    comparison_enabled = st.checkbox(
        "Enable comparison mode",
        value=st.session_state.comparison_mode,
        help="Load a second model to compare predictions"
    )
    st.session_state.comparison_mode = comparison_enabled
    
    if comparison_enabled:
        st.write("Load a second model for comparison:")
        
        model_file_2 = st.file_uploader(
            "Select second model file",
            type=["pth", "pt"],
            key="model_uploader_2"
        )
        
        if model_file_2 is not None:
            try:
                model_bytes_2 = model_file_2.read()
                model_path_2 = Path(f"/tmp/{model_file_2.name}")
                model_path_2.write_bytes(model_bytes_2)
                
                with st.spinner("Loading second model..."):
                    model_2 = load_model(model_path_2)
                    if model_2 is not None:
                        st.session_state.model_2 = model_2
                        st.session_state.model_path_2 = model_path_2
                        st.session_state.prediction_mode_2 = detect_model_type(model_2)
                        st.success(f"✅ Second model loaded: {model_file_2.name}")
                        st.info(f"Prediction mode: {st.session_state.prediction_mode_2}")
            except Exception as e:
                st.error(f"Error loading second model: {str(e)}")
                st.session_state.model_2 = None
        
        # Or load from models folder
        if models_dir.exists():
            model_files_2 = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
            if model_files_2:
                selected_model_2 = st.selectbox(
                    "Select second model from models folder",
                    options=[None] + model_files_2,
                    format_func=lambda x: x.name if x else "None",
                    key="model_select_2"
                )
                if selected_model_2:
                    with st.spinner("Loading second model..."):
                        model_2 = load_model(selected_model_2)
                        if model_2 is not None:
                            st.session_state.model_2 = model_2
                            st.session_state.model_path_2 = selected_model_2
                            st.session_state.prediction_mode_2 = detect_model_type(model_2)
                            st.success(f"✅ Second model loaded: {selected_model_2.name}")
                            st.info(f"Prediction mode: {st.session_state.prediction_mode_2}")
    else:
        # Clear second model if comparison mode is disabled
        st.session_state.model_2 = None
        st.session_state.predictions_2 = None

if st.session_state.audio_data is not None:
    st.header("Audio Player")
    
    if st.session_state.audio_bytes:
        audio_format = None
        if hasattr(st.session_state, 'audio_file_name') and st.session_state.audio_file_name:
            ext = st.session_state.audio_file_name.split('.')[-1].lower()
            format_map = {
                'mp3': 'audio/mpeg',
                'wav': 'audio/wav',
                'flac': 'audio/flac',
                'm4a': 'audio/mp4',
                'ogg': 'audio/ogg'
            }
            audio_format = format_map.get(ext)
        
        if audio_format:
            st.audio(st.session_state.audio_bytes, format=audio_format)
        else:
            st.audio(st.session_state.audio_bytes)
    
    st.divider()
    
    st.header("Analysis")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please load a model to perform analysis.")
    else:
        if st.button("🔍 Analyze Song", type="primary"):
            audio_filename = st.session_state.audio_file_name or "audio_temp.wav"
            temp_audio_path = Path(f"/tmp/{audio_filename}")
            temp_audio_path.write_bytes(st.session_state.audio_bytes)

            with st.spinner("Extracting embeddings..."):
                embds = extract_embeddings(temp_audio_path)

            with st.spinner("Performing inference..."):
                if st.session_state.prediction_mode == "Russell4Q":
                    predictions = analyze_audio_classification(embds, st.session_state.model)
                else:
                    predictions = analyze_audio(embds, st.session_state.model)
            
            if predictions is not None:
                st.session_state.predictions = predictions
                st.success("✅ Analysis complete!")
                
                if st.session_state.prediction_mode == "Russell4Q":
                    predictions_path = MODELS_DIR / "russell4q_predictions.csv"
                else:
                    predictions_path = MODELS_DIR / "va_predictions.csv"
                predictions.to_csv(predictions_path, index=False)
                st.info(f"Predictions saved to {predictions_path}")
                
            # Also run second model if comparison mode is enabled
            if st.session_state.comparison_mode and st.session_state.model_2 is not None:
                with st.spinner("Performing inference with second model..."):
                    if st.session_state.prediction_mode_2 == "Russell4Q":
                        predictions_2 = analyze_audio_classification(embds, st.session_state.model_2)
                    else:
                        predictions_2 = analyze_audio(embds, st.session_state.model_2)
                
                if predictions_2 is not None:
                    st.session_state.predictions_2 = predictions_2
                    st.success("✅ Second model analysis complete!")
        
        # Display comparison results if both models have predictions
        if (st.session_state.comparison_mode and 
            st.session_state.predictions is not None and 
            st.session_state.predictions_2 is not None):
            st.divider()
            st.header("Model Comparison Results")
            
            from mer.modeling.utils.metrics import va_to_russell_quadrant
            
            predictions_1 = st.session_state.predictions
            predictions_2 = st.session_state.predictions_2
            mode_1 = st.session_state.prediction_mode
            mode_2 = st.session_state.prediction_mode_2
            
            # Convert both to Russell 4Q for comparison
            if mode_1 == "VA":
                # Map VA to Russell 4Q
                q1 = va_to_russell_quadrant(
                    predictions_1['valence_norm'].values,
                    predictions_1['arousal_norm'].values
                )
                quadrant_names_1 = {0: "Q1 (Happy/Excited)", 1: "Q2 (Angry/Tense)", 
                                   2: "Q3 (Sad/Depressed)", 3: "Q4 (Calm/Relaxed)"}
                q1_names = [quadrant_names_1[q] for q in q1]
            else:
                q1 = predictions_1['russell4q_class'].values
                q1_names = predictions_1['russell4q_name'].values
            
            if mode_2 == "VA":
                # Map VA to Russell 4Q
                q2 = va_to_russell_quadrant(
                    predictions_2['valence_norm'].values,
                    predictions_2['arousal_norm'].values
                )
                quadrant_names_2 = {0: "Q1 (Happy/Excited)", 1: "Q2 (Angry/Tense)", 
                                   2: "Q3 (Sad/Depressed)", 3: "Q4 (Calm/Relaxed)"}
                q2_names = [quadrant_names_2[q] for q in q2]
            else:
                q2 = predictions_2['russell4q_class'].values
                q2_names = predictions_2['russell4q_name'].values
            
            # Calculate agreement
            agreement = np.mean(q1 == q2) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model 1 Type", mode_1)
            with col2:
                st.metric("Model 2 Type", mode_2)
            with col3:
                st.metric("Agreement", f"{agreement:.1f}%")
            
            # Show distribution comparison
            st.subheader("Emotion Distribution Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model 1 Distribution:**")
                unique_1, counts_1 = np.unique(q1_names, return_counts=True)
                for name, count in zip(unique_1, counts_1):
                    pct = (count / len(q1)) * 100
                    st.write(f"  - {name}: {count} frames ({pct:.1f}%)")
            
            with col2:
                st.write("**Model 2 Distribution:**")
                unique_2, counts_2 = np.unique(q2_names, return_counts=True)
                for name, count in zip(unique_2, counts_2):
                    pct = (count / len(q2)) * 100
                    st.write(f"  - {name}: {count} frames ({pct:.1f}%)")
            
            # Timeline comparison
            st.subheader("Prediction Timeline Comparison")
            comparison_fig = go.Figure()
            
            # Model 1 timeline
            comparison_fig.add_trace(go.Scatter(
                x=np.arange(len(q1)),
                y=q1,
                mode='lines',
                name=f'Model 1 ({mode_1})',
                line=dict(width=2),
                opacity=0.7
            ))
            
            # Model 2 timeline
            comparison_fig.add_trace(go.Scatter(
                x=np.arange(len(q2)),
                y=q2,
                mode='lines',
                name=f'Model 2 ({mode_2})',
                line=dict(width=2),
                opacity=0.7
            ))
            
            comparison_fig.update_layout(
                title="Quadrant Predictions Over Time",
                xaxis_title="Frame",
                yaxis_title="Quadrant Class",
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3],
                    ticktext=['Q1', 'Q2', 'Q3', 'Q4']
                ),
                height=400
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Show confusion matrix if both are classification models
            if mode_1 == "Russell4Q" or mode_2 == "Russell4Q":
                from sklearn.metrics import confusion_matrix
                
                st.subheader("Prediction Confusion Matrix (Model 1 vs Model 2)")
                cm = confusion_matrix(q1, q2, labels=[0, 1, 2, 3])
                
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Q1', 'Q2', 'Q3', 'Q4'],
                    y=['Q1', 'Q2', 'Q3', 'Q4'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 14}
                ))
                
                cm_fig.update_layout(
                    title="Confusion Matrix (Rows: Model 1, Columns: Model 2)",
                    xaxis_title="Model 2 Predictions",
                    yaxis_title="Model 1 Predictions",
                    height=500
                )
                st.plotly_chart(cm_fig, use_container_width=True)
            
            st.divider()
        
        if st.session_state.predictions is not None:
            predictions = st.session_state.predictions
            
            # Validate that predictions match the current model type
            if st.session_state.prediction_mode == "Russell4Q":
                required_cols = ['russell4q_name', 'russell4q_class']
                if not all(col in predictions.columns for col in required_cols):
                    st.error("⚠️ Predictions don't match current model type. Please re-analyze the audio.")
                    st.session_state.predictions = None
                    st.stop()
            else:
                required_cols = ['valence_norm', 'arousal_norm']
                if not all(col in predictions.columns for col in required_cols):
                    st.error("⚠️ Predictions don't match current model type. Please re-analyze the audio.")
                    st.session_state.predictions = None
                    st.stop()
            
            st.subheader("Predictions Overview")
            
            if st.session_state.prediction_mode == "Russell4Q":
                # Classification mode - show quadrant distribution
                quadrant_counts = predictions['russell4q_name'].value_counts()
                total_frames = len(predictions)
                
                col1, col2, col3, col4 = st.columns(4)
                quadrants = ["Q1 (Happy/Excited)", "Q2 (Angry/Tense)", "Q3 (Sad/Depressed)", "Q4 (Calm/Relaxed)"]
                colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800"]
                
                for col, q_name, color in zip([col1, col2, col3, col4], quadrants, colors):
                    count = quadrant_counts.get(q_name, 0)
                    percentage = (count / total_frames) * 100 if total_frames > 0 else 0
                    with col:
                        st.markdown(f"**{q_name}**")
                        st.metric("Frames", f"{count}", f"{percentage:.1f}%")
                
            else:
                # Regression mode - show mean VA values
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Valence (norm)", f"{predictions['valence_norm'].mean():.3f}")
                with col2:
                    st.metric("Mean Arousal (norm)", f"{predictions['arousal_norm'].mean():.3f}")
                with col3:
                    st.metric("Mean Valence (1-9)", f"{predictions['valence_19'].mean():.2f}")
                with col4:
                    st.metric("Mean Arousal (1-9)", f"{predictions['arousal_19'].mean():.2f}")

            if st.session_state.prediction_mode != "Russell4Q":
                scale_mode = st.radio(
                    "Scale",
                    options=["norm", "1-9"],
                    index=0 if st.session_state.scale_mode == "norm" else 1,
                    key="scale_radio",
                    help="Switch between normalized (-1 to 1) and 1-9 scale",
                    horizontal=True
                )
                st.session_state.scale_mode = scale_mode

            st.divider()

            # Visualization based on prediction mode
            if st.session_state.prediction_mode == "Russell4Q":
                st.subheader("Emotion Quadrant Over Time")

                time_points = predictions['time_seconds'].values
                russell4q_classes = predictions['russell4q_class'].values
                
                # Color map for quadrants
                color_map = {
                    0: '#4CAF50',  # Q1 - Green
                    1: '#F44336',  # Q2 - Red
                    2: '#2196F3',  # Q3 - Blue
                    3: '#FF9800',  # Q4 - Orange
                }
                
                colors = [color_map[q] for q in russell4q_classes]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=russell4q_classes,
                    mode='markers+lines',
                    marker=dict(size=10, color=colors),
                    line=dict(color='gray', width=1),
                    hovertemplate='Time: %{x:.1f}s<br>Quadrant: %{text}<extra></extra>',
                    text=predictions['russell4q_name'].values
                ))
                
                fig.update_layout(
                    xaxis_title="Time (seconds)",
                    yaxis_title="Quadrant",
                    yaxis=dict(
                        tickmode='array',
                        tickvals=[0, 1, 2, 3],
                        ticktext=['Q1 (Happy)', 'Q2 (Angry)', 'Q3 (Sad)', 'Q4 (Calm)']
                    ),
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True, key="quadrant_timeline")
                
                st.divider()
                
                # Pie chart and bar chart
                st.subheader("Emotion Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    quadrant_counts = predictions['russell4q_name'].value_counts()
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=quadrant_counts.index,
                        values=quadrant_counts.values,
                        marker=dict(colors=['#4CAF50', '#F44336', '#2196F3', '#FF9800']),
                        hovertemplate='%{label}<br>%{value} frames (%{percent})<extra></extra>'
                    )])
                    fig_pie.update_layout(height=400, title="Emotion Quadrant Distribution")
                    st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")
                
                with col2:
                    # Bar chart
                    fig_bar = go.Figure(data=[go.Bar(
                        x=quadrant_counts.index,
                        y=quadrant_counts.values,
                        marker_color=['#4CAF50', '#F44336', '#2196F3', '#FF9800'],
                        hovertemplate='%{x}<br>%{y} frames<extra></extra>'
                    )])
                    fig_bar.update_layout(
                        height=400,
                        title="Frame Count by Quadrant",
                        xaxis_title="Quadrant",
                        yaxis_title="Number of Frames"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")
                
            else:
                # VA regression mode
                st.subheader("Valence-Arousal Time Frames")
                
                valence_col = 'valence_norm' if st.session_state.scale_mode == "norm" else 'valence_19'
                arousal_col = 'arousal_norm' if st.session_state.scale_mode == "norm" else 'arousal_19'
                scale_label = "normalized" if st.session_state.scale_mode == "norm" else "1-9"
                y_axis_label_valence = f"Valence ({scale_label})"
                y_axis_label_arousal = f"Arousal ({scale_label})"
                
                time_points = np.arange(len(predictions)) * 1.0  # 1 second per frame
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Valence", "Arousal"),
                    vertical_spacing=0.1
                )
                
                hover_format = '.3f' if st.session_state.scale_mode == "norm" else '.2f'
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=predictions[valence_col],
                        mode='lines',
                        name=f'Valence ({scale_label})',
                        line=dict(color='blue', width=2),
                        hovertemplate=f'Time: %{{x:.1f}}s<br>Valence: %{{y:{hover_format}}}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=predictions[arousal_col],
                        mode='lines',
                        name=f'Arousal ({scale_label})',
                        line=dict(color='red', width=2),
                        hovertemplate=f'Time: %{{x:.1f}}s<br>Arousal: %{{y:{hover_format}}}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
                fig.update_yaxes(title_text=y_axis_label_valence, row=1, col=1)
                fig.update_yaxes(title_text=y_axis_label_arousal, row=2, col=1)
                
                fig.update_layout(
                    height=600, 
                    showlegend=False,
                    clickmode='event+select'
                )
                
                selected_data = st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    on_select="rerun", 
                    key="valence_arousal_plot"
                )
                
                if selected_data is not None:
                    try:
                        if isinstance(selected_data, dict):
                            points = selected_data.get('selection', {}).get('points', [])
                            if not points:
                                points = selected_data.get('points', [])
                            
                            if points and len(points) > 0:
                                clicked_point = points[0]
                                clicked_time = clicked_point.get('x') or clicked_point.get('pointIndex', 0)
                                
                                if isinstance(clicked_time, (int, float)):
                                    max_duration = st.session_state.audio_duration
                                    new_time = max(0.0, min(float(clicked_time), max_duration))
                                    if abs(new_time - st.session_state.current_time) > 0.1:
                                        st.session_state.current_time = new_time
                                        st.rerun()
                    except (KeyError, TypeError, AttributeError) as e:
                        pass
            
            if st.session_state.prediction_mode != "Russell4Q":
                st.divider()
                
                st.subheader("Valence-Arousal 2D Distribution")

                fig_2d = go.Figure()
                
                time_points = np.arange(len(predictions)) * 1.0
                valence_col = 'valence_norm' if st.session_state.scale_mode == "norm" else 'valence_19'
                arousal_col = 'arousal_norm' if st.session_state.scale_mode == "norm" else 'arousal_19'
                hover_format = '.3f' if st.session_state.scale_mode == "norm" else '.2f'
                
                colors = time_points
                fig_2d.add_trace(
                    go.Scatter(
                        x=predictions[valence_col],
                        y=predictions[arousal_col],
                        mode='markers+lines',
                        marker=dict(
                            size=8,
                            color=colors,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Time (s)")
                        ),
                        line=dict(color='rgba(0,0,0,0.3)', width=1),
                        hovertemplate=f'Time: %{{customdata:.1f}}s<br>Valence: %{{x:{hover_format}}}<br>Arousal: %{{y:{hover_format}}}<extra></extra>',
                        customdata=time_points
                    )
                )
                
                if st.session_state.scale_mode == "norm":
                    fig_2d.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig_2d.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    fig_2d.add_annotation(x=0.7, y=0.7, text="Happy/Excited", showarrow=False, font=dict(size=12))
                    fig_2d.add_annotation(x=-0.7, y=0.7, text="Angry/Tense", showarrow=False, font=dict(size=12))
                    fig_2d.add_annotation(x=-0.7, y=-0.7, text="Sad/Calm", showarrow=False, font=dict(size=12))
                    fig_2d.add_annotation(x=0.7, y=-0.7, text="Relaxed/Peaceful", showarrow=False, font=dict(size=12))
                    
                    fig_2d.update_layout(
                        xaxis_title="Valence (normalized)",
                        yaxis_title="Arousal (normalized)",
                        height=500,
                        xaxis=dict(range=[-1, 1]),
                        yaxis=dict(range=[-1, 1])
                    )
                else:
                    fig_2d.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)
                    fig_2d.add_vline(x=5, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    fig_2d.add_annotation(x=7.5, y=7.5, text="Happy/Excited", showarrow=False, font=dict(size=12))
                    fig_2d.add_annotation(x=2.5, y=7.5, text="Angry/Tense", showarrow=False, font=dict(size=12))
                    fig_2d.add_annotation(x=2.5, y=2.5, text="Sad/Calm", showarrow=False, font=dict(size=12))
                    fig_2d.add_annotation(x=7.5, y=2.5, text="Relaxed/Peaceful", showarrow=False, font=dict(size=12))
                    
                    fig_2d.update_layout(
                        xaxis_title="Valence (1-9)",
                        yaxis_title="Arousal (1-9)",
                        height=500,
                        xaxis=dict(range=[1, 9]),
                        yaxis=dict(range=[1, 9])
                    )
                
                st.plotly_chart(fig_2d, use_container_width=True)
            
            st.divider()
            
            st.subheader("Predictions by Fragment")
            
            fragment_size = st.slider(
                "Fragment size (seconds)",
                min_value=1,
                max_value=30,
                value=5,
                step=1
            )
            
            num_fragments = int(np.ceil(st.session_state.audio_duration / fragment_size))
            
            for i in range(num_fragments):
                start_time = i * fragment_size
                end_time = min((i + 1) * fragment_size, st.session_state.audio_duration)
                
                start_idx = int(start_time)
                end_idx = int(end_time)
                fragment_preds = predictions.iloc[start_idx:end_idx]
                
                if len(fragment_preds) > 0:
                    with st.expander(f"Fragment {i+1}: {format_time(start_time)} - {format_time(end_time)}"):
                        if st.session_state.prediction_mode == "Russell4Q":
                            # Classification mode
                            quadrant_counts = fragment_preds['russell4q_name'].value_counts()
                            dominant_quadrant = quadrant_counts.idxmax()
                            dominant_pct = (quadrant_counts.max() / len(fragment_preds)) * 100
                            
                            st.write(f"**Dominant Emotion:** {dominant_quadrant} ({dominant_pct:.1f}%)")
                            st.write("**Distribution:**")
                            for q, count in quadrant_counts.items():
                                pct = (count / len(fragment_preds)) * 100
                                st.write(f"  - {q}: {count} frames ({pct:.1f}%)")
                            
                            st.dataframe(fragment_preds[['time_seconds', 'russell4q_name']], use_container_width=True)
                        else:
                            # Regression mode
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Valence**")
                                st.write(f"Mean (norm): {fragment_preds['valence_norm'].mean():.3f}")
                                st.write(f"Mean (1-9): {fragment_preds['valence_19'].mean():.2f}")
                                st.write(f"Min: {fragment_preds['valence_norm'].min():.3f}, Max: {fragment_preds['valence_norm'].max():.3f}")
                            
                            with col2:
                                st.write("**Arousal**")
                                st.write(f"Mean (norm): {fragment_preds['arousal_norm'].mean():.3f}")
                                st.write(f"Mean (1-9): {fragment_preds['arousal_19'].mean():.2f}")
                                st.write(f"Min: {fragment_preds['arousal_norm'].min():.3f}, Max: {fragment_preds['arousal_norm'].max():.3f}")
                            
                            st.dataframe(fragment_preds, use_container_width=True)
            
            st.divider()
            
            st.subheader("Download Predictions")
            csv = predictions.to_csv(index=False)
            filename = "russell4q_predictions.csv" if st.session_state.prediction_mode == "Russell4Q" else "va_predictions.csv"
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
            
else:
    st.info("Please upload an audio file from the sidebar to get started.")

