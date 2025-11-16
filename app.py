import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mer.config import MODELS_DIR, SAMPLE_RATE
from mer.modeling.embeddings import extract_embeddings, load_audio_mono
from mer.modeling.utils.metrics import labels_convert

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


def load_model(model_path: Path, device: str = "cpu"):
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def analyze_audio(audio_path: Path, model, device: str = "cpu"):
    try:
        with st.spinner("Extracting embeddings..."):
            song_embds = extract_embeddings(audio_path)
        
        with st.spinner("Performing inference..."):
            with torch.no_grad():
                P = model(torch.from_numpy(song_embds).unsqueeze(0).float().to(device))
                P = P.squeeze(0).cpu().numpy().clip(-1.0, 1.0).astype("float32")
        
        out_data = {
            "valence_norm": P[:, 0],
            "arousal_norm": P[:, 1],
            "valence_19": labels_convert(P[:, 0], src="norm", dst="19"),
            "arousal_19": labels_convert(P[:, 1], src="norm", dst="19"),
        }
        
        df = pd.DataFrame(out_data)
        return df
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None


def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


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
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            with st.spinner("Loading model..."):
                model = load_model(model_path, device=device)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.model_path = model_path
                    st.success(f"✅ Model loaded: {model_file.name}")
                    st.info(f"Device: {device}")
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
                device = "cuda" if torch.cuda.is_available() else "cpu"
                with st.spinner("Loading model..."):
                    model = load_model(selected_model, device=device)
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.model_path = selected_model
                        st.success(f"✅ Model loaded: {selected_model.name}")
                        st.info(f"Device: {device}")

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
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            predictions = analyze_audio(temp_audio_path, st.session_state.model, device)
            
            if predictions is not None:
                st.session_state.predictions = predictions
                st.success("✅ Analysis complete!")
                
                predictions_path = MODELS_DIR / "va_predictions.csv"
                predictions.to_csv(predictions_path, index=False)
                st.info(f"Predictions saved to {predictions_path}")
        
        if st.session_state.predictions is not None:
            predictions = st.session_state.predictions
            
            st.subheader("Predictions Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Valence (norm)", f"{predictions['valence_norm'].mean():.3f}")
            with col2:
                st.metric("Mean Arousal (norm)", f"{predictions['arousal_norm'].mean():.3f}")
            with col3:
                st.metric("Mean Valence (1-9)", f"{predictions['valence_19'].mean():.2f}")
            with col4:
                st.metric("Mean Arousal (1-9)", f"{predictions['arousal_19'].mean():.2f}")

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

            st.subheader("Valence-Arousal Time Frames")

            valence_col = 'valence_norm' if scale_mode == "norm" else 'valence_19'
            arousal_col = 'arousal_norm' if scale_mode == "norm" else 'arousal_19'
            scale_label = "normalized" if scale_mode == "norm" else "1-9"
            y_axis_label_valence = f"Valence ({scale_label})"
            y_axis_label_arousal = f"Arousal ({scale_label})"
            
            time_points = np.arange(len(predictions)) * 1.0  # 1 second per frame
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Valence", "Arousal"),
                vertical_spacing=0.1
            )
            
            hover_format = '.3f' if scale_mode == "norm" else '.2f'
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
            
            st.divider()
            
            st.subheader("Valence-Arousal 2D Distribution")

            fig_2d = go.Figure()
            
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
            
            if scale_mode == "norm":
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
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name="va_predictions.csv",
                mime="text/csv"
            )
            
else:
    st.info("Please upload an audio file from the sidebar to get started.")

