import os
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
# Configuración CORS para permitir solicitudes desde cualquier origen
CORS(app, resources={"*": {"origins": "*"}})

# Configuración de la aplicación
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'encuesta-ti-redsalud-2025')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Asegurarse de que la carpeta de uploads exista
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
app.config['ALLOWED_EXTENSIONS'] = {'tsv', 'csv'}

# Asegurar que exista el directorio de uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Columnas de preguntas para el análisis
QUESTION_COLS = [
    '2.1) ¿Cuál es su nivel de satisfacción global con la gestión de TI Redsalud?',
    '2.2) ¿Cuál es su nivel de satisfacción con los sistemas que utiliza (Ej. HIS, DIS, ERP, RIS o LIS)',
    '2.3) ¿Estoy informado de la estrategia TI, sus proyectos 2024 y su nivel de avance?',
    '2.4) Ante una necesidad de TI, ¿Sé a quién y cómo contactar?',
    '2.5) ¿Puedo contar a TI cuando la necesito?',
    '2.6) ¿TI entiende mi necesidad?',
    '2.7) ¿TI me mantiene informado y cumple sus compromisos?'
]

# Función para verificar extensiones permitidas
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Función para calcular la satisfacción (estilo NPS)
def calculate_satisfaction_nps_style(data_frame, questions):
    results = {}
    for q_col in questions:
        if q_col not in data_frame.columns:
            results[q_col] = None
            continue

        # Filtrar respuestas válidas (1 a 5)
        valid_responses = data_frame[q_col].dropna()
        valid_responses = valid_responses[valid_responses.between(1, 5)]

        total_valid_responses = len(valid_responses)

        if total_valid_responses == 0:
            results[q_col] = None
            continue

        # Contar cada tipo de respuesta
        count_5 = valid_responses[valid_responses == 5].count()
        count_4 = valid_responses[valid_responses == 4].count()
        count_2 = valid_responses[valid_responses == 2].count()
        count_1 = valid_responses[valid_responses == 1].count()

        # Calcular el índice de satisfacción (estilo NPS)
        promoters_count = count_5 + count_4
        detractors_count = count_1 + count_2

        satisfaction_percentage = ((promoters_count - detractors_count) / total_valid_responses) * 100

        results[q_col] = satisfaction_percentage

    return pd.DataFrame.from_dict(results, orient='index', columns=['Satisfacción (%)'])

# Función para calcular satisfacción por gerencia
def calculate_satisfaction_by_gerencia(df_filtered, questions):
    """
    Calcula la satisfacción por gerencia para cada pregunta
    """
    gerencias = df_filtered['Gerencia'].dropna().unique()
    results = {}

    for gerencia in gerencias:
        gerencia_df = df_filtered[df_filtered['Gerencia'] == gerencia]
        gerencia_satisfaction = calculate_satisfaction_nps_style(gerencia_df, questions)
        results[gerencia] = gerencia_satisfaction['Satisfacción (%)'].to_dict()

    return results

# Función para obtener detalles de usuarios que respondieron
def get_user_details(df_filtered, question_col, gerencia=None):
    """
    Obtiene los detalles de usuarios que respondieron una pregunta específica
    usando el campo 'Colaborador - Nombre Completo'
    """
    # Filtrar por gerencia si se especifica
    if gerencia:
        df_subset = df_filtered[df_filtered['Gerencia'] == gerencia].copy()
    else:
        df_subset = df_filtered.copy()

    # Filtrar respuestas válidas para la pregunta
    if question_col not in df_subset.columns:
        return [], []

    # Filtrar respuestas válidas (1 a 5) excluyendo NaN
    valid_responses = df_subset[df_subset[question_col].notna() & df_subset[question_col].between(1, 5)]

    # Obtener nombres completos del campo específico
    if 'Colaborador - Nombre Completo' in valid_responses.columns:
        user_names = valid_responses['Colaborador - Nombre Completo'].fillna('Sin nombre').tolist()
    elif 'Nombre' in valid_responses.columns:
        # Fallback al campo 'Nombre' si 'Colaborador - Nombre Completo' no existe
        user_names = valid_responses['Nombre'].fillna('Sin nombre').tolist()
    else:
        # Último fallback usar índices
        user_names = [f"Usuario {idx}" for idx in valid_responses.index]

    user_evaluations = valid_responses[question_col].tolist()

    return user_names, user_evaluations

# Función para crear gráfico interactivo por clínica
def create_interactive_chart(df_filtered, clinic_name, questions):
    """
    Crea un gráfico de líneas interactivo para una clínica específica
    con nombres completos de colaboradores
    """
    # Calcular satisfacción por gerencia
    satisfaction_by_gerencia = calculate_satisfaction_by_gerencia(df_filtered, questions)

    if not satisfaction_by_gerencia:
        return None

    # Crear el gráfico
    fig = go.Figure()

    # Colores para cada gerencia
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    gerencias = list(satisfaction_by_gerencia.keys())

    # Calcular cantidad de usuarios por gerencia para la leyenda
    user_counts = {}
    for gerencia in gerencias:
        gerencia_df = df_filtered[df_filtered['Gerencia'] == gerencia]
        user_counts[gerencia] = len(gerencia_df)

    for idx, gerencia in enumerate(gerencias):
        satisfaction_values = []
        hover_texts = []

        # Preparar datos para cada pregunta
        for question in questions:
            satisfaction_pct = satisfaction_by_gerencia[gerencia].get(question, None)

            if satisfaction_pct is not None:
                satisfaction_values.append(satisfaction_pct)

                # Obtener detalles con nombres completos
                user_names, user_evaluations = get_user_details(df_filtered, question, gerencia)

                # Crear texto del tooltip con nombres completos
                hover_text = f"<b>{gerencia}</b><br>"
                hover_text += f"<b>Satisfacción:</b> {satisfaction_pct:.2f}%<br>"
                hover_text += f"<b>Usuarios que respondieron:</b> {len(user_names)}<br>"

                if user_names:
                    hover_text += "<b>Detalles:</b><br>"
                    for name, eval_score in zip(user_names, user_evaluations):
                        hover_text += f"• {name}: {eval_score}/5<br>"

                hover_texts.append(hover_text)
            else:
                satisfaction_values.append(None)
                hover_texts.append(f"<b>{gerencia}</b><br>Sin datos disponibles")

        # Agregar línea al gráfico
        fig.add_trace(go.Scatter(
            x=list(range(len(questions))),
            y=satisfaction_values,
            mode='lines+markers',
            name=f'{gerencia} ({user_counts[gerencia]} usuarios)',
            line=dict(color=colors[idx % len(colors)], width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_texts,
            connectgaps=False
        ))

    # Personalizar el diseño
    fig.update_layout(
        title=f"Análisis de Satisfacción por Gerencia - {clinic_name}",
        xaxis=dict(
            title='',
            tickvals=list(range(len(questions))),
            ticktext=[f"P{i+1}" for i in range(len(questions))],
            tickangle=45,
            gridcolor='lightgray',
            gridwidth=1,
            showline=True,
            linecolor='black',
            titlefont=dict(size=14)
        ),
        yaxis=dict(
            title='Índice de Satisfacción (%)',
            gridcolor='lightgray',
            gridwidth=1,
            showline=True,
            linecolor='black',
            zeroline=True,
            zerolinecolor='red',
            zerolinewidth=2,
            titlefont=dict(size=14)
        ),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        ),
    )

    # Agregar anotaciones para las preguntas
    annotations = []

    for i, question in enumerate(questions):
        question_text = question.split(')')[1].strip() if ')' in question else question
        if len(question_text) > 75:
            question_text = question_text[:72] + "..."

        annotations.append(
            dict(
                x=i,
                y=-0.3,  # Ajuste para la posición vertical de las etiquetas
                xref="x",
                yref="paper",
                text=f"<b>P{i+1}:</b> {question_text}",
                showarrow=False,
                font=dict(size=10, color='#2c3e50'),
                textangle=0,
                xanchor="center",
                align="center"
            )
        )

    fig.update_layout(annotations=annotations, height=700, margin=dict(b=200))

    return fig

# Rutas de la aplicación
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No se seleccionó ningún archivo', 'danger')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No se seleccionó ningún archivo', 'danger')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Procesar el archivo
        try:
            # Detectar separador (tab o coma)
            sep = '\t' if filename.endswith('.tsv') else ','
            
            # Intentar diferentes codificaciones
            try:
                df = pd.read_csv(filepath, sep=sep, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(filepath, sep=sep, encoding='latin1')
                except:
                    df = pd.read_csv(filepath, sep=sep, encoding='ISO-8859-1')
            
            # Guardar en sesión para uso posterior
            session['filename'] = filename
            
            # Convertir columnas de preguntas a numéricas
            for col in QUESTION_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Guardar el DataFrame procesado
            df.to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename + '.pkl'))
            
            flash('Archivo cargado y procesado exitosamente', 'success')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            flash(f'Error al procesar el archivo: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash('Tipo de archivo no permitido. Use archivos .tsv o .csv', 'warning')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    print("DEBUG: Entrando a la ruta /dashboard")  # Debug
    if 'filename' not in session:
        flash('Por favor, sube un archivo primero', 'warning')
        return redirect(url_for('index'))
    
    try:
        print("DEBUG: Leyendo archivo procesado")  # Debug
        processed_file = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + session['filename'] + '.pkl')
        print(f"DEBUG: Ruta del archivo procesado: {processed_file}")  # Debug
        
        if not os.path.exists(processed_file):
            flash('Archivo procesado no encontrado. Por favor, sube el archivo nuevamente.', 'error')
            return redirect(url_for('index'))
            
        df = pd.read_pickle(processed_file)
        print(f"DEBUG: DataFrame cargado con {len(df)} filas")  # Debug
        
        # Calcular satisfacción general
        print("DEBUG: Calculando satisfacción general")  # Debug
        satisfaction_general = calculate_satisfaction_nps_style(df, QUESTION_COLS)
        
        # Obtener lista de filiales disponibles
        print("DEBUG: Obteniendo lista de filiales")  # Debug
        filiales = df['Filial homologado'].dropna().unique().tolist()
        print(f"DEBUG: Filiales encontradas: {filiales}")  # Debug
        
        # Diccionario para almacenar gráficos por filial
        charts = {}
        
        # Crear gráficos para cada filial
        print("DEBUG: Iniciando generación de gráficos")  # Debug
        for filial in filiales:
            df_filial = df[df['Filial homologado'] == filial].copy()
            if not df_filial.empty:
                fig = create_interactive_chart(df_filial, filial, QUESTION_COLS)
                if fig:
                    chart_dict = {
                        'data': [trace.to_plotly_json() for trace in fig.data],
                        'layout': fig.layout.to_plotly_json()
                    }
                    charts[filial] = chart_dict
        
        # Preparar datos de satisfacción general para la tabla
        satisfaction_table = satisfaction_general.reset_index()
        satisfaction_table.columns = ['Pregunta', 'Satisfacción (%)']    
        satisfaction_table['Satisfacción (%)'] = satisfaction_table['Satisfacción (%)'].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        
        # Crear lista de preguntas para las referencias en la visualización
        questions_ref = []
        for i, q in enumerate(QUESTION_COLS):
            q_short = q.split(')')[1].strip() if ')' in q else q
            questions_ref.append(f"P{i+1}: {q_short}")
        
        return render_template(
            'dashboard.html', 
            satisfaction_table=satisfaction_table.to_dict('records'),
            charts=charts,
            filiales=filiales,
            questions_ref=questions_ref,
            filename=session['filename']
        )
    except Exception as e:
        app.logger.error(f'Error en dashboard: {str(e)}')
        flash(f'Error al procesar los datos: {str(e)}', 'danger')
        return redirect(url_for('index'))
        df_filial = df[df['Filial homologado'] == filial].copy()
        if not df_filial.empty:
            fig = create_interactive_chart(df_filial, filial, QUESTION_COLS)
            if fig:
                charts[filial] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Preparar datos de satisfacción general para la tabla
    satisfaction_table = satisfaction_general.reset_index()
    satisfaction_table.columns = ['Pregunta', 'Satisfacción (%)']    
    satisfaction_table['Satisfacción (%)'] = satisfaction_table['Satisfacción (%)'].apply(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    
    # Crear lista de preguntas para las referencias en la visualización
    questions_ref = []
    for i, q in enumerate(QUESTION_COLS):
        q_short = q.split(')')[1].strip() if ')' in q else q
        questions_ref.append(f"P{i+1}: {q_short}")
    
    return render_template(
        'dashboard.html', 
        satisfaction_table=satisfaction_table.to_dict('records'),
        charts=charts,
        filiales=filiales,
        questions_ref=questions_ref,
        filename=filename
    )

@app.route('/reset')
def reset():
    # Limpiar la sesión
    session.clear()
    flash('Sesión reiniciada. Cargue un nuevo archivo para comenzar.', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Configuración para desarrollo local
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
