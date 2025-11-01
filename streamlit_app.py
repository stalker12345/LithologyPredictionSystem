# ================================================================================
# STREAMLIT ПРИЛОЖЕНИЕ ДЛЯ ПРЕДСКАЗАНИЯ ЛИТОЛОГИИ
# ================================================================================
# Система для демонстрации модели машинного обучения, предсказывающей
# литологию на основе геологических параметров скважин
# ================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ================================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ================================================================================
st.set_page_config(
    page_title="Система предсказания литологии",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# ЗАГРУЗКА МОДЕЛИ И МЕТАДАННЫХ
# ================================================================================

# Константы для демо-режима (определяем до использования)
WELL_NAMES_DEMO = [
    '15/9-23', '16/2-7', '16/7-6', '17/4-1', 
    '25/10-9', '31/2-10', '31/2-21 S', 
    '34/3-2 S', '35/9-7'
]

@st.cache_resource
def load_model():
    """
    Загружает обученную модель из notebook переменных или файла
    """
    try:
        # Пытаемся загрузить из сохраненного файла
        if os.path.exists('best_pipeline_final.pkl'):
            return joblib.load('best_pipeline_final.pkl'), False  # False = не демо
        
        # Если файла модели нет, но есть гиперпараметры - используем их
        hyperparams_path = 'best_hyperparameters.json'
        if os.path.exists(hyperparams_path):
            with open(hyperparams_path, 'r', encoding='utf-8') as f:
                hyperparams = json.load(f)
            
            # Создаем модель с правильными гиперпараметрами из JSON
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import RobustScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            
            # Извлекаем параметры
            params = hyperparams.get('best_params', {})
            
            # Преобразуем параметры в правильные типы
            model_params = {}
            for key, value in params.items():
                if value == "None":
                    model_params[key] = None
                elif value == "True":
                    model_params[key] = True
                elif value == "False":
                    model_params[key] = False
                elif key == 'n_estimators':
                    model_params[key] = int(value)
                elif key in ['min_samples_split', 'min_samples_leaf']:
                    model_params[key] = int(value)
                elif key == 'max_features':
                    if value in ['sqrt', 'log2']:
                        model_params[key] = value
                    else:
                        model_params[key] = float(value)
                elif key == 'max_depth':
                    if value == "None":
                        model_params[key] = None
                    else:
                        model_params[key] = int(value) if value.isdigit() else None
                else:
                    model_params[key] = value
            
            # Создаем модель с правильными параметрами
            numeric_features = ['GR', 'RHOB', 'NPHI', 'RDEP']
            categorical_features = ['WELL_NAME']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', RobustScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
                ],
                remainder='passthrough'
            )
            
            # Создаем модель с гиперпараметрами из JSON
            rf_model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                **model_params
            )
            
            demo_model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', rf_model)
            ])
            
            # Обучаем на фиктивных данных (все равно демо, но с правильными параметрами)
            demo_X = pd.DataFrame({
                'GR': np.random.uniform(0, 200, 100),
                'RHOB': np.random.uniform(1.5, 3.0, 100),
                'NPHI': np.random.uniform(0, 0.5, 100),
                'RDEP': np.random.uniform(0.1, 1000, 100),
                'WELL_NAME': np.random.choice(WELL_NAMES_DEMO, 100)
            })
            demo_y = np.random.randint(0, 8, 100)
            
            demo_model.fit(demo_X, demo_y)
            
            return demo_model, True  # True = демо-режим, но с правильными гиперпараметрами
        
        else:
            # Демо-режим: создаем простую модель для тестирования без гиперпараметров
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import RobustScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            
            # Создаем простую демо-модель
            numeric_features = ['GR', 'RHOB', 'NPHI', 'RDEP']
            categorical_features = ['WELL_NAME']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', RobustScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
                ],
                remainder='passthrough'
            )
            
            demo_model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1))
            ])
            
            # Создаем фиктивные данные для обучения демо-модели
            demo_X = pd.DataFrame({
                'GR': np.random.uniform(0, 200, 100),
                'RHOB': np.random.uniform(1.5, 3.0, 100),
                'NPHI': np.random.uniform(0, 0.5, 100),
                'RDEP': np.random.uniform(0.1, 1000, 100),
                'WELL_NAME': np.random.choice(WELL_NAMES_DEMO, 100)
            })
            demo_y = np.random.randint(0, 8, 100)
            
            demo_model.fit(demo_X, demo_y)
            
            return demo_model, True  # True = демо-режим
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False

@st.cache_data
def load_hyperparameters():
    """Загружает информацию о гиперпараметрах модели"""
    try:
        if os.path.exists('best_hyperparameters.json'):
            with open('best_hyperparameters.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Ошибка загрузки гиперпараметров: {str(e)}")
        return None

# Классы литологии (соответствие из notebook)
LITHOLOGY_CLASSES = {
    0: "30000 (Sandstone - Песчаник)",
    1: "65000 (Shale - Глина)",
    2: "65030 (Marl - Мергель)",
    3: "70000 (Limestone - Известняк)",
    4: "70032 (Dolomite - Доломит)",
    5: "80000 (Coal - Уголь)",
    6: "88000 (Anhydrite - Ангидрит)",
    7: "Other (Прочее: 74000, 86000, 90000, 99000)"
}

# Русские названия для отображения
LITHOLOGY_CLASSES_RU = {
    0: "Песчаник (Sandstone)",
    1: "Глина (Shale)",
    2: "Мергель (Marl)",
    3: "Известняк (Limestone)",
    4: "Доломит (Dolomite)",
    5: "Уголь (Coal)",
    6: "Ангидрит (Anhydrite)",
    7: "Прочее (Other: 74000, 86000, 90000, 99000)"
}

# Обратный маппинг: исходные значения литологии -> индексы классов
LITHOLOGY_TO_INDEX = {
    30000: 0,
    65000: 1,
    65030: 2,
    70000: 3,
    70032: 4,
    80000: 5,
    88000: 6,
    # Редкие категории, объединенные в "Other" (класс 7)
    74000: 7,
    86000: 7,
    90000: 7,
    99000: 7,
    'Other': 7,
    '30000': 0,  # На случай строкового формата
    '65000': 1,
    '65030': 2,
    '70000': 3,
    '70032': 4,
    '80000': 5,
    '88000': 6,
    '74000': 7,
    '86000': 7,
    '90000': 7,
    '99000': 7
}

def convert_prediction_to_class_index(prediction):
    """
    Преобразует предсказание модели в индекс класса (0-7)
    Модель может возвращать либо индекс класса, либо исходное значение литологии
    """
    # Сначала пытаемся преобразовать в число или строку
    if isinstance(prediction, np.ndarray):
        pred_value = prediction.item()
        # Если это скалярное значение numpy, конвертируем в Python тип
        if hasattr(pred_value, 'item'):
            pred_value = pred_value.item()
    elif isinstance(prediction, (np.integer, np.int64, np.int32, np.float64)):
        pred_value = int(prediction)
    else:
        pred_value = prediction
    
    # Преобразуем в строку для проверки, можно ли конвертировать в int
    try:
        if isinstance(pred_value, str) and pred_value.isdigit():
            pred_value = int(pred_value)
        elif not isinstance(pred_value, (int, str)):
            pred_value = int(pred_value)
    except (ValueError, TypeError):
        pass
    
    # Проверяем, является ли это индексом класса (0-7)
    if isinstance(pred_value, int) and 0 <= pred_value <= 7:
        return pred_value
    
    # Если нет, пытаемся найти в маппинге исходных значений
    if pred_value in LITHOLOGY_TO_INDEX:
        return LITHOLOGY_TO_INDEX[pred_value]
    
    # Если значение строковое (например, "Other")
    if isinstance(pred_value, str) and pred_value in LITHOLOGY_TO_INDEX:
        return LITHOLOGY_TO_INDEX[pred_value]
    
    # Если ничего не подошло, возвращаем -1 для обработки ошибки
    return -1

# Примеры значений для справки
FEATURE_RANGES = {
    'GR': (0, 200, "Гамма-каротаж (API units)"),
    'RHOB': (1.5, 3.0, "Объемная плотность (g/cm³)"),
    'NPHI': (0, 0.5, "Нейтронная пористость (v/v)"),
    'RDEP': (0.1, 1000, "Глубокое сопротивление (Ohm.m)")
}

WELL_NAMES = [
    '15/9-23', '16/2-7', '16/7-6', '17/4-1', 
    '25/10-9', '31/2-10', '31/2-21 S', 
    '34/3-2 S', '35/9-7'
]

# ================================================================================
# ОСНОВНОЙ КОД ПРИЛОЖЕНИЯ
# ================================================================================

def main():
    # Заголовок
    st.title("🪨 Система предсказания литологии")
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; margin-bottom: 30px;'>
            <h2 style='color: white; margin: 0;'>Интеллектуальная система для определения типа горной породы</h2>
            <p style='color: #f0f0f0; margin: 10px 0 0 0;'>На основе геологических параметров скважины</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Загружаем модель
    model, is_demo = load_model()
    hyperparams = load_hyperparameters()
    
    # Показываем предупреждение о демо-режиме
    if is_demo:
        if os.path.exists('best_hyperparameters.json'):
            st.info("""
            ℹ️ **РЕЖИМ С ГИПЕРПАРАМЕТРАМИ**: Модель создана с использованием оптимальных гиперпараметров из `best_hyperparameters.json`.
            ⚠️ **ВНИМАНИЕ**: Модель обучена на демо-данных. Для использования реальной обученной модели выполните ячейку 53 в notebook для сохранения `best_pipeline_final.pkl`.
            """)
        else:
            st.warning("""
            ⚠️ **ДЕМО-РЕЖИМ**: Используется тестовая модель. 
            Для полной функциональности загрузите обученную модель из notebook.
            """)
    
    if model is None:
        st.error("""
        ### ⚠️ Модель не загружена
        
        Для работы приложения необходимо:
        1. Выполнить все ячейки notebook до ячейки с оптимизацией гиперпараметров
        2. Сохранить модель в файл, добавив в notebook после обучения:
        
        ```python
        import joblib
        joblib.dump(best_pipeline_final, 'best_pipeline_final.pkl')
        ```
        """)
        return
    
    # Боковая панель с информацией о модели
    with st.sidebar:
        st.header("📊 Информация о модели")
        
        if hyperparams:
            st.subheader("Модель")
            st.info(f"**{hyperparams.get('model', 'Unknown')}**")
            
            st.subheader("Производительность")
            st.success(f"**F1-score:** {hyperparams.get('best_score', 0):.4f}")
            
            st.subheader("Гиперпараметры")
            params = hyperparams.get('best_params', {})
            for param, value in params.items():
                st.text(f"• {param}: {value}")
        
        st.markdown("---")
        st.header("ℹ️ О системе")
        st.markdown("""
        Система использует модель машинного обучения для предсказания 
        типа литологии (горной породы) на основе геологических параметров 
        скважины.
        
        **Входные параметры:**
        - GR (Гамма-каротаж)
        - RHOB (Объемная плотность)
        - NPHI (Нейтронная пористость)
        - RDEP (Глубокое сопротивление)
        - WELL_NAME (Название скважины)
        """)
        
        st.markdown("---")
        st.markdown("**Классы литологии:**")
        for class_id in sorted(LITHOLOGY_CLASSES_RU.keys()):
            class_name_ru = LITHOLOGY_CLASSES_RU[class_id]
            st.text(f"• {class_id}: {class_name_ru}")
    
    # Основные вкладки
    tab1, tab2, tab3 = st.tabs(["🔮 Одиночное предсказание", "📁 Пакетное предсказание", "📈 Визуализация"])
    
    # ================================================================================
    # ВКЛАДКА 1: ОДИНОЧНОЕ ПРЕДСКАЗАНИЕ
    # ================================================================================
    with tab1:
        st.header("Введите параметры скважины")
        st.markdown("Заполните форму ниже для получения предсказания литологии")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Числовые параметры")
            gr = st.number_input(
                "GR (Гамма-каротаж)",
                min_value=0.0,
                max_value=500.0,
                value=50.0,
                step=0.1,
                help="Гамма-каротаж в API units"
            )
            
            rhob = st.number_input(
                "RHOB (Объемная плотность)",
                min_value=1.0,
                max_value=5.0,
                value=2.5,
                step=0.01,
                help="Объемная плотность в g/cm³"
            )
            
            nphi = st.number_input(
                "NPHI (Нейтронная пористость)",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.01,
                help="Нейтронная пористость в v/v"
            )
            
            rdep = st.number_input(
                "RDEP (Глубокое сопротивление)",
                min_value=0.1,
                max_value=10000.0,
                value=10.0,
                step=0.1,
                help="Глубокое сопротивление в Ohm.m"
            )
        
        with col2:
            st.subheader("Категориальные параметры")
            well_name = st.selectbox(
                "WELL_NAME (Название скважины)",
                options=WELL_NAMES,
                help="Выберите скважину из списка"
            )
            
            # Дополнительная информация
            st.markdown("---")
            st.markdown("### 📋 Диапазоны значений")
            for feature, (min_val, max_val, desc) in FEATURE_RANGES.items():
                st.caption(f"**{feature}**: {desc}")
                st.caption(f"Типичный диапазон: {min_val} - {max_val}")
        
        # Кнопка предсказания
        if st.button("🎯 Получить предсказание", type="primary", use_container_width=True):
            # Формируем данные для предсказания
            input_data = pd.DataFrame({
                'GR': [gr],
                'RHOB': [rhob],
                'NPHI': [nphi],
                'RDEP': [rdep],
                'WELL_NAME': [well_name]
            })
            
            try:
                # Предсказание
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                
                # Преобразуем prediction в индекс класса (0-7)
                prediction_int = convert_prediction_to_class_index(prediction)
                
                # Проверяем, что prediction_int валидный
                if not (0 <= prediction_int <= 7):
                    st.error(f"❌ Модель вернула неожиданное значение: {prediction} (тип: {type(prediction)}). Ожидались индексы 0-7 или исходные значения литологии (30000, 65000, и т.д.)")
                    st.info(f"Возможные значения: {list(LITHOLOGY_TO_INDEX.keys())}")
                    return
                
                # Проверяем размер массива вероятностей
                if len(probabilities) != 8:
                    st.error(f"❌ Неверный размер массива вероятностей: {len(probabilities)}, ожидалось 8")
                    return
                
                # Результаты
                st.markdown("---")
                st.header("🎯 Результат предсказания")
                
                col_result1, col_result2 = st.columns([1, 2])
                
                with col_result1:
                    st.subheader("Предсказанный класс")
                    predicted_class_name = LITHOLOGY_CLASSES.get(prediction_int, f"Класс {prediction_int}")
                    
                    # Большое отображение результата
                    st.markdown(f"""
                    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                border-radius: 15px; margin: 20px 0;'>
                        <h1 style='color: white; margin: 0; font-size: 48px;'>{prediction_int}</h1>
                        <p style='color: #f0f0f0; margin: 10px 0 0 0; font-size: 18px;'>{predicted_class_name}</p>
                        <p style='color: #e0e0e0; margin: 5px 0 0 0; font-size: 14px;'>
                            Вероятность: {probabilities[prediction_int]*100:.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_result2:
                    st.subheader("Вероятности всех классов")
                    
                    # Создаем DataFrame с вероятностями
                    prob_df = pd.DataFrame({
                        'Класс': [LITHOLOGY_CLASSES.get(i, f"Класс {i}") for i in range(len(probabilities))],
                        'Вероятность': probabilities
                    }).sort_values('Вероятность', ascending=False)
                    
                    # Визуализация вероятностей
                    fig = px.bar(
                        prob_df,
                        x='Вероятность',
                        y='Класс',
                        orientation='h',
                        color='Вероятность',
                        color_continuous_scale='Viridis',
                        title="Распределение вероятностей по классам",
                        labels={'Вероятность': 'Вероятность (%)', 'Класс': 'Класс литологии'}
                    )
                    fig.update_layout(
                        height=400,
                        xaxis=dict(range=[0, 1], tickformat='.1%'),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Таблица с вероятностями
                    prob_df['Вероятность (%)'] = (prob_df['Вероятность'] * 100).round(2)
                    st.dataframe(
                        prob_df[['Класс', 'Вероятность (%)']].style.format({'Вероятность (%)': '{:.2f}%'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
            except Exception as e:
                st.error(f"Ошибка при предсказании: {str(e)}")
                st.exception(e)
    
    # ================================================================================
    # ВКЛАДКА 2: ПАКЕТНОЕ ПРЕДСКАЗАНИЕ
    # ================================================================================
    with tab2:
        st.header("Пакетное предсказание из файла")
        st.markdown("Загрузите CSV файл с данными для массового предсказания")
        
        # Инструкции
        with st.expander("📋 Инструкции по формату файла"):
            st.markdown("""
            CSV файл должен содержать следующие столбцы:
            - `GR` (числовое)
            - `RHOB` (числовое)
            - `NPHI` (числовое)
            - `RDEP` (числовое)
            - `WELL_NAME` (текстовое, одно из значений: {})
            
            **Пример файла:**
            ```csv
            GR,RHOB,NPHI,RDEP,WELL_NAME
            50.0,2.5,0.2,10.0,15/9-23
            45.0,2.3,0.15,8.5,16/2-7
            ```
            """.format(', '.join(WELL_NAMES)))
        
        uploaded_file = st.file_uploader(
            "Выберите CSV файл",
            type=['csv'],
            help="Загрузите CSV файл с данными скважин"
        )
        
        if uploaded_file is not None:
            try:
                # Загружаем данные
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Файл загружен: {len(df)} записей")
                
                # Проверяем наличие необходимых столбцов
                required_columns = ['GR', 'RHOB', 'NPHI', 'RDEP', 'WELL_NAME']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Отсутствуют необходимые столбцы: {', '.join(missing_columns)}")
                    st.info(f"Найденные столбцы: {', '.join(df.columns)}")
                else:
                    # Показываем предпросмотр данных
                    st.subheader("📊 Предпросмотр данных")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Кнопка предсказания
                    if st.button("🚀 Выполнить пакетное предсказание", type="primary", use_container_width=True):
                        with st.spinner("Выполняется предсказание..."):
                            # Предсказания
                            predictions = model.predict(df[required_columns])
                            probabilities = model.predict_proba(df[required_columns])
                            
                            # Добавляем результаты в DataFrame
                            result_df = df.copy()
                            # Преобразуем predictions в индексы классов (0-7)
                            predictions_int = []
                            for p in predictions:
                                pred_idx = convert_prediction_to_class_index(p)
                                if 0 <= pred_idx <= 7:
                                    predictions_int.append(pred_idx)
                                else:
                                    # Если не удалось преобразовать, оставляем исходное значение
                                    predictions_int.append(pred_idx)
                            result_df['Predicted_Class'] = predictions_int
                            result_df['Predicted_Lithology'] = [
                                LITHOLOGY_CLASSES.get(p, f"Класс {p}") for p in predictions_int
                            ]
                            
                            # Добавляем вероятности
                            for i, class_name in LITHOLOGY_CLASSES.items():
                                result_df[f'Prob_Class_{i}'] = probabilities[:, i]
                            
                            st.success(f"✅ Предсказание завершено для {len(df)} записей")
                            
                            # Статистика предсказаний
                            st.subheader("📈 Статистика предсказаний")
                            
                            col_stat1, col_stat2 = st.columns(2)
                            
                            with col_stat1:
                                # Распределение классов
                                class_counts = pd.Series(predictions_int).value_counts().sort_index()
                                class_counts_df = pd.DataFrame({
                                    'Класс': [LITHOLOGY_CLASSES.get(i, f"Класс {i}") for i in class_counts.index],
                                    'Количество': class_counts.values,
                                    'Процент': (class_counts.values / len(predictions) * 100).round(2)
                                })
                                
                                fig_dist = px.bar(
                                    class_counts_df,
                                    x='Класс',
                                    y='Количество',
                                    color='Количество',
                                    color_continuous_scale='Blues',
                                    title="Распределение предсказанных классов"
                                )
                                fig_dist.update_layout(showlegend=False)
                                st.plotly_chart(fig_dist, use_container_width=True)
                            
                            with col_stat2:
                                # Таблица статистики
                                st.dataframe(
                                    class_counts_df.style.format({'Процент': '{:.2f}%'}),
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            # Результаты в таблице
                            st.subheader("📋 Результаты предсказания")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Кнопка скачивания результатов
                            csv_result = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="💾 Скачать результаты (CSV)",
                                data=csv_result,
                                file_name="predictions_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
            
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {str(e)}")
                st.exception(e)
    
    # ================================================================================
    # ВКЛАДКА 3: ВИЗУАЛИЗАЦИЯ
    # ================================================================================
    with tab3:
        st.header("Визуализация и анализ")
        
        st.markdown("""
        ### Интерактивная визуализация параметров
        
        Используйте слайдеры ниже для изменения параметров и наблюдения 
        за изменениями предсказания в реальном времени.
        """)
        
        col_viz1, col_viz2 = st.columns([2, 1])
        
        with col_viz1:
            # Слайдеры для параметров
            gr_viz = st.slider("GR", 0.0, 200.0, 50.0, 1.0, key="gr_viz")
            rhob_viz = st.slider("RHOB", 1.5, 3.0, 2.5, 0.01, key="rhob_viz")
            nphi_viz = st.slider("NPHI", 0.0, 0.5, 0.2, 0.01, key="nphi_viz")
            rdep_viz = st.slider("RDEP", 0.1, 1000.0, 10.0, 0.1, key="rdep_viz")
        
        with col_viz2:
            well_name_viz = st.selectbox("WELL_NAME", WELL_NAMES, key="well_viz")
            
            # Быстрое предсказание
            viz_input = pd.DataFrame({
                'GR': [gr_viz],
                'RHOB': [rhob_viz],
                'NPHI': [nphi_viz],
                'RDEP': [rdep_viz],
                'WELL_NAME': [well_name_viz]
            })
            
            try:
                viz_pred = model.predict(viz_input)[0]
                viz_probs = model.predict_proba(viz_input)[0]
                
                # Преобразуем в индекс класса (0-7)
                viz_pred_int = convert_prediction_to_class_index(viz_pred)
                
                st.metric(
                    "Предсказанный класс",
                    f"{viz_pred_int}",
                    LITHOLOGY_CLASSES.get(viz_pred_int, f"Класс {viz_pred_int}")
                )
                st.metric(
                    "Уверенность",
                    f"{viz_probs[viz_pred_int]*100:.1f}%"
                )
            except:
                pass
        
        # Визуализация влияния параметров
        st.subheader("Анализ влияния параметров")
        
        analysis_param = st.selectbox(
            "Выберите параметр для анализа",
            ['GR', 'RHOB', 'NPHI', 'RDEP'],
            help="Параметр будет варьироваться, остальные останутся постоянными"
        )
        
        # Значения по умолчанию
        default_values = {
            'GR': gr_viz,
            'RHOB': rhob_viz,
            'NPHI': nphi_viz,
            'RDEP': rdep_viz
        }
        
        # Создаем диапазон значений для анализа
        if analysis_param == 'GR':
            param_range = np.linspace(0, 200, 50)
        elif analysis_param == 'RHOB':
            param_range = np.linspace(1.5, 3.0, 50)
        elif analysis_param == 'NPHI':
            param_range = np.linspace(0, 0.5, 50)
        else:  # RDEP
            param_range = np.logspace(np.log10(0.1), np.log10(1000), 50)
        
        # Выполняем предсказания для диапазона
        predictions_range = []
        probabilities_range = []
        
        for param_val in param_range:
            input_data = pd.DataFrame({
                'GR': [default_values['GR']],
                'RHOB': [default_values['RHOB']],
                'NPHI': [default_values['NPHI']],
                'RDEP': [default_values['RDEP']],
                'WELL_NAME': [well_name_viz]
            })
            input_data[analysis_param] = param_val
            
            try:
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0]
                # Преобразуем в индекс класса (0-7)
                pred_int = convert_prediction_to_class_index(pred)
                if 0 <= pred_int <= 7:
                    predictions_range.append(pred_int)
                    probabilities_range.append(prob)
            except Exception as e:
                pass
        
        if predictions_range:
            # График изменения предсказания
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=param_range[:len(predictions_range)],
                y=predictions_range,
                mode='lines+markers',
                name='Предсказанный класс',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f"Влияние параметра {analysis_param} на предсказание",
                xaxis_title=analysis_param,
                yaxis_title="Предсказанный класс",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Футер
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Система предсказания литологии | Разработано с использованием Streamlit и scikit-learn</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

