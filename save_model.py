#!/usr/bin/env python3
"""
Скрипт для сохранения модели из notebook в файл
"""
import sys
import os

# Добавляем путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

print("="*80)
print("СОХРАНЕНИЕ МОДЕЛИ ДЛЯ STREAMLIT ПРИЛОЖЕНИЯ")
print("="*80)

# Пытаемся загрузить переменные из notebook через pickle/IPython
# Или создаем модель на основе сохраненных данных

# Загружаем гиперпараметры
if os.path.exists('best_hyperparameters.json'):
    with open('best_hyperparameters.json', 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)
    
    print(f"\n✅ Гиперпараметры загружены из best_hyperparameters.json")
    print(f"   Модель: {hyperparams.get('model', 'Unknown')}")
    print(f"   F1-score: {hyperparams.get('best_score', 0):.4f}")
else:
    print("❌ Файл best_hyperparameters.json не найден!")
    sys.exit(1)

# Проверяем наличие данных для обучения
data_path = "data"
if not os.path.exists(data_path):
    print(f"❌ Директория с данными '{data_path}' не найдена!")
    print("   Невозможно создать модель без данных для обучения.")
    sys.exit(1)

print(f"\n📊 Загрузка данных из {data_path}...")

# Загружаем данные (простая версия - берем первые доступные файлы)
try:
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if not csv_files:
        print("❌ CSV файлы не найдены!")
        sys.exit(1)
    
    # Загружаем один файл для обучения (или можно объединить все)
    # Здесь используем упрощенный подход - загружаем первый файл с данными
    print(f"   Найдено {len(csv_files)} CSV файлов")
    
    # Пробуем загрузить данные из файлов, которые содержат нужные столбцы
    required_cols = ['GR', 'RHOB', 'NPHI', 'RDEP', 'FORCE_2020_LITHOFACIES_LITHOLOGY']
    
    all_data = []
    for file in csv_files[:5]:  # Берем первые 5 файлов
        try:
            file_path = os.path.join(data_path, file)
            df = pd.read_csv(file_path)
            
            # Проверяем наличие необходимых столбцов
            if all(col in df.columns for col in required_cols):
                # Добавляем WELL_NAME если его нет
                if 'WELL_NAME' not in df.columns:
                    df['WELL_NAME'] = file.replace('.csv', '')
                all_data.append(df[required_cols + ['WELL_NAME']])
        except Exception as e:
            continue
    
    if not all_data:
        print("❌ Не найдены файлы с необходимыми столбцами!")
        sys.exit(1)
    
    # Объединяем данные
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"   ✅ Загружено {len(combined_df)} записей")
    
    # Обрабатываем категориальные данные
    if 'FORCE_2020_LITHOFACIES_LITHOLOGY' in combined_df.columns:
        # Преобразуем целевую переменную если нужно
        if combined_df['FORCE_2020_LITHOFACIES_LITHOLOGY'].dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            combined_df['FORCE_2020_LITHOFACIES_LITHOLOGY'] = le.fit_transform(
                combined_df['FORCE_2020_LITHOFACIES_LITHOLOGY']
            )
    
    # Удаляем пропуски (или можно заполнить)
    combined_df = combined_df.dropna()
    
    if len(combined_df) < 100:
        print(f"⚠️ Мало данных после обработки: {len(combined_df)} записей")
    
    # Разделяем на X и y
    X = combined_df[['GR', 'RHOB', 'NPHI', 'RDEP', 'WELL_NAME']]
    y = combined_df['FORCE_2020_LITHOFACIES_LITHOLOGY']
    
    print(f"\n🔨 Создание модели с оптимальными гиперпараметрами...")
    
    # Извлекаем параметры
    params = hyperparams.get('best_params', {})
    
    # Преобразуем параметры
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
                model_params[key] = int(value) if str(value).isdigit() else None
        else:
            model_params[key] = value
    
    # Создаем pipeline
    numeric_features = ['GR', 'RHOB', 'NPHI', 'RDEP']
    categorical_features = ['WELL_NAME']
    
    # Преобразуем WELL_NAME в категориальный тип
    X['WELL_NAME'] = X['WELL_NAME'].astype('category')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Создаем модель с гиперпараметрами
    rf_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **model_params
    )
    
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_model)
    ])
    
    print(f"   Параметры модели:")
    for param, value in model_params.items():
        print(f"      - {param}: {value}")
    
    print(f"\n⏳ Обучение модели на {len(X)} записях...")
    final_pipeline.fit(X, y)
    
    print(f"✅ Модель обучена!")
    
    # Сохраняем модель
    model_filename = 'best_pipeline_final.pkl'
    joblib.dump(final_pipeline, model_filename)
    
    print(f"\n✅ Модель успешно сохранена в файл: {model_filename}")
    print(f"   Размер файла: {os.path.getsize(model_filename) / (1024*1024):.2f} MB")
    
    print(f"\n💡 Теперь можно перезапустить Streamlit приложение:")
    print(f"   streamlit run streamlit_app.py")
    
except Exception as e:
    print(f"\n❌ Ошибка: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

