import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Настройка страницы
st.set_page_config(page_title="AgroScore KZ | inDrive", page_icon="🌾", layout="wide")

# 1. Загрузка кэшированных моделей
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/agro_model.pkl')
        encoders = joblib.load('models/encoders.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        return model, encoders, feature_cols
    except Exception as e:
        st.error(f"⚠️ Ошибка загрузки. Убедитесь, что вы запустили scoring.py и папка models/ существует.\nОшибка: {e}")
        return None, None, None

model, encoders, feature_cols = load_models()

# 2. Функция предобработки для новых данных
def preprocess_data(df, encoders, feature_cols):
    df_proc = df.copy()
    
    # Заполнение пропусков и расчеты
    df_proc['Причитающая сумма'] = pd.to_numeric(df_proc['Причитающая сумма'], errors='coerce').fillna(0)
    df_proc['Норматив'] = pd.to_numeric(df_proc['Норматив'], errors='coerce').fillna(1)
    df_proc['поголовье'] = df_proc['Причитающая сумма'] / df_proc['Норматив'].replace(0, 1)

    # Логарифмирование
    df_proc['log_сумма'] = np.log1p(df_proc['Причитающая сумма'])
    df_proc['log_норматив'] = np.log1p(df_proc['Норматив'])
    df_proc['log_поголовье'] = np.log1p(df_proc['поголовье'])

    # Категориальные признаки
    for col, le in encoders.items():
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna('Неизвестно').astype(str)
            # Защита от неизвестных категорий
            known_classes = set(le.classes_)
            df_proc[col] = df_proc[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            df_proc[col + '_enc'] = le.transform(df_proc[col])

    return df_proc, df_proc[feature_cols]

# ==========================================
# ИНТЕРФЕЙС
# ==========================================
st.title("🌾 AgroScore KZ — Скоринг субсидий")
st.markdown("""
Система ранжирования сельхозпроизводителей (Merit-based подход).  
*Решение для хакатона Decentrathon 5.0 (Кейс 2: inDrive GovTech).*
""")

# Загрузка файла
uploaded_file = st.file_uploader("📂 Загрузите реестр заявок (Excel из subsidy.plem.kz)", type=["xlsx"])

if uploaded_file is not None and model is not None:
    with st.spinner("Анализ заявок ИИ-моделью..."):
        # Читаем Excel (пропускаем первые 4 строки, как в вашем скрипте)
        df_raw = pd.read_excel(uploaded_file, header=4, engine='openpyxl')
        df_raw = df_raw.dropna(how='all').reset_index(drop=True)
        
        # Получаем предсказания
        df_full, X = preprocess_data(df_raw, encoders, feature_cols)
        probabilities = model.predict_proba(X.values)[:, 1]
        
        # Добавляем результаты
        df_full['Score'] = probabilities
        df_full['Рекомендация'] = df_full['Score'].apply(
            lambda x: '✅ Рекомендовать' if x >= 0.80 else ('⚠️ На рассмотрение' if x >= 0.50 else '❌ Отклонить')
        )
        
        # Красивая таблица для показа
        cols_to_show = ['Номер заявки', 'Область', 'Направление водства', 'Причитающая сумма', 'Score', 'Рекомендация']
        available_cols = [c for c in cols_to_show if c in df_full.columns]
        df_display = df_full[available_cols].sort_values(by='Score', ascending=False)

    # --- БЛОК 1: Метрики ---
    st.markdown("---")
    st.subheader("📊 Сводка по загруженному реестру")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего заявок", len(df_display))
    c2.metric("✅ К одобрению (>0.80)", len(df_display[df_display['Score'] >= 0.8]))
    c3.metric("⚠️ На комиссию", len(df_display[(df_display['Score'] < 0.8) & (df_display['Score'] >= 0.5)]))
    c4.metric("❌ Отклонить (<0.50)", len(df_display[df_display['Score'] < 0.5]))

    # --- БЛОК 2: Shortlist ---
    st.subheader("📋 Shortlist заявителей (Ранжированный список)")
    st.dataframe(df_display.style.background_gradient(subset=['Score'], cmap='RdYlGn'), height=300, use_container_width=True)

    # --- БЛОК 3: Explainability (SHAP) ---
    st.markdown("---")
    st.subheader("🔍 Explainability: Объяснение решения модели")
    st.markdown("Выберите номер заявки, чтобы увидеть, **почему** система поставила такой балл. Это убирает эффект «черного ящика».")
    
    selected_id = st.selectbox("Выберите номер заявки:", df_display['Номер заявки'].unique())
    
    if selected_id:
        # Находим строку данных для этой заявки
        idx = df_full[df_full['Номер заявки'] == selected_id].index[0]
        row_features = X.iloc[[idx]]
        score = df_full.loc[idx, 'Score']
        
        st.write(f"**Текущий балл заявки:** {score:.2f} ({df_full.loc[idx, 'Рекомендация']})")
        
        # Считаем SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row_features)
        
        # Рисуем красивый график
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Читаемые названия признаков
        features_renamed = ['Сумма (log)', 'Норматив (log)', 'Поголовье (log)', 'Область', 'Направление', 'Тип субсидии', 'Район']
        contributions = shap_values[0]
        
        colors = ['#2ca02c' if val > 0 else '#d62728' for val in contributions]
        y_pos = np.arange(len(features_renamed))
        
        ax.barh(y_pos, contributions, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features_renamed)
        ax.set_xlabel('Влияние на финальный Score')
        ax.set_title('Факторы: Зеленые повысили шанс одобрения, Красные — понизили')
        
        st.pyplot(fig)
