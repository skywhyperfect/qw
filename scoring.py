"""
AgroScore KZ — Система скоринга сельхозпроизводителей
Decentrathon 5.0 · Кейс 2 · inDrive

Использование:
    python scoring.py --data data/raw/subsidies_2025.xlsx
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
import argparse
import os

warnings.filterwarnings('ignore')


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
def load_data(filepath: str) -> pd.DataFrame:
    """Загрузка и первичная очистка реестра заявок."""
    df = pd.read_excel(filepath, header=4, engine='openpyxl')
    df = df.dropna(how='all').reset_index(drop=True)
    print(f"✅ Загружено: {len(df):,} заявок")
    return df


# ============================================================
# 2. СОЗДАНИЕ TARGET-ПЕРЕМЕННОЙ
# ============================================================
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target = был ли заявитель одобрен.

    Логика:
      1  → Исполнена / Одобрена / Сформировано поручение  (субсидия выдана)
      0  → Отклонена / Получена (не исполнена)
      NaN → Отозвано (исключаем — нет сигнала)

    Это реальный исторический сигнал из системы subsidy.plem.kz.
    """
    status_map = {
        'Исполнена': 1,
        'Одобрена': 1,
        'Сформировано поручение': 1,
        'Отклонена': 0,
        'Получена': 0,
    }
    df = df.copy()
    df['target'] = df['Статус заявки'].map(status_map)
    before = len(df)
    df = df[df['target'].notna()].reset_index(drop=True)
    print(f"✅ Target создан. Исключено 'Отозвано': {before - len(df)} заявок")
    print(f"   Одобрено:  {df['target'].sum():,.0f}  ({df['target'].mean()*100:.1f}%)")
    print(f"   Отклонено: {(df['target']==0).sum():,}  ({(1-df['target'].mean())*100:.1f}%)")
    return df


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Создание признаков из доступных колонок реестра.

    Числовые:
      - log(Причитающая сумма)  — масштаб заявки
      - log(Норматив)           — ставка субсидии на голову
      - log(поголовье)          — расчётное число голов = сумма / норматив

    Категориальные (label encoded):
      - Область                 — региональная принадлежность
      - Направление водства     — вид животноводства
      - Наименование субсидирования — тип субсидии (46 видов)
      - Район хозяйства         — район расположения хозяйства
    """
    df = df.copy()
    df['Причитающая сумма'] = pd.to_numeric(df['Причитающая сумма'], errors='coerce').fillna(0)
    df['Норматив'] = pd.to_numeric(df['Норматив'], errors='coerce').fillna(1)

    # Расчётное поголовье
    df['поголовье'] = df['Причитающая сумма'] / df['Норматив'].replace(0, 1)

    # Log-трансформации (устраняем правосторонний скос)
    df['log_сумма'] = np.log1p(df['Причитающая сумма'])
    df['log_норматив'] = np.log1p(df['Норматив'])
    df['log_поголовье'] = np.log1p(df['поголовье'])

    # Label encoding категориальных признаков
    cat_cols = [
        'Область',
        'Направление водства',
        'Наименование субсидирования',
        'Район хозяйства'
    ]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].fillna('Неизвестно'))
        encoders[col] = le

    feature_cols = [
        'log_сумма',
        'log_норматив',
        'log_поголовье',
        'Область_enc',
        'Направление водства_enc',
        'Наименование субсидирования_enc',
        'Район хозяйства_enc',
    ]

    print(f"✅ Feature engineering завершён. Признаков: {len(feature_cols)}")
    return df, feature_cols


# ============================================================
# 4. ОБУЧЕНИЕ МОДЕЛИ
# ============================================================
def train_model(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Gradient Boosting классификатор.

    Почему Gradient Boosting:
    - Высокое качество на табличных данных
    - Встроенная feature importance (для explainability)
    - Устойчивость к пропускам и выбросам
    - Работает без нормализации
    """
    X = df[feature_cols].values
    y = df['target'].astype(int).values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        verbose=0
    )

    print("⏳ Обучение модели...")
    model.fit(X_train, y_train)
    print("✅ Модель обучена")

    return model, X_test, y_test, idx_test


# ============================================================
# 5. ОЦЕНКА КАЧЕСТВА
# ============================================================
def evaluate_model(model, X_test, y_test, feature_cols: list):
    """Метрики качества + feature importance."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*60)
    print("📊 КАЧЕСТВО МОДЕЛИ")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['Отклонена', 'Одобрена']))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    print("\n📈 ВАЖНОСТЬ ПРИЗНАКОВ (объяснение модели):")
    importance_df = pd.DataFrame({
        'Признак': feature_cols,
        'Важность (%)': (model.feature_importances_ * 100).round(2)
    }).sort_values('Важность (%)', ascending=False)
    print(importance_df.to_string(index=False))

    return y_proba, auc


# ============================================================
# 6. СКОРИНГ И SHORTLIST
# ============================================================
def generate_scoring(df: pd.DataFrame, idx_test, y_proba) -> pd.DataFrame:
    """
    Формирование итогового ранжированного списка заявителей.

    Категории рекомендаций:
      ✅ Рекомендовать   — score >= 0.80
      ⚠️  На рассмотрение — score >= 0.50
      ❌ Отклонить       — score < 0.50

    Финальное решение остаётся за комиссией.
    """
    df_scored = df.loc[idx_test].copy()
    df_scored['score'] = y_proba
    df_scored['score_pct'] = (y_proba * 100).round(1)
    df_scored['рекомендация'] = df_scored['score'].apply(
        lambda x: '✅ Рекомендовать' if x >= 0.80
        else ('⚠️ На рассмотрение' if x >= 0.50
              else '❌ Отклонить')
    )

    # Ранг внутри каждой области
    df_scored['ранг_в_области'] = df_scored.groupby('Область')['score'].rank(
        ascending=False, method='min'
    ).astype(int)

    df_scored = df_scored.sort_values('score', ascending=False).reset_index(drop=True)

    print("\n" + "="*60)
    print("🏆 ИТОГОВЫЙ СКОРИНГ")
    print("="*60)
    print(df_scored['рекомендация'].value_counts().to_string())

    return df_scored


# ============================================================
# 7. ОБЪЯСНЕНИЕ РЕШЕНИЙ (Explainability без SHAP)
# ============================================================
def explain_decision(row: pd.Series, model, feature_cols: list, feature_values: np.ndarray) -> str:
    """
    Простое объяснение решения для одной заявки.
    Использует feature importance модели.
    """
    importance = model.feature_importances_
    contributions = []

    feat_labels = {
        'log_сумма': f"сумма {row['Причитающая сумма']:,.0f} тг",
        'log_норматив': f"норматив {row['Норматив']:,.0f} тг/гол",
        'log_поголовье': f"поголовье ~{row.get('поголовье', 0):,.0f} гол",
        'Область_enc': f"область: {row['Область']}",
        'Направление водства_enc': f"направление: {row['Направление водства']}",
        'Наименование субсидирования_enc': "тип субсидии",
        'Район хозяйства_enc': f"район: {row['Район хозяйства']}",
    }

    explanation = []
    for feat, imp in sorted(zip(feature_cols, importance), key=lambda x: -x[1])[:3]:
        explanation.append(f"{feat_labels.get(feat, feat)} (вес {imp*100:.1f}%)")

    return "Ключевые факторы: " + " | ".join(explanation)


# ============================================================
# MAIN
# ============================================================
def main(data_path: str, output_path: str = 'outputs/scoring_results.csv'):
    print("\n🌾 AgroScore KZ — Запуск скоринга")
    print("="*60)

    # Пайплайн
    df = load_data(data_path)
    df = create_target(df)
    df, feature_cols = engineer_features(df)
    model, X_test, y_test, idx_test = train_model(df, feature_cols)
    y_proba, auc = evaluate_model(model, X_test, y_test, feature_cols)
    df_scored = generate_scoring(df, idx_test, y_proba)

    # Сохраняем результаты
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    output_cols = [
        'Номер заявки', 'Область', 'Район хозяйства',
        'Направление водства', 'Причитающая сумма',
        'Норматив', 'Статус заявки',
        'score', 'score_pct', 'рекомендация', 'ранг_в_области'
    ]
    available = [c for c in output_cols if c in df_scored.columns]
    df_scored[available].to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n✅ Результаты сохранены: {output_path}")
    print(f"   ROC-AUC модели: {auc:.4f}")
    print(f"   Всего оценено заявок: {len(df_scored):,}")
    print("\n⚠️  Финальное решение остаётся за комиссией.")
    print("="*60)

    return df_scored, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AgroScore KZ — скоринг сельхозпроизводителей')
    parser.add_argument('--data', type=str,
                        default='data/raw/subsidies_2025.xlsx',
                        help='Путь к файлу с реестром заявок (.xlsx)')
    parser.add_argument('--output', type=str,
                        default='outputs/scoring_results.csv',
                        help='Путь для сохранения результатов')
    args = parser.parse_args()

    df_scored, model = main(args.data, args.output)
