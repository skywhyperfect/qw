# 🌾 AgroScore KZ — Интеллектуальный скоринг субсидий

> **Decentrathon 5.0 · Кейс 2 · inDrive (Этап 2 — Прототип)**
> Система перехода от принципа «кто успел» к **Merit-based** распределению субсидий в животноводстве Казахстана.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-blue)](https://github.com/slundberg/shap)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.81-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📺 Демонстрация

| | |
|---|---|
| 🖥️ **Веб-интерфейс** | Доступен локально через Streamlit (`streamlit run app.py`) |
| 🎥 **Видео-демо** | [➡️ ССЫЛКА НА ВИДЕО](https://drive.google.com/drive/folders/12YZRy4URXzlEAsUPpbic_Yjc7p2Q7SNV?usp=sharing) |

---

## 🚀 Что нового в Этапе 2?

Согласно техническому заданию, реализован полноценный рабочий прототип:

| # | Функция | Описание |
|---|---------|----------|
| 1 | 🖥️ **Web UI (Streamlit)** | Интерактивный дашборд для загрузки реестров и визуализации шортлистов |
| 2 | 🔍 **Explainable AI (SHAP)** | Детальное объяснение каждого решения модели — больше никакого «чёрного ящика» |
| 3 | 💾 **Сохранение артефактов** | Модель и энкодеры сериализованы (`.pkl`) — система работает без переобучения |

---

## 🧠 Технический стек

| Компонент | Технология |
|-----------|------------|
| **Язык** | Python 3.10+ |
| **Модель** | Gradient Boosting Classifier |
| **Качество** | ROC-AUC: **0.81** |
| **UI** | Streamlit |
| **Объяснимость** | SHAP (Shapley Additive Explanations) |
| **Данные** | 36 651 запись из `subsidy.plem.kz` |

**Ключевые признаки модели:**
- Регион (Область / Район)
- Тип субсидии
- Экономический масштаб (Поголовье / Сумма)

---

## 🛠️ Инструкция по запуску

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/agroscore-kz.git
cd agroscore-kz
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Обучение модели (опционально)

Если нужно переобучить модель на новых данных:

```bash
python scoring.py --data data/raw/subsidies_2025.xlsx
```

> Скрипт создаст папку `models/` с необходимыми весами и энкодерами.

### 4. Запуск веб-интерфейса

```bash
streamlit run app.py
```

Откройте браузер по адресу `http://localhost:8501`

---

## 🔍 Объяснимость решений (Explainability)

Реализовано требование **Explainability** с использованием значений SHAP (Shapley Additive Explanations).

В интерфейсе для каждой заявки строится **график влияния факторов**:

```
Заявка #1234  →  Балл: 78 / 100  →  ✅ Рекомендовано к одобрению

Факторы влияния:
  🟢 Высокое поголовье (+12.4)
  🟢 Регион: Алматинская обл. (+8.1)
  🔴 Тип субсидии: несоответствие норматива (-5.3)
  🔴 История выплат: неполная (-2.7)
```

| Цвет | Значение |
|------|----------|
| 🟢 Зелёный | Фактор **повышает** вероятность одобрения |
| 🔴 Красный | Фактор **снижает** итоговый балл |

---

## 🏗️ Структура проекта

```
📂 agroscore-kz/
├── 📂 data/
│   └── 📂 raw/               # Исходный датасет (.xlsx)
├── 📂 models/                # Сохранённые модели и энкодеры (.pkl)
├── 📂 outputs/               # Результаты скоринга в CSV
├── app.py                    # Веб-интерфейс (Streamlit)
├── scoring.py                # Логика обучения и обработки данных
├── requirements.txt          # Зависимости проекта
└── README.md                 # Документация
```

---

## 🗺️ Roadmap

- [x] **Этап 1** — Baseline модель и разведочный анализ данных (EDA)
- [x] **Этап 2** — Рабочий прототип, Web UI, SHAP integration ← *текущий этап*
- [ ] **Этап 3** — Финальная презентация и масштабирование

---

## 📦 Зависимости

Основные библиотеки из `requirements.txt`:

```
streamlit
scikit-learn
pandas
numpy
shap
openpyxl
joblib
matplotlib
```

---

## 📄 Лицензия и команда

| | |
|---|---|
| 👥 **Команда** | [skyrocket] |
| 📋 **Лицензия** | [MIT](LICENSE) |
| 🏛️ **Задача** | AI for Government — inDrive / Правительство РК |
| 📎 **Форма сдачи** | [forms.gle/cXPKUVinCWayZJCRA](https://forms.gle/cXPKUVinCWayZJCRA) |

---

<div align="center">

**AgroScore KZ** — честное распределение субсидий через силу данных

*Decentrathon 5.0 · 2025*

</div>
