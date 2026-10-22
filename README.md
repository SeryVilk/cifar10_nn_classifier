# cifar10_nn_classifier

Классификатор изображений на датасете **CIFAR-10** на основе свёрточной нейронной сети (CNN) в TensorFlow/Keras. Реализация - в Jupyter Notebook.

## 📋 О проекте

Ноутбук `cifar10_cnn.ipynb` обучает CNN распознавать 10 классов объектов CIFAR-10:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

Модель после обучения сохраняется в `my_model.keras` и переиспользуется при повторном запуске — повторное обучение не требуется.

## ✨ Возможности

- Загрузка и нормализация датасета CIFAR-10 через `tensorflow.keras.datasets`
- CNN с `BatchNormalization` и `Dropout` (защита от переобучения)
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`
- Сохранение/загрузка модели (`my_model.keras`)
- Графики функции потерь и точности
- Оценка на тестовой выборке
- Визуализация верных и ошибочных предсказаний
- Предсказания на случайных изображениях из теста

## 🧠 Архитектура

```
Input (32×32×3)
├── Conv2D(32) × 2 + BN
├── MaxPooling2D + Dropout(0.25)
├── Conv2D(64) × 2 + BN
├── MaxPooling2D + Dropout(0.25)
├── Conv2D(128) × 2 + BN
├── MaxPooling2D + Dropout(0.25)
├── Conv2D(256) × 2 + BN
├── MaxPooling2D + Dropout(0.25)
├── Flatten
├── Dense(512) + Dropout(0.5)
├── Dense(256) + Dropout(0.5)
└── Dense(10, softmax)
```

- **Оптимизатор**: Adam (lr = 0.001)
- **Loss**: categorical crossentropy
- **Batch size**: 64
- **Эпохи**: 15 (с EarlyStopping, patience=5)

## ⚙️ Установка

Требуется **Python 3.9+**.

```bash
pip install -r requirements.txt
```

Для GPU-ускорения — `tensorflow[and-cuda]` (Linux/Windows) или `tensorflow-metal` (macOS Apple Silicon).

## 🚀 Запуск

Открой `cifar10_cnn.ipynb` в PyCharm / Jupyter и выполни ячейки сверху вниз (**Run All**).

При первом запуске:
1. Скачается CIFAR-10 (~170 МБ)
2. Обучится модель (или загрузится `my_model.keras`, если уже существует)
3. Сохранятся `training_history.png` и `predictions_examples.png`
4. Выведется точность на тесте и примеры предсказаний

## 📊 Ожидаемые результаты

Точность на тестовой выборке ≈ **80–85%** (зависит от эпох и железа).

```
Test Loss: 0.5097
Test Accuracy: 0.8320
```

## 📁 Структура

```
cifar10_nn_classifier/
├── cifar10_cnn.ipynb          # Основной ноутбук
├── requirements.txt
├── README.md
├── .gitignore
```

## 🛠 Технологии

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook