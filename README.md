# fraud-prevention-rsrch

Инструментарий для оценки алгоритмов машинного обучения в антифрод-системах.  
Библиотека реализует полный цикл: синтез данных → оценка устойчивости к сдвигам распределений → интерпретация ответов моделей.

---

## Содержание

- [Структура репозитория](#структура-репозитория)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Модули](#модули)
  - [transaction_gan.py](#transaction_ganpy)
  - [shift_generator.py](#shift_generatorpy)
  - [robustness_metrics.py](#robustness_metricspy)
  - [shift_visualizer.py](#shift_visualizerpy)
  - [interpretability_tool.py](#interpretability_toolpy)
- [Методика оценки эффективности](#методика-оценки-эффективности)
- [Пример полного пайплайна](#пример-полного-пайплайна)

---

## Структура репозитория

```
fraud-prevention-rsrch/
├── transaction_gan.py        # Условная GAN для генерации синтетических транзакций
├── shift_generator.py        # Генератор синтетических сдвигов распределений
├── robustness_metrics.py     # Метрики устойчивости и дрейфа
├── shift_visualizer.py       # Визуализация результатов оценки
├── interpretability_tool.py  # Интерпретация ответов моделей (SHAP, LIME)
└── minimal_pipeline.py       # Демонстрационный скрипт
```

---

## Установка

```bash
pip install numpy pandas scikit-learn scipy matplotlib torch shap lime
```

**Минимальные зависимости** (без SHAP/LIME и PyTorch):
```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

---

## Быстрый старт

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from shift_generator import ShiftPipeline
from robustness_metrics import RobustnessEvaluator, RobustnessReport
from shift_visualizer import ShiftVisualizer

# 1. Данные и модель
X, y = make_classification(n_samples=2000, weights=[0.88, 0.12], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = RandomForestClassifier().fit(X_train, y_train)
predict_fn = lambda X: model.predict_proba(X)[:, 1]

# 2. Генерация сдвигов
pipeline = ShiftPipeline()
shifts = pipeline.apply_sweep(X_test, y_test, intensities=[0.1, 0.3, 0.5, 0.7, 1.0])

# 3. Оценка устойчивости
evaluator = RobustnessEvaluator(model, predict_fn, X_test, y_test)
report = RobustnessReport(evaluator.evaluate_all(shifts))
report.to_csv("robustness_report.csv")

# 4. Визуализация
viz = ShiftVisualizer(report, output_dir="plots")
viz.plot_performance_curves()
viz.plot_degradation_bar()
viz.save_all()
```

---

## Модули

### transaction_gan.py

Условная генеративно-состязательная сеть (Conditional GAN) для генерации синтетических транзакционных данных с заданным соотношением классов.

**Архитектура:**
- Генератор: `[шум (100) + метка класса]` → 5 FC-слоёв (512→1024→2048→1024→N) → Tanh
- Дискриминатор: `[признаки + метка]` → 4 FC-слоя (1024→512→256→128→1) → Sigmoid
- Оптимизатор: Adam, lr=0.0002, β₁=0.5; функция потерь: BCE

**Ключевые классы:**

| Класс / функция | Описание |
|---|---|
| `TransactionDataset(csv_path)` | Загрузка и предобработка датасета (one-hot, StandardScaler) |
| `TransactionGAN(latent_dim, data_dim)` | Инициализация GAN |
| `TransactionGAN.train(dataloader, epochs)` | Обучение |
| `TransactionGAN.generate_samples(n, label)` | Генерация N записей заданного класса |

**Пример:**
```python
from transaction_gan import TransactionDataset, TransactionGAN
from torch.utils.data import DataLoader

dataset = TransactionDataset("transactions.csv", label_column="isFraud")
loader  = DataLoader(dataset, batch_size=256, shuffle=True)

gan = TransactionGAN(latent_dim=100, data_dim=dataset.features.shape[1])
gan.train(loader, num_epochs=200)

# Генерация 10 000 записей, 10% мошеннических
fraud, lf = gan.generate_samples(1000, label_value=1)
legit, ll = gan.generate_samples(9000, label_value=0)
```

---

### shift_generator.py

Генератор синтетических сдвигов распределений. Все генераторы реализуют единый интерфейс `apply(X, y, intensity)`, где `intensity ∈ [0, 1]`.

**Типы сдвигов:**

| Тип | Класс | Что меняется |
|---|---|---|
| Ковариационный | `CovariateShiftGenerator` | P(X), логика мошенничества неизменна |
| Сдвиг меток | `LabelShiftGenerator` | P(Y), соотношение классов |
| Концептуальный дрейф | `ConceptDriftGenerator` | P(Y\|X), связь признаков с метками |
| Состязательный | `AdversarialShiftGenerator` | Целенаправленное возмущение признаков |

**Ключевые структуры:**
```python
@dataclass
class ShiftConfig:
    shift_type: str    # "covariate" | "label" | "concept" | "adversarial"
    intensity: float   # 0.0..1.0
    mode: str          # sub-режим генератора

@dataclass
class ShiftedDataset:
    X: np.ndarray
    y: np.ndarray
    config: ShiftConfig
    metadata: dict     # actual_fraud_rate, n_flipped, perturbation_l2, …
```

**Режимы CovariateShiftGenerator:**
- `gaussian_noise` — гауссов шум N(0, intensity·σ) по числовым признакам *(по умолчанию)*
- `scaling` — мультипликативное масштабирование на 1 + intensity·U(−1,1)
- `correlation` — поворот пространства признаков через QR-разложение

**Режимы ConceptDriftGenerator:**
- `random_flip` — случайное переключение меток *(по умолчанию)*
- `boundary_flip` — переключение меток у объектов с уверенностью ≈ 0.5

**AdversarialShiftGenerator:**
- PyTorch-модель → FGSM: `X_adv = X + ε·sign(∇_X L)`
- sklearn-модель → статистическая атака: сдвиг к центроиду противоположного класса

**Пример:**
```python
from shift_generator import ShiftPipeline, CovariateShiftGenerator

# Одиночный генератор
gen = CovariateShiftGenerator()
shifted = gen.apply(X, y, intensity=0.5, mode="gaussian_noise")

# Пайплайн по всем типам и нескольким интенсивностям
pipeline = ShiftPipeline()
all_shifts = pipeline.apply_sweep(
    X, y,
    intensities=[0.1, 0.3, 0.5, 0.7, 1.0],
    generator_kwargs={"concept": {"mode": "boundary_flip"}}
)
# all_shifts["covariate"] → list[ShiftedDataset]
```

---

### robustness_metrics.py

Количественная оценка влияния сдвигов на качество модели. Включает метрики дрейфа распределений и метрики качества модели.

**Метрики дрейфа распределений** (per-feature):

| Функция | Метрика | Интерпретация |
|---|---|---|
| `psi(base, shifted)` | PSI | < 0.10 стабильно · 0.10–0.25 умеренно · > 0.25 значимо |
| `ks_statistic(base, shifted)` | KS-статистика + p-value | Максимальное расхождение CDF |
| `wasserstein_distance(base, shifted)` | Расстояние Вассерштейна | «Стоимость» переноса распределения |
| `js_divergence(base, shifted)` | JS-дивергенция | Симметричная мера расхождения ∈ [0, 1] |
| `compute_distribution_metrics(X_base, X_shifted)` | DataFrame всех метрик | feature × {psi, ks_stat, wasserstein, js} |

**Метрики качества модели:**

| Функция | Описание |
|---|---|
| `compute_performance_metrics(y_true, y_prob)` | AUC-ROC, PR-AUC, F1, Precision, Recall |
| `relative_degradation(baseline, shifted)` | RD = (m_shifted − m_base) / \|m_base\| · 100% |

**Ключевые классы:**

```python
# Оценка одной модели
evaluator = RobustnessEvaluator(
    model=model,
    predict_fn=predict_fn,      # callable: X → probabilities
    X_baseline=X_test,
    y_baseline=y_test,
    feature_names=feature_names,
)
result  = evaluator.evaluate(shifted_dataset)        # → RobustnessResult
results = evaluator.evaluate_all(shifts_dict)        # → list[RobustnessResult]

# Отчёт
report = RobustnessReport(results)
df     = report.summary_table()                      # shift_type × intensity × metric
report.to_csv("report.csv")
critical = report.flag_critical("avg_precision", threshold=10.0)
```

**RobustnessResult** содержит:
- `shift_type`, `intensity`, `mode`
- `performance` — dict метрик качества
- `degradation` — dict относительной деградации (%)
- `distribution_drift` — DataFrame per-feature метрик дрейфа

---

### shift_visualizer.py

Визуализация результатов оценки устойчивости.

**Класс `ShiftVisualizer`:**

```python
viz = ShiftVisualizer(report, output_dir="plots")
```

| Метод | График |
|---|---|
| `plot_performance_curves(metrics)` | Кривые метрик vs intensity, одна линия на тип сдвига |
| `plot_degradation_bar(metric)` | Столбчатая диаграмма деградации при max intensity |
| `plot_distribution_shifts(X_base, X_shifted, feature_names, top_n)` | Гистограммы топ-N признаков по PSI |
| `plot_psi_heatmap(drift_results)` | Тепловая карта PSI: признак × тип сдвига |
| `plot_model_comparison_heatmap(reports_dict, metric)` | Сравнение моделей: модель × тип сдвига |
| `plot_radar_chart(reports_dict, metric, intensity)` | Радарная диаграмма устойчивости |
| `save_all(formats)` | Сохранение всех накопленных фигур |

---

### interpretability_tool.py

Анализ и интерпретация ответов моделей. Поддерживает глобальные и локальные объяснения.

**Поддерживаемые методы:**

| Метод | Тип | Описание |
|---|---|---|
| `shap` | Глобальный / локальный | SHAP KernelExplainer, важность признаков |
| `lime` | Локальный | Локальные линейные приближения |
| `permutation` | Глобальный | Важность через перестановку признаков |
| `pdp` | Глобальный | Partial Dependence Plot |

**Ключевые функции и классы:**

```python
from interpretability_tool import run_experiment, run_interpretability_suite, ReportBuilder

# Запуск нескольких методов на одной модели
results = run_experiment(
    model=model,
    X=X_test,
    y=y_test,
    methods=["shap", "lime", "permutation"],
    predict_fn=predict_fn,           # для нейросетей
)

# Полный набор метрик интерпретируемости
suite = run_interpretability_suite(
    model=model, X=X_test, y=y_test,
    methods=["shap", "lime"],
    lime_fp_threshold=0.6,           # порог для отбора ложноположительных
    lime_max_examples=3,
)

# Сводный отчёт
builder = ReportBuilder()
table   = builder.summary_table(results)
builder.to_csv(table, "interpretability_report.csv")
```

**Метрики качества объяснений** (MetricEngine):
- **Fidelity** — насколько объяснение точно воспроизводит поведение модели
- **Stability** — устойчивость объяснений при малых возмущениях входа
- **Sparsity** — компактность: доля нулевых/незначимых коэффициентов

**Визуализация** (InterpretabilityVisualizer):
```python
from interpretability_tool import InterpretabilityVisualizer

viz = InterpretabilityVisualizer(output_dir="interp_plots")
viz.plot_shap_global(explanation, feature_names, top_n=15, model_name="RF")
viz.plot_lime_local(explanation, feature_names, top_n=10, model_name="RF")
viz.plot_method_comparison(explanations_dict, feature_names)
viz.plot_metrics_table(metrics_results, model_name="RF")
viz.save_all()
```

---

## Методика оценки эффективности

### Общий подход

Методика направлена на оценку устойчивости антифрод-моделей к изменениям условий эксплуатации. Включает три уровня анализа:

1. **Базовое качество** — оценка метрик модели на чистых данных без сдвигов
2. **Устойчивость к сдвигам** — количественная оценка деградации при каждом типе сдвига
3. **Интерпретируемость** — анализ важности признаков и качества объяснений

### Шаг 1. Подготовка данных

Рекомендуется использовать реальные транзакционные данные или синтетические данные, сгенерированные `transaction_gan.py`. Ключевые требования:
- Наличие бинарной целевой переменной (0 — легитимная, 1 — мошенническая)
- Дисбаланс классов
- Фиксированное разбиение train/test (`random_state` для воспроизводимости)

### Шаг 2. Обучение базовой модели

Модель обучается на тренировочной выборке. Базовые метрики фиксируются на тестовой выборке **без применения каких-либо сдвигов**.

Рекомендуемые метрики для антифрод-задачи (в порядке приоритета):
1. **PR-AUC** — приоритетная метрика для несбалансированных данных; чувствительна к качеству обнаружения мошенничества
2. **F1** — баланс между точностью и полнотой

### Шаг 3. Генерация сдвигов

Применить `ShiftPipeline.apply_sweep()` с интенсивностями `[0.1, 0.2, 0.3, 0.5, 0.7, 1.0]`.

| Тип сдвига | Имитируемый сценарий |
|---|---|
| Ковариационный | Изменение признаков транзакций: новые суммы, каналы, география |
| Сдвиг меток | Рост доли мошенничества: атаки, сезонность, утечки данных |
| Концептуальный | Смена схем мошенничества: новые методы, адаптация злоумышленников |
| Состязательный | Целенаправленная маскировка мошеннических транзакций под легитимные |

### Шаг 4. Оценка устойчивости

`RobustnessEvaluator` вычисляет для каждого сдвига:

- **Метрики дрейфа признаков** (PSI, KS, Wasserstein, JS) — фиксируют изменение входных данных
- **Метрики качества модели** (AUC-ROC, PR-AUC, F1) — фиксируют деградацию предсказаний
- **Относительную деградацию RD** — позволяет сравнивать разные метрики и модели

> Ключевое наблюдение: метрики дрейфа признаков не обнаруживают концептуальный дрейф (он меняет только метки). Это обосновывает необходимость мониторинга обеих групп метрик в production.

### Шаг 5. Интерпретация результатов

Пороги для практических решений:

| Условие | Рекомендуемое действие |
|---|---|
| PSI > 0.25 по ключевому признаку | Проверить источник данных, инициировать переобучение |
| PR-AUC деградировал > 10% | Инициировать переобучение модели |
| AUC-ROC деградировал > 20% | Немедленное переобучение или замена модели |
| Состязательный сдвиг AUC-ROC → 0 | Модель полностью нейтрализована — применить adversarial training |

### Шаг 6. Анализ интерпретируемости

Использовать `run_interpretability_suite()` для получения:
- Глобальной важности признаков (SHAP, Permutation)
- Локальных объяснений для ложноположительных срабатываний (LIME)
- Метрик качества объяснений (fidelity, stability, sparsity)

---

## Пример полного пайплайна

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from shift_generator import ShiftPipeline
from robustness_metrics import RobustnessEvaluator, RobustnessReport
from shift_visualizer import ShiftVisualizer
from interpretability_tool import run_interpretability_suite, ReportBuilder

# ── Данные ──────────────────────────────────────────────────────────────────
X, y = make_classification(
    n_samples=4000, n_features=30, n_informative=15,
    weights=[0.88, 0.12], random_state=42,
)
feature_names = [f"feat_{i:02d}" for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ── Модель ───────────────────────────────────────────────────────────────────
model = GradientBoostingClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
predict_fn = lambda X: model.predict_proba(X)[:, 1]

# ── Сдвиги и оценка устойчивости ─────────────────────────────────────────────
shifts   = ShiftPipeline().apply_sweep(X_test, y_test)
evaluator = RobustnessEvaluator(model, predict_fn, X_test, y_test, feature_names)
report   = RobustnessReport(evaluator.evaluate_all(shifts))
report.to_csv("robustness_report.csv")

# ── Визуализация ──────────────────────────────────────────────────────────────
viz = ShiftVisualizer(report, output_dir="plots")
viz.plot_performance_curves(metrics=["avg_precision", "f1"])
viz.plot_degradation_bar(metric="avg_precision")
viz.save_all()

# ── Интерпретируемость ────────────────────────────────────────────────────────
suite   = run_interpretability_suite(model, X_test, y_test, methods=["shap", "lime"])
builder = ReportBuilder()
builder.to_csv(builder.summary_table(suite), "interpretability_report.csv")

print("Готово. Отчёты: robustness_report.csv, interpretability_report.csv")
print("Графики: plots/")
```

---

## Цитирование

Работа выполнена в рамках НИР «Методика оценки алгоритмов машинного обучения в антифрод-системах».

Использованные методы:
- SHAP: Lundberg & Lee, NeurIPS 2017
- LIME: Ribeiro et al., KDD 2016
- FGSM: Goodfellow et al., ICLR 2015
- PSI: стандарт банковской отрасли (Basel II)
- Dataset Shift: Moreno-Torres et al., Pattern Recognition 2012
