from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xlrd
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Результаты_анализа"
SEED = 31
TEST_SIZE = 5
ALPHA = 0.05

FILES = {
    "innovation": ROOT / "01_Затраты_на_инновации_Росстат.xls",
    "grp": ROOT / "02_ВРП_Росстат.xlsx",
    "rd": ROOT / "03_Внутренние_затраты_на_НИР_Росстат.xlsx",
}

AGGREGATE_KEYS = {
    "архангельская область",
    "тюменская область",
}

# Качественный фактор: сторона света. Признак строится по координатам
# административных центров регионов (широта/долгота, градусы):
#   Север  — широта не ниже 60° с. ш.;
#   Восток — остальные регионы с долготой не менее 60° в. д.;
#   Юг     — остальные регионы с широтой менее 50° с. ш.;
#   Запад  — остальные регионы европейской части (базовая категория).
# Для Ленинградской области и Московской области использованы координаты
# Санкт-Петербурга и Москвы соответственно.
COORDS = {
    "Алтайский край": (53.35, 83.78),
    "Амурская область": (50.29, 127.53),
    "Астраханская область": (46.35, 48.04),
    "Белгородская область": (50.60, 36.59),
    "Брянская область": (53.24, 34.36),
    "Владимирская область": (56.13, 40.41),
    "Волгоградская область": (48.71, 44.51),
    "Вологодская область": (59.22, 39.89),
    "Воронежская область": (51.67, 39.18),
    "Забайкальский край": (52.03, 113.50),
    "Ивановская область": (57.00, 40.97),
    "Иркутская область": (52.29, 104.28),
    "Кабардино-Балкарская Республика": (43.48, 43.62),
    "Калининградская область": (54.71, 20.51),
    "Калужская область": (54.51, 36.26),
    "Камчатский край": (53.02, 158.65),
    "Карачаево-Черкесская Республика": (44.22, 42.06),
    "Кемеровская область": (55.35, 86.09),
    "Кировская область": (58.60, 49.66),
    "Костромская область": (57.77, 40.93),
    "Краснодарский край": (45.04, 38.98),
    "Красноярский край": (56.01, 92.87),
    "Курганская область": (55.44, 65.34),
    "Курская область": (51.73, 36.19),
    "Ленинградская область": (59.94, 30.31),
    "Липецкая область": (52.61, 39.60),
    "Магаданская область": (59.56, 150.80),
    "Московская область": (55.76, 37.62),
    "Мурманская область": (68.97, 33.08),
    "Нижегородская область": (56.33, 44.00),
    "Новгородская область": (58.52, 31.27),
    "Новосибирская область": (55.03, 82.92),
    "Омская область": (54.99, 73.37),
    "Оренбургская область": (51.77, 55.10),
    "Орловская область": (52.97, 36.06),
    "Пензенская область": (53.20, 45.00),
    "Пермский край": (58.01, 56.25),
    "Приморский край": (43.12, 131.89),
    "Псковская область": (57.82, 28.33),
    "Республика Адыгея (Адыгея)": (44.61, 40.10),
    "Республика Алтай": (51.96, 85.96),
    "Республика Башкортостан": (54.74, 55.97),
    "Республика Бурятия": (51.83, 107.58),
    "Республика Дагестан": (42.98, 47.50),
    "Республика Ингушетия": (43.17, 44.81),
    "Республика Калмыкия": (46.31, 44.27),
    "Республика Карелия": (61.79, 34.35),
    "Республика Коми": (61.67, 50.84),
    "Республика Крым": (44.95, 34.10),
    "Республика Марий Эл": (56.63, 47.90),
    "Республика Мордовия": (54.18, 45.18),
    "Республика Саха (Якутия)": (62.03, 129.73),
    "Республика Северная Осетия - Алания": (43.04, 44.67),
    "Республика Татарстан (Татарстан)": (55.79, 49.12),
    "Республика Тыва": (51.72, 94.45),
    "Республика Хакасия": (53.72, 91.43),
    "Ростовская область": (47.23, 39.72),
    "Рязанская область": (54.63, 39.74),
    "Самарская область": (53.20, 50.15),
    "Саратовская область": (51.53, 46.03),
    "Сахалинская область": (46.96, 142.74),
    "Свердловская область": (56.84, 60.61),
    "Смоленская область": (54.78, 32.05),
    "Ставропольский край": (45.04, 41.97),
    "Тамбовская область": (52.72, 41.45),
    "Тверская область": (56.86, 35.91),
    "Томская область": (56.50, 84.97),
    "Тульская область": (54.20, 37.62),
    "Удмуртская Республика": (56.85, 53.20),
    "Ульяновская область": (54.32, 48.40),
    "Хабаровский край": (48.48, 135.07),
    "Челябинская область": (55.16, 61.40),
    "Чеченская Республика": (43.32, 45.70),
    "Чувашская Республика -Чувашия": (56.13, 47.25),
    "Ямало-Ненецкий автономный округ": (66.53, 66.63),
    "Ярославская область": (57.63, 39.87),
    "г. Москва": (55.76, 37.62),
    "г. Санкт-Петербург": (59.94, 30.31),
    "г. Севастополь": (44.62, 33.53),
    # Территории, которые присутствуют в исходной таблице инноваций, но
    # исключаются из-за неполного сопоставления с ВРП/НИР (координаты нужны
    # только для прохождения проверки при чтении источника).
    "Ненецкий автономный округ": (67.64, 53.01),
    "Ханты-Мансийский автономный округ - Югра": (61.00, 69.02),
    "Тюменская область без автономных округов": (57.15, 65.53),
    "Еврейская автономная область": (48.79, 132.92),
    "Чукотский автономный округ": (64.73, 177.51),
}
SIDE_ORDER = ["Запад", "Север", "Юг", "Восток"]
BASE_SIDE = "Запад"


def side_by_coords(lat: float, lon: float) -> str:
    if lat >= 60:
        return "Север"
    if lon >= 60:
        return "Восток"
    if lat < 50:
        return "Юг"
    return "Запад"

ALIASES = {
    "республика адыгея адыгея": "республика адыгея",
    "адыгея": "республика адыгея",
    "республика татарстан татарстан": "республика татарстан",
    "татарстан": "республика татарстан",
    "чувашская республика чувашия": "чувашская республика",
    "чувашская республика чувашия": "чувашская республика",
    "республика северная осетия алания": "республика северная осетия алания",
    "кемеровская область кузбасс": "кемеровская область",
    "г москва": "москва",
    "москва": "москва",
    "г санкт петербург": "санкт петербург",
    "санкт петербург": "санкт петербург",
    "архангельская область без ненецкого авт округа": "архангельская область без ао",
    "архангельская область кроме ненецкого автономного округа": "архангельская область без ао",
    "архангельская область без ао": "архангельская область без ао",
    "тюменская область без автономных округов": "тюменская область без ао",
    "тюменская область без ханты мансийского авт округа югра и ямало ненецкого авт округа": "тюменская область без ао",
    "тюменская область без ао": "тюменская область без ао",
    "ханты мансийский авт округ югра": "ханты мансийский авт округ югра",
    "республика северная осетия алания": "республика северная осетия алания",
    "калинингpадская область": "калининградская область",
}


def normalized_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    text = re.sub(r"\bг\.?\s*", "г ", text)
    text = re.sub(r"\bавтономного округа\b", "авт округа", text)
    text = re.sub(r"\bавтономный округ\b", "авт округ", text)
    text = re.sub(r"\bавтономного округа\b", "авт округа", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\bбез ханты мансийского авт округа югра и ямало ненецкого авт округа\b", "без ао", text)
    text = re.sub(r"\bбез ненецкого авт округа\b", "без ао", text)
    text = re.sub(r"\bбез автономных округов\b", "без ао", text)
    text = re.sub(r"\bбез ао\b", "без ао", text)
    text = re.sub(r"\bв том числе\b", " ", text)
    text = re.sub(r"^в т ч\s+", "", text)
    text = re.sub(r"\d+\)?", " ", text)
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    text = " ".join(text.split())
    return ALIASES.get(text, text)


def numeric_value(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(" ", " ").replace(" ", "")
    if not text:
        return np.nan
    if text in {"-", "–", "—"}:
        return 0.0
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_innovation() -> pd.DataFrame:
    book = xlrd.open_workbook(FILES["innovation"])
    sheet = book.sheet_by_name("5")
    rows: list[dict[str, object]] = []
    district = ""
    for row_idx in range(7, sheet.nrows):
        raw_name = sheet.cell_value(row_idx, 0)
        name = str(raw_name).strip()
        key = normalized_name(name)
        if not key or key == "российская федерация":
            continue
        if "федеральный округ" in key:
            district = re.sub(r"\s*\d+\)?$", "", name).strip()
            continue
        if "в том числе" in key or key in AGGREGATE_KEYS:
            continue
        value = numeric_value(sheet.cell_value(row_idx, 1))
        if not np.isfinite(value):
            continue
        if name not in COORDS:
            raise ValueError(f"Нет координат административного центра для региона: «{name}»")
        lat, lon = COORDS[name]
        rows.append({
            "region_key": key,
            "region": name,
            "district": district,
            "side": side_by_coords(lat, lon),
            "lat": lat,
            "lon": lon,
            "innovation_mln": value / 1000,
            "source_row_y": row_idx + 1,
        })
    frame = pd.DataFrame(rows)
    if frame["region_key"].duplicated().any():
        duplicates = frame.loc[frame["region_key"].duplicated(False), "region"].tolist()
        raise ValueError(f"Повторные территории в исходном Y: {duplicates}")
    return frame


def read_xlsx_table(path: Path, sheet_name: str, name_col: str, value_col: str,
                    start_row: int, variable: str) -> pd.DataFrame:
    sheet = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    rows: list[dict[str, object]] = []
    for idx in range(start_row - 1, len(sheet)):
        raw_name = sheet.iloc[idx, ord(name_col.upper()) - ord("A")]
        raw_value = sheet.iloc[idx, ord(value_col.upper()) - ord("A")]
        key = normalized_name(raw_name)
        if not key or key == "российская федерация" or "федеральный округ" in key:
            continue
        if "в том числе" in key or key in AGGREGATE_KEYS:
            continue
        value = numeric_value(raw_value)
        if not np.isfinite(value):
            continue
        rows.append({"region_key": key, variable: value, f"source_row_{variable}": idx + 1})
    frame = pd.DataFrame(rows)
    if frame["region_key"].duplicated().any():
        duplicates = frame.loc[frame["region_key"].duplicated(False), "region_key"].tolist()
        raise ValueError(f"Повторные территории в {path.name}: {duplicates}")
    return frame


def build_dataset() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    for path in FILES.values():
        if not path.exists():
            raise FileNotFoundError(f"Не найден исходный файл: {path.name}")

    y = read_innovation()
    grp = read_xlsx_table(FILES["grp"], "1", "A", "R", 6, "grp_mln")
    rd = read_xlsx_table(FILES["rd"], "2", "A", "F", 5, "rd_mln")

    audit: dict[str, list[str]] = {}
    for name, source in (("grp", grp), ("rd", rd)):
        audit[f"unmatched_y_{name}"] = sorted(set(y.region_key) - set(source.region_key))
        audit[f"unmatched_{name}_y"] = sorted(set(source.region_key) - set(y.region_key))

    data = y.merge(grp, on="region_key", how="left", validate="one_to_one")
    data = data.merge(rd, on="region_key", how="left", validate="one_to_one")
    missing = data[["innovation_mln", "grp_mln", "rd_mln", "district", "side", "lat", "lon"]].isna().any(axis=1)
    audit["incomplete_regions"] = data.loc[missing, "region"].tolist()
    data = data.loc[~missing].copy()
    data = data.sort_values("region").reset_index(drop=True)
    data["side"] = pd.Categorical(data["side"], categories=SIDE_ORDER, ordered=False)

    if len(data) <= TEST_SIZE + 12:
        raise ValueError(f"После объединения осталось мало наблюдений: n={len(data)}")
    if data["region"].duplicated().any():
        raise ValueError("В итоговой таблице повторяются названия регионов")
    if (data[["innovation_mln", "grp_mln", "rd_mln"]] < 0).any().any():
        raise ValueError("Обнаружены отрицательные значения денежных показателей")
    return data, audit


def save_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUT / filename, index=False, sep=";", decimal=",", encoding="utf-8-sig")


def describe_training(train: pd.DataFrame) -> pd.DataFrame:
    records = []
    labels = {
        "innovation_mln": "Затраты на технологические инновации, млн руб.",
        "grp_mln": "ВРП, млн руб.",
        "rd_mln": "Внутренние затраты на НИР, млн руб.",
    }
    for column, label in labels.items():
        values = train[column].astype(float)
        mean = values.mean()
        records.append({
            "Показатель": label,
            "N": values.count(),
            "Среднее": mean,
            "Медиана": values.median(),
            "Стандартное отклонение": values.std(ddof=1),
            "Минимум": values.min(),
            "Q1_25%": values.quantile(0.25),
            "Q2_50%": values.quantile(0.50),
            "Q3_75%": values.quantile(0.75),
            "IQR": values.quantile(0.75) - values.quantile(0.25),
            "Коэффициент вариации_%": values.std(ddof=1) / mean * 100 if mean else np.nan,
            "Максимум": values.max(),
        })
    return pd.DataFrame(records)


def normalization_and_outliers(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["innovation_mln", "grp_mln", "rd_mln"]
    standardized = train[["region", "district"]].copy()
    outlier_rows = []
    for column in columns:
        mean = train[column].mean()
        scale = train[column].std(ddof=1)
        standardized[f"{column}_z"] = (train[column] - mean) / scale if scale else np.nan
        q1 = train[column].quantile(0.25)
        q3 = train[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        for _, row in train.loc[(train[column] < lower) | (train[column] > upper)].iterrows():
            outlier_rows.append({
                "Показатель": column,
                "Регион": row["region"],
                "Значение": row[column],
                "Нижняя_граница_1_5IQR": lower,
                "Верхняя_граница_1_5IQR": upper,
                "Правило": "Тьюки: Q1-1,5*IQR / Q3+1,5*IQR",
            })
    return standardized, pd.DataFrame(outlier_rows, columns=[
        "Показатель", "Регион", "Значение", "Нижняя_граница_1_5IQR",
        "Верхняя_граница_1_5IQR", "Правило",
    ])


def bootstrap_mean_median(values: np.ndarray, repetitions: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    observed = float(np.mean(values) - np.median(values))
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True)
    differences = draws.mean(axis=1) - np.median(draws, axis=1)
    centered = differences - observed
    p_value = float((np.count_nonzero(np.abs(centered) >= abs(observed)) + 1) / (repetitions + 1))
    low, high = np.quantile(differences, [ALPHA / 2, 1 - ALPHA / 2])
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "difference_mean_minus_median": observed,
        "bootstrap_ci_2_5": float(low),
        "bootstrap_ci_97_5": float(high),
        "bootstrap_p_value_centered": p_value,
        "bootstrap_repetitions": repetitions,
    }


def fit_models(train: pd.DataFrame) -> dict[str, object]:
    formula_level = "innovation_mln ~ grp_mln + rd_mln + C(side)"
    formula_log = "log_innovation ~ log_grp + log_rd + C(side)"
    level = smf.ols(formula_level, data=train).fit()
    positive = train.loc[
        (train[["innovation_mln", "grp_mln", "rd_mln"]] > 0).all(axis=1)
    ].copy()
    if len(positive) != len(train):
        raise ValueError("Для логарифмической модификации нужны положительные значения всех денежных показателей")
    positive["log_innovation"] = np.log(positive["innovation_mln"])
    positive["log_grp"] = np.log(positive["grp_mln"])
    positive["log_rd"] = np.log(positive["rd_mln"])
    log_model = smf.ols(formula_log, data=positive).fit()
    return {"level": level, "log": log_model}


def coefficient_table(model: object, label: str) -> pd.DataFrame:
    ci = model.conf_int()
    t_critical = float(stats.t.ppf(1 - ALPHA / 2, int(model.df_resid)))
    return pd.DataFrame({
        "Модель": label,
        "Коэффициент": model.params.index,
        "Оценка": model.params.values,
        "Стандартная ошибка": model.bse.values,
        "t_расчетное": model.tvalues.values,
        "df": int(model.df_resid),
        "t_критическое_двустороннее_alpha_0_05": t_critical,
        "p_value": model.pvalues.values,
        "ДИ95_нижняя": ci.iloc[:, 0].values,
        "ДИ95_верхняя": ci.iloc[:, 1].values,
    })


def model_summary_table(models: dict[str, object]) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        rows.append({
            "Модель": "Уровни" if name == "level" else "Логарифмы",
            "N_обучение": int(model.nobs),
            "R2": model.rsquared,
            "R2_скорректированный": model.rsquared_adj,
            "F_расчетное": model.fvalue,
            "F_df_model": int(model.df_model),
            "F_df_resid": int(model.df_resid),
            "F_критическое_alpha_0_05": float(stats.f.ppf(1 - ALPHA, model.df_model, model.df_resid)),
            "F_p_value": model.f_pvalue,
            "AIC": model.aic,
            "BIC": model.bic,
            "RMSE_обучение": float(np.sqrt(model.mse_resid)),
        })
    return pd.DataFrame(rows)


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    nonzero = actual != 0
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "MAPE_%": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan,
        "Средняя ошибка": float(np.mean(error)),
    }


def diagnostics(model: object, train: pd.DataFrame) -> dict[str, float | int]:
    residuals = np.asarray(model.resid)
    exog = np.asarray(model.model.exog)
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, exog)
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    return {
        "n": len(residuals),
        "residual_df": int(model.df_resid),
        "Jarque_Bera": float(stats.jarque_bera(residuals).statistic),
        "Jarque_Bera_p_value": float(stats.jarque_bera(residuals).pvalue),
        "Shapiro_W": float(shapiro_stat),
        "Shapiro_p_value": float(shapiro_p),
        "Breusch_Pagan_LM": float(bp_lm),
        "Breusch_Pagan_LM_df": int(exog.shape[1] - 1),
        "Breusch_Pagan_chi2_critical_alpha_0_05": float(
            stats.chi2.ppf(1 - ALPHA, exog.shape[1] - 1)
        ),
        "Breusch_Pagan_LM_p_value": float(bp_lm_p),
        "Breusch_Pagan_F": float(bp_f),
        "Breusch_Pagan_F_p_value": float(bp_f_p),
        "condition_number": float(model.condition_number),
    }


def create_charts(train: pd.DataFrame, test_predictions: pd.DataFrame,
                  level_model: object) -> None:
    numeric = ["innovation_mln", "grp_mln", "rd_mln"]
    titles = {
        "innovation_mln": "Затраты на технологические инновации, млн руб.",
        "grp_mln": "Валовой региональный продукт, млн руб.",
        "rd_mln": "Затраты на исследования и разработки, млн руб.",
    }
    for column in numeric:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(train[column], bins="auto", edgecolor="white", color="#377eb8")
        ax.set_title(f"Распределение: {titles[column]}")
        ax.set_xlabel(titles[column])
        ax.set_ylabel("Число регионов")
        fig.tight_layout()
        fig.savefig(OUT / f"hist_{column}.png", dpi=180)
        plt.close(fig)

    side_counts = train["side"].value_counts().reindex(SIDE_ORDER).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(side_counts.index[::-1], side_counts.values[::-1], color="#4daf4a")
    ax.set_title("Регионы обучающей выборки по сторонам света")
    ax.set_xlabel("Число регионов")
    ax.set_ylabel("Сторона света")
    fig.tight_layout()
    fig.savefig(OUT / "regions_by_side.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.pie(side_counts.values, labels=side_counts.index, autopct="%1.0f%%", startangle=90)
    ax.set_title("Структура обучающей выборки по сторонам света")
    fig.tight_layout()
    fig.savefig(OUT / "side_share_pie.png", dpi=180)
    plt.close(fig)

    palette = {"Запад": "#377eb8", "Север": "#4daf4a", "Юг": "#e41a1c", "Восток": "#ff7f00"}
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for side_name in SIDE_ORDER:
        part = train.loc[train["side"] == side_name]
        if len(part):
            ax.scatter(part["lon"], part["lat"], s=36, alpha=0.85,
                       color=palette[side_name], label=f"{side_name} ({len(part)})")
    ax.axhline(60, color="#555555", linestyle="--", linewidth=1)
    ax.axhline(50, color="#555555", linestyle="--", linewidth=1)
    ax.axvline(60, color="#555555", linestyle="--", linewidth=1)
    ax.text(166, 60.4, "60° с. ш.", ha="right", va="bottom", fontsize=9, color="#333333")
    ax.text(166, 50.4, "50° с. ш.", ha="right", va="bottom", fontsize=9, color="#333333")
    ax.text(61, 40.8, "60° в. д.", ha="left", va="bottom", fontsize=9, color="#333333")
    ax.set_xlim(15, 170)
    ax.set_ylim(40, 72)
    ax.set_xlabel("Долгота административного центра, ° в. д.")
    ax.set_ylabel("Широта административного центра, ° с. ш.")
    ax.set_title("Распределение регионов обучающей выборки по сторонам света")
    ax.legend(title="Сторона света (число регионов)", loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "sides_map.png", dpi=180)
    plt.close(fig)

    corr = train[numeric].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(7, 6))
    plot = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric)), ["Инновационные затраты", "ВРП", "НИР"], rotation=20, ha="right")
    ax.set_yticks(range(len(numeric)), ["Инновационные затраты", "ВРП", "НИР"])
    for row in range(len(numeric)):
        for col in range(len(numeric)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center")
    ax.set_title("Корреляции Пирсона, обучающая выборка")
    fig.colorbar(plot, ax=ax, label="Коэффициент корреляции")
    fig.tight_layout()
    fig.savefig(OUT / "correlation_matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(train["grp_mln"], train["innovation_mln"], alpha=0.8)
    axes[0].set_xlabel("ВРП, млн руб.")
    axes[0].set_ylabel("Затраты на технологические инновации, млн руб.")
    axes[0].set_title("Инновационные затраты и ВРП")
    axes[1].scatter(train["rd_mln"], train["innovation_mln"], alpha=0.8, color="#e41a1c")
    axes[1].set_xlabel("Внутренние затраты на НИР, млн руб.")
    axes[1].set_ylabel("Затраты на технологические инновации, млн руб.")
    axes[1].set_title("Инновационные затраты и НИР")
    fig.tight_layout()
    fig.savefig(OUT / "scatterplots.png", dpi=180)
    plt.close(fig)

    fitted = np.asarray(level_model.fittedvalues)
    residuals = np.asarray(level_model.resid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(fitted, residuals, alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Расчетные затраты, млн руб.")
    axes[0].set_ylabel("Остатки, млн руб.")
    axes[0].set_title("Остатки против расчетных значений")
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q–Q график остатков")
    fig.tight_layout()
    fig.savefig(OUT / "residual_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(test_predictions["actual"], test_predictions["predicted_level"], label="Модель в уровнях")
    ax.scatter(test_predictions["actual"], test_predictions["predicted_log"], label="Лог-модель")
    upper = max(test_predictions["actual"].max(), test_predictions["predicted_level"].max(),
                test_predictions["predicted_log"].max())
    ax.plot([0, upper], [0, upper], color="black", linestyle="--", label="Идеальный прогноз")
    ax.set_xlabel("Фактические затраты, млн руб.")
    ax.set_ylabel("Прогноз, млн руб.")
    ax.set_title("Фактические и прогнозные затраты на тестовой выборке")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "test_predictions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data, audit = build_dataset()
    rng = np.random.default_rng(SEED)
    test_indices = set(rng.choice(data.index.to_numpy(), size=TEST_SIZE, replace=False).tolist())
    test = data.loc[data.index.isin(test_indices)].copy().sort_values("region")
    train = data.loc[~data.index.isin(test_indices)].copy().sort_values("region")

    data_out = data[["region", "district", "side", "lat", "lon", "innovation_mln", "grp_mln", "rd_mln"]].copy()
    data_out.columns = ["Регион", "Федеральный округ", "Сторона света",
                        "Широта центра, ° с. ш.", "Долгота центра, ° в. д.",
                        "Затраты на инновации, млн руб.",
                        "ВРП, млн руб.", "Затраты на НИР, млн руб."]
    save_csv(data_out, "01_сводная_таблица.csv")
    save_csv(train[["region", "district", "side", "lat", "lon", "innovation_mln", "grp_mln", "rd_mln"]],
             "02_обучающая_выборка.csv")
    save_csv(test[["region", "district", "side", "lat", "lon", "innovation_mln", "grp_mln", "rd_mln"]],
             "03_тестовые_регионы.csv")

    audit_rows = []
    for check, regions in audit.items():
        for region in regions:
            audit_rows.append({"Проверка": check, "Регион_или_ключ": region})
    save_csv(pd.DataFrame(audit_rows, columns=["Проверка", "Регион_или_ключ"]), "04_аудит_сопоставления.csv")

    descriptions = describe_training(train)
    save_csv(descriptions, "05_описательная_статистика.csv")
    standardized, outliers = normalization_and_outliers(train)
    save_csv(standardized, "05a_z_нормализация_обучающей_выборки.csv")
    save_csv(outliers, "05b_выбросы_по_правилу_IQR.csv")
    corr = train[["innovation_mln", "grp_mln", "rd_mln"]].corr(method="pearson")
    save_csv(corr.rename_axis("Показатель").reset_index(), "06_корреляции.csv")
    spearman = train[["innovation_mln", "grp_mln", "rd_mln"]].corr(method="spearman")
    save_csv(spearman.rename_axis("Показатель").reset_index(), "06a_ранговые_корреляции.csv")

    pairs = [
        ("Затраты — ВРП", "innovation_mln", "grp_mln"),
        ("Затраты — НИР", "innovation_mln", "rd_mln"),
        ("ВРП — НИР", "grp_mln", "rd_mln"),
    ]
    corr_rows = []
    corr_df = len(train) - 2
    t_crit_corr = float(stats.t.ppf(1 - ALPHA / 2, corr_df))
    for label, first, second in pairs:
        x = train[first].to_numpy()
        yv = train[second].to_numpy()
        r = float(np.corrcoef(x, yv)[0, 1])
        t_r = float(r * np.sqrt(corr_df / (1 - r * r)))
        p_r = float(2 * stats.t.sf(abs(t_r), corr_df))
        rs = float(stats.spearmanr(x, yv).statistic)
        t_s = float(rs * np.sqrt(corr_df / (1 - rs * rs)))
        p_s = float(2 * stats.t.sf(abs(t_s), corr_df))
        corr_rows.append({
            "Пара": label,
            "r_Пирсона": r, "t_Пирсона": t_r, "p_Пирсона": p_r,
            "r_Спирмена": rs, "t_Спирмена": t_s, "p_Спирмена": p_s,
            "df": corr_df, "t_критическое": t_crit_corr,
        })
    save_csv(pd.DataFrame(corr_rows), "06b_значимость_корреляций.csv")

    side_groups = [(s, train.loc[train["side"] == s, "innovation_mln"].to_numpy())
                   for s in SIDE_ORDER if (train["side"] == s).any()]
    f_anova, p_anova = stats.f_oneway(*[g for _, g in side_groups])
    grand_mean = float(train["innovation_mln"].mean())
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in side_groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for _, g in side_groups)
    ss_total = ss_between + ss_within
    df_between = len(side_groups) - 1
    df_within = len(train) - len(side_groups)
    eta2 = float(ss_between / ss_total)
    save_csv(pd.DataFrame([
        {"Источник": "Межгрупповая (сторона света)", "SS": ss_between, "df": df_between,
         "MS": ss_between / df_between, "F": f_anova, "p_value": p_anova, "eta2": eta2},
        {"Источник": "Внутригрупповая (остаток)", "SS": ss_within, "df": df_within,
         "MS": ss_within / df_within, "F": np.nan, "p_value": np.nan, "eta2": np.nan},
        {"Источник": "Итого", "SS": ss_total, "df": len(train) - 1,
         "MS": np.nan, "F": np.nan, "p_value": np.nan, "eta2": np.nan},
    ]), "07a_anova_по_сторонам.csv")
    side_means = []
    for side_name, _ in side_groups:
        part = train.loc[train["side"] == side_name]
        side_means.append({
            "Сторона света": side_name, "n": len(part),
            "Среднее Y": part["innovation_mln"].mean(),
            "Медиана Y": part["innovation_mln"].median(),
            "Среднее ВРП": part["grp_mln"].mean(),
            "Среднее НИР": part["rd_mln"].mean(),
        })
    save_csv(pd.DataFrame(side_means), "07b_средние_по_сторонам.csv")

    median_test = bootstrap_mean_median(train["innovation_mln"].to_numpy())
    pd.DataFrame([median_test]).to_csv(OUT / "07_тест_среднего_и_медианы.csv", index=False,
                                      sep=";", decimal=",", encoding="utf-8-sig")

    models = fit_models(train)
    save_csv(pd.concat([coefficient_table(models["level"], "Линейная модель в уровнях"),
                        coefficient_table(models["log"], "Модификация: лог-линейная")], ignore_index=True),
             "08_коэффициенты_и_доверительные_интервалы.csv")
    summaries = model_summary_table(models)
    save_csv(summaries, "09_сводка_моделей.csv")

    level_model = models["level"]
    save_csv(pd.DataFrame([
        {"Источник": "Регрессия", "SS": level_model.ess, "df": int(level_model.df_model),
         "MS": level_model.mse_model, "F": level_model.fvalue, "p_value": level_model.f_pvalue},
        {"Источник": "Остаток", "SS": level_model.ssr, "df": int(level_model.df_resid),
         "MS": level_model.mse_resid, "F": np.nan, "p_value": np.nan},
        {"Источник": "Итого", "SS": level_model.ess + level_model.ssr,
         "df": int(level_model.nobs) - 1, "MS": np.nan, "F": np.nan, "p_value": np.nan},
    ]), "09a_дисперсионный_анализ_модели.csv")

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_rows = []
    for i, param_name in enumerate(level_model.model.exog_names):
        if param_name == "Intercept":
            continue
        label = {"grp_mln": "ВРП", "rd_mln": "НИР"}.get(
            param_name, param_name.replace("C(side)[T.", "Сторона света: ").replace("]", ""))
        vif_rows.append({"Переменная": label,
                         "VIF": float(variance_inflation_factor(level_model.model.exog, i))})
    save_csv(pd.DataFrame(vif_rows), "11c_VIF.csv")

    train_reset = train.copy()
    train_reset["fitted_sq"] = np.asarray(level_model.fittedvalues) ** 2
    reset_model = smf.ols(
        "innovation_mln ~ grp_mln + rd_mln + C(side) + fitted_sq", data=train_reset
    ).fit()
    reset_t = float(reset_model.tvalues["fitted_sq"])
    reset_p = float(reset_model.pvalues["fitted_sq"])

    one_sided_p = (models["level"].pvalues["grp_mln"] / 2
                   if models["level"].params["grp_mln"] > 0
                   else 1 - models["level"].pvalues["grp_mln"] / 2)
    grp_df = int(models["level"].df_resid)
    grp_t_critical = float(stats.t.ppf(1 - ALPHA, grp_df))
    pd.DataFrame([{
        "Гипотеза": "Положительная связь ВРП и затрат на инновации при контроле НИР и стороны света",
        "Оценка_коэффициента_ВРП": models["level"].params["grp_mln"],
        "t_расчетное": models["level"].tvalues["grp_mln"],
        "t_критическое_одностороннее": grp_t_critical,
        "df": grp_df,
        "p_value_одностороннее": one_sided_p,
        "alpha": ALPHA,
        "Решение": "Отвергаем H0" if one_sided_p < ALPHA else "Не отвергаем H0",
    }]).to_csv(OUT / "10_проверка_гипотезы_ВРП.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")

    diagnostic_rows = []
    for name, model in models.items():
        result = diagnostics(model, train)
        if name == "level":
            result["RESET_F"] = reset_t ** 2
            result["RESET_p_value"] = reset_p
        result["model"] = "Уровни" if name == "level" else "Логарифмы"
        diagnostic_rows.append(result)
    pd.DataFrame(diagnostic_rows).to_csv(OUT / "11_диагностика_моделей.csv", index=False,
                                         sep=";", decimal=",", encoding="utf-8-sig")

    level_model = models["level"]
    influence_frame = level_model.get_influence().summary_frame()
    influence_frame.insert(0, "region", train["region"].to_numpy())
    influence_frame.to_csv(OUT / "11a_влияние_наблюдений.csv", index=False,
                           sep=";", decimal=",", encoding="utf-8-sig")
    most_influential_index = int(np.argmax(influence_frame["cooks_d"].to_numpy()))
    most_influential_region = train.iloc[most_influential_index]["region"]
    sensitivity_train = train.drop(train.index[most_influential_index])
    sensitivity_model = smf.ols(
        "innovation_mln ~ grp_mln + rd_mln + C(side)", data=sensitivity_train
    ).fit()
    pd.DataFrame([
        {
            "Модель": "Все обучающие регионы",
            "Удаленный_регион": "",
            "Коэффициент_ВРП": level_model.params["grp_mln"],
            "R2": level_model.rsquared,
            "p_value_ВРП": level_model.pvalues["grp_mln"],
            "N": int(level_model.nobs),
        },
        {
            "Модель": "Без наблюдения с максимальным расстоянием Кука",
            "Удаленный_регион": most_influential_region,
            "Коэффициент_ВРП": sensitivity_model.params["grp_mln"],
            "R2": sensitivity_model.rsquared,
            "p_value_ВРП": sensitivity_model.pvalues["grp_mln"],
            "N": int(sensitivity_model.nobs),
        },
    ]).to_csv(OUT / "11b_чувствительность_к_влиятельному_региону.csv", index=False,
             sep=";", decimal=",", encoding="utf-8-sig")

    f_critical = float(stats.f.ppf(1 - ALPHA, level_model.df_model, level_model.df_resid))
    median_ci_contains_zero = median_test["bootstrap_ci_2_5"] <= 0 <= median_test["bootstrap_ci_97_5"]
    median_decision = (
        "не отвергаем равенство среднего и медианы"
        if median_ci_contains_zero else "отвергаем равенство среднего и медианы"
    )
    model_decision = (
        "модель в целом статистически значима"
        if level_model.f_pvalue < ALPHA else "нет оснований считать модель в целом статистически значимой"
    )
    grp_decision = (
        "отвергаем H0 в пользу положительной связи"
        if one_sided_p < ALPHA else "не отвергаем H0 о неположительном коэффициенте ВРП"
    )
    coefficient_lines = []
    for parameter in level_model.params.index:
        estimate = float(level_model.params[parameter])
        standard_error = float(level_model.bse[parameter])
        statistic = float(level_model.tvalues[parameter])
        p_value = float(level_model.pvalues[parameter])
        coefficient_lines.append(
            f"{parameter}: t = {estimate:.8g} / {standard_error:.8g} = {statistic:.6g}; "
            f"df={grp_df}; p-value={p_value:.6g}."
        )
    r_squared = float(level_model.rsquared)
    model_k = int(level_model.df_model)
    model_n = int(level_model.nobs)
    model_p = model_k + 1
    f_substitution = (
        f"F = ({r_squared:.8g}/{model_k}) / "
        f"((1-{r_squared:.8g})/({model_n}-{model_p})) = {level_model.fvalue:.6g}; "
        f"df1={model_k}; df2={model_n-model_p}."
    )
    inference_text = (
        "ПРОВЕРКА СРЕДНЕГО И МЕДИАНЫ (обучающая выборка)\n"
        f"Среднее = {median_test['mean']:.6f} млн руб.; медиана = {median_test['median']:.6f} млн руб.\n"
        f"Разность d = среднее - медиана = {median_test['difference_mean_minus_median']:.6f} млн руб.\n"
        f"95%-й бутстрэп-интервал для d: [{median_test['bootstrap_ci_2_5']:.6f}; {median_test['bootstrap_ci_97_5']:.6f}].\n"
        f"Центрированное бутстрэп p-value = {median_test['bootstrap_p_value_centered']:.6g}; "
        f"повторов = {median_test['bootstrap_repetitions']}.\n"
        f"Решение при alpha={ALPHA}: {median_decision}.\n\n"
        "ОБЩАЯ ЗНАЧИМОСТЬ ЛИНЕЙНОЙ МОДЕЛИ\n"
        f"F = {level_model.fvalue:.6f}; df1 = {int(level_model.df_model)}; "
        f"df2 = {int(level_model.df_resid)}; Fкр(alpha={ALPHA}) = {f_critical:.6f}; "
        f"p-value = {level_model.f_pvalue:.6g}.\n"
        f"Решение: {model_decision}.\n\n"
        "ЧУВСТВИТЕЛЬНОСТЬ К КАЧЕСТВУ ДАННЫХ\n"
        f"Максимальное расстояние Кука у региона «{most_influential_region}». "
        "Сопоставьте коэффициент ВРП, R2 и p-value в файле "
        "11b_чувствительность_к_влиятельному_региону.csv.\n\n"
        "ПРОВЕРКА НАПРАВЛЕННОЙ ГИПОТЕЗЫ О ВРП\n"
        f"t = {level_model.tvalues['grp_mln']:.6f}; df = {grp_df}; "
        f"tкр(alpha={ALPHA}, односторонний тест) = {grp_t_critical:.6f}; "
        f"p-value = {one_sided_p:.6g}.\n"
        f"Решение: {grp_decision}.\n\n"
        "ФОРМУЛЫ ДЛЯ ОТЧЕТА\n"
        "Критерий Стьюдента для коэффициента: t_j = b_j / SE(b_j), df = n - p. "
        "Оценки и ошибки приведены в 08_коэффициенты_и_доверительные_интервалы.csv.\n"
    ) + "\n".join(coefficient_lines) + (
        "\nДоверительный интервал коэффициента: b_j +/- t_(1-alpha/2; df) * SE(b_j).\n"
        "Критерий Фишера: F = (R2/k) / ((1-R2)/(n-p)); df1=k, df2=n-p. "
        "Параметры приведены в 09_сводка_моделей.csv.\n"
    ) + f"Подстановка для линейной модели: {f_substitution}\n" + (
        "Для гипотезы о равенстве среднего и медианы использован бутстрэп-интервал "
        "для разности (среднее - медиана); ноль внутри интервала означает, что "
        "на уровне значимости 0,05 нет оснований отвергнуть равенство.\n"
    )
    (OUT / "14_формулы_и_интерпретация.txt").write_text(inference_text, encoding="utf-8")

    level_pred = models["level"].predict(test)
    log_frame = test.copy()
    log_frame["log_grp"] = np.log(log_frame["grp_mln"])
    log_frame["log_rd"] = np.log(log_frame["rd_mln"])
    log_pred = np.exp(models["log"].predict(log_frame) + models["log"].mse_resid / 2)
    level_sf = models["level"].get_prediction(test).summary_frame(alpha=ALPHA)
    log_sf = models["log"].get_prediction(log_frame).summary_frame(alpha=ALPHA)
    predictions = pd.DataFrame({
        "region": test["region"].to_numpy(),
        "district": test["district"].to_numpy(),
        "side": test["side"].astype(str).to_numpy(),
        "actual": test["innovation_mln"].to_numpy(),
        "predicted_level": np.asarray(level_pred),
        "lower_level": np.asarray(level_sf["obs_ci_lower"]),
        "upper_level": np.asarray(level_sf["obs_ci_upper"]),
        "predicted_log": np.asarray(log_pred),
        "lower_log": np.exp(np.asarray(log_sf["obs_ci_lower"])),
        "upper_log": np.exp(np.asarray(log_sf["obs_ci_upper"])),
    })
    save_csv(predictions, "12_прогнозы_тестовых_регионов.csv")
    metric_rows = []
    for label, prediction in (("Линейная модель в уровнях", predictions.predicted_level.to_numpy()),
                              ("Модификация: лог-линейная", predictions.predicted_log.to_numpy())):
        metric_rows.append({"Модель": label, "N_тест": len(test), **prediction_metrics(predictions.actual.to_numpy(), prediction)})
    save_csv(pd.DataFrame(metric_rows), "13_метрики_тестового_прогноза.csv")

    create_charts(train, predictions, models["level"])
    print(f"Готово. Регионов после сопоставления: {len(data)}; обучение: {len(train)}; тест: {len(test)}.")
    print(f"Результаты сохранены в: {OUT}")
    print("Проверьте 04_аудит_сопоставления.csv; пустой файл означает, что все регионы сопоставлены без остатков.")
    print("Логарифмическая модификация использует только положительные значения.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Ошибка анализа: {exc}", file=sys.stderr)
        raise
