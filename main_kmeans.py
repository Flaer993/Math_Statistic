"""Разделение записей из Excel на 5 кластеров методом K-means.

Запуск: python main.py
Нужен Python 3.10+; дополнительные библиотеки не требуются.
Исходная таблица должна лежать рядом с main.py, либо передайте --input.
Метод: https://scikit-learn.org/stable/modules/clustering.html#k-means
"""

import argparse
import csv
import math
import posixpath
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from math import dist, isfinite
from pathlib import Path
from random import Random
from zipfile import ZipFile, BadZipFile

FEATURES = [
    "Base", "ProfStatistics", "AdvanStatistics", "TimeSeries", "PresenTabl",
    "PersepMaps", "CHAID", "Mapping", "NeuralNets",
]
INPUT_NAME = "Пакеты ПО исходник.xlsx"
K = 5
N_INIT = 500
SEED = 42


def read_data(path):
    """Читает первый лист .xlsx (ZIP с XML) без установки библиотек."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(item.itertext()) for item in root]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = workbook.find("m:sheets/m:sheet", ns)
        if sheet is None:
            raise ValueError("В книге нет листов.")
        rel_id = sheet.attrib[f"{{{rel_ns}}}id"]
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.attrib["Target"] for r in relationships
                      if r.attrib["Id"] == rel_id)
        target = (target.lstrip("/") if target.startswith("/")
                  else posixpath.normpath(posixpath.join("xl", target)))
        data = ET.fromstring(archive.read(target))
        table = []
        for row in data.findall("m:sheetData/m:row", ns):
            values = {}
            for cell in row.findall("m:c", ns):
                address = cell.attrib["r"]
                column = 0
                for char in address:
                    if char.isalpha():
                        column = column * 26 + ord(char.upper()) - ord("A") + 1
                if cell.find("m:f", ns) is not None:
                    raise ValueError(f"Ячейка {address} содержит формулу. Нужны исходные значения.")
                value = cell.find("m:v", ns)
                kind = cell.attrib.get("t", "n")
                if kind == "inlineStr":
                    inline = cell.find("m:is", ns)
                    value = "".join(inline.itertext()) if inline is not None else ""
                elif value is None or value.text is None:
                    value = None
                elif kind == "s":
                    value = strings[int(value.text)]
                elif kind in ("n", "b"):
                    number = float(value.text)
                    if not math.isfinite(number):
                        raise ValueError(f"Некорректное число в {address}.")
                    value = int(number) if number.is_integer() else number
                elif kind == "e":
                    raise ValueError(f"Ошибка Excel в ячейке {address}: {value.text}")
                else:
                    value = value.text
                values[column - 1] = value
            if values and any(value is not None for value in values.values()):
                table.append((int(row.attrib["r"]), values))

    if len(table) < K + 1:
        raise ValueError("Недостаточно строк данных для пяти кластеров.")
    width = max(table[0][1]) + 1
    headers = [str(table[0][1].get(i, "")).strip() for i in range(width)]
    if len(set(headers)) != width or "" in headers:
        raise ValueError("Заголовки должны быть заполненными и уникальными.")
    missing = set(FEATURES) - set(headers)
    if missing:
        raise ValueError("Отсутствуют признаки: " + ", ".join(sorted(missing)))
    indices = [headers.index(name) for name in FEATURES]
    rows, points, row_numbers = [], [], []
    for number, values in table[1:]:
        row = [values.get(i) for i in range(width)]
        point = [row[index] for index in indices]
        if any(value not in (0, 1) for value in point):
            raise ValueError(f"Строка Excel {number}: все признаки должны быть числами 0 или 1.")
        rows.append(row)
        points.append(tuple(float(value) for value in point))
        row_numbers.append(number)
    return sheet.attrib["name"], headers, rows, points, row_numbers


def _initial_centers(points, counts, k, rng):
    """Выбирает начальные центры по правилу K-means++."""
    centers = [rng.choices(points, weights=counts)[0]]
    distances = [dist(point, centers[0]) for point in points]
    while len(centers) < k:
        weights = [
            count * (distance * distance)
            for count, distance in zip(counts, distances)
        ]
        total_weight = sum(weights)
        if not isfinite(total_weight) or total_weight == 0:
            raise ValueError("Невозможно вычислить веса K-means++ в данном масштабе.")
        center = rng.choices(points, weights=weights)[0]
        centers.append(center)
        distances = [
            min(distance, dist(point, center))
            for point, distance in zip(points, distances)
        ]
    return centers


def _assign(points, centers):
    labels = []
    distances = []
    for point in points:
        distance_to_centers = [dist(point, center) for center in centers]
        label = min(range(len(centers)), key=distance_to_centers.__getitem__)
        labels.append(label)
        distances.append(distance_to_centers[label])
    return labels, distances


def kmeans(points, k=5, n_init=100, seed=42, max_iter=300):
    """Возвращает метки от нуля, центры, сумму квадратов расстояний и число итераций.

    Использует K-means++ и итерации Ллойда. Из всех сошедшихся запусков
    выбирает разбиение с наименьшей суммой квадратов расстояний.
    """
    if any(type(value) is not int or value < 1 for value in (k, n_init, max_iter)):
        raise ValueError("k, n_init и max_iter должны быть положительными целыми числами.")

    rows = [tuple(map(float, point)) for point in points]
    if not rows or not rows[0]:
        raise ValueError("Для кластеризации нужны непустые векторы признаков.")
    dimension = len(rows[0])
    if any(len(row) != dimension or not all(map(isfinite, row)) for row in rows):
        raise ValueError("Векторы должны иметь одинаковую длину и конечные значения.")

    # Одинаковые строки обрабатываются вместе с сохранением их веса.
    frequencies = Counter(rows)
    if len(frequencies) < k:
        raise ValueError(f"Для {k} кластеров нужно не менее {k} разных наблюдений.")
    unique_points = list(frequencies)
    counts = list(frequencies.values())
    rng = Random(seed)
    best = None

    for _ in range(n_init):
        centers = _initial_centers(unique_points, counts, k, rng)
        labels, _ = _assign(unique_points, centers)
        for iteration in range(1, max_iter + 1):
            sizes = [0] * k
            coordinate_sums = [[0.0] * dimension for _ in range(k)]
            for point, count, label in zip(unique_points, counts, labels):
                sizes[label] += count
                sums = coordinate_sums[label]
                for coordinate, value in enumerate(point):
                    sums[coordinate] += count * value
            # Запуск с пустым кластером заменяется следующей инициализацией.
            if 0 in sizes:
                break
            centers = [
                tuple(total / size for total in sums)
                for sums, size in zip(coordinate_sums, sizes)
            ]
            new_labels, distances = _assign(unique_points, centers)
            if labels == new_labels:
                inertia = sum(
                    count * (distance * distance)
                    for count, distance in zip(counts, distances)
                )
                if not isfinite(inertia):
                    raise ValueError("Сумма квадратов расстояний превышает диапазон чисел.")
                if best is None or inertia < best[2]:
                    best = (labels, centers, inertia, iteration)
                break
            labels = new_labels

    if best is None:
        raise RuntimeError("K-means не сошёлся ни в одном из запусков.")
    labels, centers, inertia, iterations = best
    label_by_point = dict(zip(unique_points, labels))
    return [label_by_point[row] for row in rows], centers, inertia, iterations



def write_csv(path, headers, rows):
    """UTF-8 с BOM и разделителем ; для открытия в русской версии Excel."""
    def cell(value):
        # Десятичная запятая соответствует локали русской версии Excel.
        return format(value, ".10g").replace(".", ",") if isinstance(value, float) else value

    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output, delimiter=";")
        writer.writerow(headers)
        writer.writerows([[cell(value) for value in row] for row in rows])


def main():
    parser = argparse.ArgumentParser(description="K-means: 5 кластеров по 9 признакам использования ПО")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / INPUT_NAME,
                        help="Путь к исходному файлу .xlsx")
    parser.add_argument("--output-dir", type=Path, help="Папка результатов (по умолчанию рядом с исходником)")
    args = parser.parse_args()
    source = args.input.resolve()
    destination = args.output_dir.resolve() if args.output_dir else source.parent
    sheet, headers, rows, points, row_numbers = read_data(source)

    # Все признаки бинарные и имеют одинаковый масштаб. Work_area и текстовая
    # область деятельности — категории, поэтому в расстоянии не участвуют.
    print(f"Прочитано записей: {len(points)}. Признаков: {len(FEATURES)}.", flush=True)
    print(f"K-means: {N_INIT} запусков, k={K}, seed={SEED}...", flush=True)
    labels, centers, inertia, iterations = kmeans(points, K, N_INIT, SEED)

    # Номера условны. Для удобства нумеруем группы по убыванию размера.
    sizes = Counter(labels)
    order = sorted(range(K), key=lambda cluster: (-sizes[cluster], tuple(centers[cluster])))
    mapping = {old: new + 1 for new, old in enumerate(order)}
    clusters = [mapping[label] for label in labels]
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "Кластеры_5.csv", ["Строка_Excel"] + headers + ["Кластер"],
              [[number] + row + [cluster] for number, row, cluster in zip(row_numbers, rows, clusters)])
    write_csv(destination / "Центры_кластеров.csv", ["Кластер", "Количество", "Доля"] + FEATURES,
              [[mapping[old], sizes[old], sizes[old] / len(rows)] + list(centers[old]) for old in order])

    lines = [
        "Кластеризация методом K-means (k-средних)",
        f"Источник: {source.name}, лист {sheet}",
        f"Записей: {len(rows)}. Кластеров: {K}.",
        "Признаки: " + ", ".join(FEATURES),
        "Work_area и Область_деятельности сохранены как описание и не участвуют в расстоянии.",
        "Масштабирование не применялось: все признаки равны 0 или 1 и имеют одинаковый вес.",
        f"Инициализация k-means++, seed={SEED}, запусков={N_INIT}.",
        "Из сошедшихся запусков выбрано разбиение с минимальной суммой квадратов расстояний.",
        f"Сумма квадратов расстояний (SSE): {inertia:.6f}.",
        f"Итераций в выбранном запуске: {iterations}.",
        "Номера кластеров условны и упорядочены по убыванию числа записей.",
        "",
    ]
    for old in order:
        line = f"Кластер {mapping[old]}: {sizes[old]} записей ({sizes[old] / len(rows):.1%})."
        print(line)
        lines.append(line)
    lines += [
        "",
        "Кластеры_5.csv: все исходные строки, их номера в Excel и назначенные кластеры.",
        "Центры_кластеров.csv: размер каждой группы и средние значения девяти признаков.",
        "Центр 0,8 по признаку означает: у 80% записей этого кластера признак равен 1.",
        "CSV: UTF-8 с BOM, разделитель ;, десятичная запятая (для русской версии Excel).",
        "Повторный запуск: python main.py",
        "Python 3.10 или новее. Установка дополнительных библиотек не требуется.",
        "",
        "Как работает расчёт:",
        "1. Читаются девять признаков из Excel. Каждая запись становится точкой в 9 измерениях.",
        "2. k-means++ выбирает пять начальных центров, отдавая предпочтение удалённым точкам.",
        "3. Каждая точка относится к ближайшему центру по евклидову расстоянию.",
        "4. Каждый центр заменяется средним значением признаков точек своей группы.",
        "5. Шаги 3–4 повторяются до стабилизации принадлежности точек к группам.",
        f"6. Выполняются {N_INIT} инициализаций и выбирается результат с наименьшим SSE.",
        "Пять — заданное число кластеров, а не найденное автоматически оптимальное число.",
        "K-means может находить разные локальные минимумы; глобальный минимум не гарантирован.",
        "Описание метода: https://scikit-learn.org/stable/modules/clustering.html#k-means",
    ]
    (destination / "Результат_KMeans.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"SSE: {inertia:.6f}. Результаты сохранены: {destination}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, BadZipFile, ET.ParseError, RuntimeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
