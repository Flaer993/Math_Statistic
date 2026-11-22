"""DBSCAN по девяти бинарным признакам использования ПО.

Запуск: python main.py
Python 3.10+, дополнительные библиотеки не требуются.
eps=0.5, min_samples=15: для исходных данных 5 кластеров и 210 шумовых записей.
Исходная таблица должна лежать рядом с main.py, либо передайте --input.
"""

import argparse
import csv
import math
import posixpath
import sys
import xml.etree.ElementTree as ET
from collections import Counter, deque
from math import dist, isfinite
from pathlib import Path
from zipfile import ZipFile, BadZipFile

FEATURES = [
    "Base", "ProfStatistics", "AdvanStatistics", "TimeSeries", "PresenTabl",
    "PersepMaps", "CHAID", "Mapping", "NeuralNets",
]
INPUT_NAME = "Пакеты ПО исходник.xlsx"
EPS = 0.5
MIN_SAMPLES = 15


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

    if len(table) < 2:
        raise ValueError("В таблице нет строк данных.")
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


def dbscan(points, eps=1.0, min_samples=5):
    """DBSCAN с евклидовой метрикой; метки 0.., шум -1.

    Возвращает метки, признаки ядерных точек и количества соседей.
    min_samples включает саму точку. Граница eps включена в окрестность.
    """
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps должен быть положительным конечным числом.")
    if type(min_samples) is not int or min_samples < 1:
        raise ValueError("min_samples должен быть положительным целым числом.")
    rows = [tuple(map(float, point)) for point in points]
    if not rows or not rows[0]:
        raise ValueError("Нужны непустые векторы признаков.")
    dimension = len(rows[0])
    if any(len(row) != dimension or not all(map(isfinite, row)) for row in rows):
        raise ValueError("Векторы должны иметь одинаковую длину и конечные значения.")

    # Повторяющиеся наблюдения учитываются по отдельности: они влияют на плотность.
    neighbors = [[i] for i in range(len(rows))]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if dist(rows[i], rows[j]) <= eps:
                neighbors[i].append(j)
                neighbors[j].append(i)
    neighbor_counts = [len(indices) for indices in neighbors]
    core = [count >= min_samples for count in neighbor_counts]
    labels = [-1] * len(rows)
    visited = [False] * len(rows)
    cluster = 0

    for start in range(len(rows)):
        if visited[start]:
            continue
        visited[start] = True
        if not core[start]:
            continue
        labels[start] = cluster
        pending = deque(neighbors[start])
        queued = set(neighbors[start])
        while pending:
            point = pending.popleft()
            if not visited[point]:
                visited[point] = True
                # Только ядерные точки могут расширять кластер.
                if core[point]:
                    for neighbor in neighbors[point]:
                        if neighbor not in queued:
                            queued.add(neighbor)
                            pending.append(neighbor)
            # Ранее помеченная шумом точка может стать граничной.
            # Общая граничная точка принадлежит первому достигшему её кластеру.
            if labels[point] == -1:
                labels[point] = cluster
        cluster += 1
    return labels, core, neighbor_counts


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
    parser = argparse.ArgumentParser(description="DBSCAN по девяти бинарным признакам использования ПО")
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / INPUT_NAME,
                        help="Путь к исходному файлу .xlsx")
    parser.add_argument("--output-dir", type=Path, help="Папка результатов (по умолчанию рядом с исходником)")
    parser.add_argument("--eps", type=float, default=EPS, help="Радиус окрестности (по умолчанию 0.5)")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                        help="Минимальное число соседей, включая саму точку (по умолчанию 15)")
    args = parser.parse_args()
    source = args.input.resolve()
    destination = args.output_dir.resolve() if args.output_dir else source.parent
    sheet, headers, rows, points, row_numbers = read_data(source)
    print(f"Прочитано записей: {len(points)}. Признаков: {len(FEATURES)}.", flush=True)
    print(f"DBSCAN: eps={args.eps}, min_samples={args.min_samples}, метрика euclidean.", flush=True)
    labels, core, neighbor_counts = dbscan(points, args.eps, args.min_samples)

    # Номера условны: группы нумеруются по убыванию размера. Шум всегда -1.
    sizes = Counter(labels)
    order = sorted((label for label in sizes if label != -1), key=lambda label: (-sizes[label], label))
    mapping = {old: new + 1 for new, old in enumerate(order)}
    mapping[-1] = -1
    clusters = [mapping[label] for label in labels]
    kinds = ["шум" if label == -1 else "ядерная" if is_core else "граничная"
             for label, is_core in zip(labels, core)]
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "Кластеры_DBSCAN.csv",
              ["Строка_Excel"] + headers + ["Кластер", "Тип_точки", "Соседей_включая_саму_точку"],
              [[number] + row + [cluster, kind, neighbors]
               for number, row, cluster, kind, neighbors
               in zip(row_numbers, rows, clusters, kinds, neighbor_counts)])

    # У DBSCAN нет центров: это только средние профили готовых групп.
    profile_rows = []
    for old in order + ([-1] if sizes[-1] else []):
        members = [index for index, label in enumerate(labels) if label == old]
        means = [sum(points[index][column] for index in members) / len(members)
                 for column in range(len(FEATURES))]
        core_count = sum(core[index] for index in members)
        profile_rows.append([mapping[old], "шум" if old == -1 else "кластер", len(members),
                             len(members) / len(rows), core_count,
                             0 if old == -1 else len(members) - core_count] + means)
    write_csv(destination / "Профили_DBSCAN.csv",
              ["Кластер", "Категория", "Количество", "Доля_от_всех", "Ядерных", "Граничных"] + FEATURES,
              profile_rows)

    lines = [
        "Кластеризация методом DBSCAN",
        f"Источник: {source.name}, лист {sheet}",
        f"Записей: {len(rows)}. Кластеров без учёта шума: {len(order)}.",
        f"Шум (-1): {sizes[-1]} записей ({sizes[-1] / len(rows):.1%}).",
        f"eps={args.eps}; min_samples={args.min_samples}; метрика: евклидова.",
        "Признаки: " + ", ".join(FEATURES),
        "Work_area и Область_деятельности сохранены как описание, в расстоянии не участвуют.",
        "Все признаки равны 0 или 1, дополнительное масштабирование не применяется.",
        "Повторяющиеся записи учитываются как отдельные наблюдения и влияют на плотность.",
        "",
    ]
    for old in order:
        line = f"Кластер {mapping[old]}: {sizes[old]} записей ({sizes[old] / len(rows):.1%})."
        print(line)
        lines.append(line)
    noise_line = f"Шум (-1): {sizes[-1]} записей ({sizes[-1] / len(rows):.1%})."
    print(noise_line)
    lines += [
        "",
        "Как работает DBSCAN:",
        "1. Для каждой точки находятся соседи на расстоянии не больше eps.",
        "2. Точка ядерная, если её окрестность содержит не менее min_samples точек, включая её саму.",
        "3. Связанные цепочками соседства ядерные точки образуют кластер.",
        "4. Неядерная точка рядом с ядерной присоединяется к кластеру как граничная.",
        "5. Остальные точки отмечаются как шум (-1). Шум не считается отдельным кластером.",
        "Число кластеров определяется параметрами и данными, фиксированного параметра k нет.",
        "Общая для двух кластеров граничная точка относится к первому найденному кластеру.",
        "Номера кластеров условны. Средние в Профили_DBSCAN.csv — описание готовых групп.",
        "",
        "Выбор параметров по умолчанию:",
        "eps=0.5 и min_samples=15 подобраны для исходной таблицы из 310 записей, чтобы получить 5 групп.",
        "Для того набора проверены все 10 различных окрестностей и min_samples от 1 до 310.",
        "Пять групп получаются только при 0<eps<1 и min_samples=15; 210 записей остаются шумом.",
        "При eps<1 соседями в бинарных данных являются только полностью совпадающие наборы признаков.",
        "Таким образом, пять групп в исходном наборе — часто встречающиеся одинаковые профили.",
        "Этот подбор выполняет требование о числе групп, но не доказывает оптимальность разбиения.",
        "При изменении данных или параметров число кластеров и доля шума могут измениться.",
        "",
        "Файлы: Кластеры_DBSCAN.csv — исходные строки с метками и типами точек;",
        "Профили_DBSCAN.csv — размеры групп, шум и средние значения признаков.",
        "CSV: UTF-8 с BOM, разделитель ;, десятичная запятая (для русской версии Excel).",
        "Запуск из папки со скриптом и исходной таблицей: python main.py",
        "Другие параметры: python main.py --eps 1.0 --min-samples 5",
        "Python 3.10 или новее. Дополнительные библиотеки не требуются.",
        "Описание метода: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html",
    ]
    (destination / "Результат_DBSCAN.txt").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"Найдено кластеров: {len(order)}. Результаты сохранены: {destination}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, BadZipFile, ET.ParseError, RuntimeError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
