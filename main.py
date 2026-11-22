from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

EPS = 0.5
MIN_SAMPLES = 15
folder = Path(__file__).resolve().parent

data = pd.read_excel(folder / "Пакеты ПО исходник.xlsx")
features = [
    "Base", "ProfStatistics", "AdvanStatistics", "TimeSeries", "PresenTabl",
    "PersepMaps", "CHAID", "Mapping", "NeuralNets",
]
X = data[features]
if not X.isin([0, 1]).all().all():
    raise ValueError("В признаках должны быть только числа 0 или 1 без пропусков.")

labels = pd.Series(DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(X))

sizes = labels[labels != -1].value_counts()
numbers = {label: number for number, label in enumerate(sizes.index, start=1)}
numbers[-1] = 0
data["Кластер"] = labels.map(numbers)
data.insert(0, "Строка_Excel", data.index + 2)

data.to_csv(folder / "Кластеры_DBSCAN.csv", index=False, sep=";", encoding="utf-8-sig")
profiles = data.groupby("Кластер")[features].mean()
profiles.insert(0, "Количество", data["Кластер"].value_counts())
profiles.to_csv(folder / "Профили_DBSCAN.csv", sep=";", decimal=",", encoding="utf-8-sig")

pca = PCA(n_components=2).fit(X)
points = data.groupby(features + ["Кластер"]).size().reset_index(name="Количество")
points[["x", "y"]] = pca.transform(points[features])

fig, ax = plt.subplots(figsize=(10.5, 6.5))
fig.subplots_adjust(right=0.75, bottom=0.20, top=0.83)
for cluster, group in points.groupby("Кластер"):
    name = "Шум (0)" if cluster == 0 else f"Кластер {cluster}"
    color = "#9ca3af" if cluster == 0 else plt.get_cmap("tab10")((cluster - 1) % 10)
    ax.scatter(group["x"], group["y"], s=group["Количество"] * 18,
               color=color, alpha=0.5 if cluster == 0 else 0.95,
               edgecolors="white", linewidths=0.7,
               label=f"{name}: {group['Количество'].sum()}", zorder=2)

fig.suptitle(f"DBSCAN: {len(sizes)} кластеров и {(labels == -1).sum()} шумовых записей",
             fontsize=16, fontweight="bold")
ax.set_title(f"eps = {EPS}   |   min_samples = {MIN_SAMPLES}", fontsize=11, pad=12)
ax.set_xlabel(f"Главная компонента 1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)")
ax.set_ylabel(f"Главная компонента 2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)")
ax.legend(title="Группы и число записей", loc="upper left", bbox_to_anchor=(1.02, 1),
          frameon=False, markerscale=0.55, labelspacing=1.2)
ax.grid(alpha=0.15, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.12, 0.04, "Размер точки отражает число одинаковых записей.\n"
         "PCA используется для отображения. DBSCAN рассчитан по 9 исходным признакам.", fontsize=10)
fig.savefig(folder / "График_DBSCAN.png", dpi=200, facecolor="white")
plt.close(fig)

print(f"Кластеров: {len(sizes)}. Шумовых записей: {(labels == -1).sum()}.")
print(profiles["Количество"].to_string())
print(f"График сохранён: {folder / 'График_DBSCAN.png'}")
