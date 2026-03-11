import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# CREATE DATA
# --------------------------
data = {
    "Category": ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"],
    "Views": [120, 150, 90, 60, 80, 110],
    "Rating": [4.5, 4.0, 4.8, 3.5, 4.2, 4.6]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

print("DataFrame:")
print(df)

# --------------------------
# CREATE SUBPLOTS
# --------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --------------------------
# BAR PLOT: Views by Category
# --------------------------
sns.barplot(
    x="Category",
    y="Views",
    data=df,
    ax=axes[0],
    palette="viridis"
)
axes[0].set_title("Views by Category")
axes[0].set_xlabel("Category")
axes[0].set_ylabel("Views")

# --------------------------
# SCATTER PLOT: Rating vs Views
# --------------------------
sns.scatterplot(
    x="Views",
    y="Rating",
    data=df,
    hue="Category",
    s=100,
    ax=axes[1]
)
axes[1].set_title("Rating vs Views")
axes[1].set_xlabel("Views")
axes[1].set_ylabel("Rating")

# Adjust layout
plt.tight_layout()
plt.show()
