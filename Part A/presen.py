# ============================================
# DIGITS DATASET: CLASSIFICATION + PATTERN MINING
# ============================================

# ----------- Imports -----------
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt

# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

def load_and_preprocess():
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Use float for numerical stability in KNN
    X_bin = (X >= 8).astype(float)

    return X_bin, y

# ============================================
# DATA SPLITTING
# ============================================

def split_data(X, y):
    return X[:179], y[:179], X[179:], y[179:]

# ============================================
# TASK 1: KNN CLASSIFICATION
# ============================================

def run_knn(X_train, y_train, X_test, y_test):
    k_values = [1, 3, 5, 10, 20]
    results = []

    print("\n===== TASK 1: KNN CLASSIFICATION =====\n")

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        correct = np.sum(y_pred == y_test)
        total = len(y_test)
        accuracy = correct / total

        results.append((k, accuracy))

        print(f"k = {k}")
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy*100:2.2f}%")
        print("-" * 30)

    return results

# ============================================
# PLOT RESULTS (KNN)
# ============================================

def plot_knn_results(results):
    k_vals = [k for k, acc in results]
    acc_vals = [acc for k, acc in results]

    plt.figure()
    plt.plot(k_vals, acc_vals, marker='o')
    plt.xlabel("k (Neighbors)")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs k")
    plt.grid()
    plt.savefig('knn_results.png')
    plt.close()

# ============================================
# TASK 2: FP-GROWTH
# ============================================

def run_fpgrowth(X_train):
    print("\n===== TASK 2: FP-GROWTH =====\n")

    
    df = pd.DataFrame(X_train.astype(bool))
    df.columns = [f'pixel_{i}' for i in range(64)]

    minsup_values = [0.1, 0.3, 0.5, 0.7]
    results = {}

    for minsup in minsup_values:
        freq_itemsets = fpgrowth(df, min_support=minsup, use_colnames=True)

        print(f"minsup = {minsup}")
        print("Number of patterns:", len(freq_itemsets))
        print(freq_itemsets.head())
        print("-" * 40)

        results[minsup] = freq_itemsets

    return results

# ============================================
# VISUALIZE MULTIPLE PATTERNS (IMPROVED)
# ============================================

def visualize_patterns_grid(fp_results):
    plt.figure(figsize=(10, 8))
    plot_index = 1

    for minsup, freq_itemsets in fp_results.items():

        # Filter meaningful patterns (avoid tiny ones)
        filtered = freq_itemsets[freq_itemsets['itemsets'].apply(lambda x: len(x) > 4)]

        if len(filtered) == 0:
            continue

        pattern = filtered.iloc[0]['itemsets']

        img = np.zeros(64)

        for item in pattern:
            idx = int(item.split('_')[1])
            img[idx] = 1

        img = img.reshape(8, 8)

        plt.subplot(2, 2, plot_index)
        plt.imshow(img, cmap='gray')
        plt.title(f"minsup = {minsup}")
        plt.axis('off')

        plot_index += 1

    plt.suptitle("Frequent Patterns at Different Support Levels")
    plt.tight_layout()
    plt.savefig('fp_patterns_grid.png')
    plt.close()

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess()

    X_train, y_train, X_test, y_test = split_data(X, y)

    # KNN
    knn_results = run_knn(X_train, y_train, X_test, y_test)
    plot_knn_results(knn_results)

    # FP-Growth
    fp_results = run_fpgrowth(X_train)

    # Improved visualization
    visualize_patterns_grid(fp_results)

    print("\nExecution completed successfully.")
    print("Saved: knn_results.png and fp_patterns_grid.png")

# Run program
if __name__ == "__main__":
    main()