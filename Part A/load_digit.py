from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load dataset
digits = load_digits()

# Data and labels
X = digits.data        # (1797, 64)
y = digits.target      # (1797,)
images = digits.images # (1797, 8, 8)

# Basic info
print("Shape of data:", X.shape)
print("Shape of images:", images.shape)
print("Unique labels:", set(y))

# Show first 10 images
plt.figure(figsize=(8, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()