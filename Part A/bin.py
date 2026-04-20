from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
digits = load_digits()

images = digits.images
labels = digits.target

# Convert to binary
binary_images = (images >= 8).astype(int)

# Show few examples
plt.figure(figsize=(8, 4))

for i in range(5):
    # Original
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Orig: {labels[i]}")
    plt.axis('off')

    # Binary
    plt.subplot(2, 5, i + 6)
    plt.imshow(binary_images[i], cmap='gray')
    plt.title("Binary")
    plt.axis('off')

plt.suptitle("Original vs Binary Images")
plt.tight_layout()
plt.savefig('original_vs_binary.png')