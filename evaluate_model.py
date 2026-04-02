import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

model = keras.models.load_model("dog_cat_model.keras")
print("Model loaded successfully!")

validation_ds = keras.utils.image_dataset_from_directory(
    directory=r'C:\6th sem\Deep Learning\PROJECT\test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256),
    shuffle=False
)

def process(image, label):
    image = tf.cast(image/255.0, tf.float32)
    return image, label

validation_ds = validation_ds.map(process)

images = []
labels = []

for image_batch, label_batch in validation_ds:
    images.append(image_batch.numpy())
    labels.append(label_batch.numpy())

X_val = np.concatenate(images, axis=0)
y_val = np.concatenate(labels, axis=0)

print(f"Validation dataset: {X_val.shape[0]} images")

y_pred_proba = model.predict(X_val)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
y_true = y_val.flatten()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*50)

print("\nCLASSIFICATION REPORT")
print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cat', 'Dog'],
            yticklabels=['Cat', 'Dog'],
            cbar_kws={'label': 'Count'})

plt.title('Confusion Matrix\n(Total Images: {})'.format(len(y_true)), fontsize=14, pad=20)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.show()

print("\nCONFUSION MATRIX BREAKDOWN")
print("                Predicted")
print("               Cat    Dog")
print("Actual  Cat  | {:4d} | {:4d} |".format(cm[0,0], cm[0,1]))
print("        Dog  | {:4d} | {:4d} |".format(cm[1,0], cm[1,1]))
print("Total Images:", len(y_true))

print("\nSUMMARY TABLE")
print("-" * 30)
print(f"{'Metric':<12} {'Value':<10} {'Percentage'}")
print("-" * 30)
print(f"{'Accuracy':<12} {accuracy:<10.4f} {accuracy*100:<8.2f}%")
print(f"{'Precision':<12} {precision:<10.4f} {precision*100:<8.2f}%")
print(f"{'Recall':<12} {recall:<10.4f} {recall*100:<8.2f}%")
print(f"{'F1-Score':<12} {f1:<10.4f} {f1*100:<8.2f}%")