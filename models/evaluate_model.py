import tensorflow as tf
import numpy as np
from dataset.dataset_loader import load_and_preprocess_data
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def visualize_predictions(x_test, y_test, y_pred, num_images=100):

    plt.figure(figsize=(15, 15))
    num_rows = int(np.ceil(np.sqrt(num_images)))
    num_cols = int(np.ceil(num_images / num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(x_test[i].squeeze())
        plt.title(f"True: {y_test[i]}, Pred: {y_pred[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def evaluate_model():
    train_ds, test_ds = load_and_preprocess_data()

    x_test = []
    y_test = []
    for batch in test_ds:
        images, labels = batch
        x_test.extend(images.numpy())
        y_test.extend(labels.numpy())

    x_test = np.array(x_test)
    y_test = np.argmax(np.array(y_test), axis=1)

    model_layer = tf.keras.layers.TFSMLayer('../saved_models/densenet/', call_endpoint='serving_default')

    predictions = model_layer(x_test)
    if isinstance(predictions, dict):
        predictions = predictions[list(predictions.keys())[0]]
    predictions = predictions.numpy()

    y_pred = np.argmax(predictions, axis=1)

    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("Wizualizacja wyników...")
    visualize_predictions(x_test, y_test, y_pred, num_images=16)


if __name__ == "__main__":
    evaluate_model()