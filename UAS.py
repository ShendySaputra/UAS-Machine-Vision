{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "72d47a6b-326c-4d51-ad2c-639d7df66b49",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import library\n",
    "import numpy as np\n",
    "from sklearn import datasets\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report\n",
    "from skimage.feature import hog\n",
    "\n",
    "# Load dataset\n",
    "mnist = datasets.fetch_openml('mnist_784', as_frame=False, parser='auto')\n",
    "data = mnist.data.astype(\"uint8\")\n",
    "target = mnist.target.astype(\"uint8\")\n",
    "\n",
    "# Ekstraksi fitur HOG\n",
    "def extract_hog_features(images):\n",
    "    hog_features = []\n",
    "    for image in images:\n",
    "        fd, hog_image = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)\n",
    "        hog_features.append(fd)\n",
    "    return np.array(hog_features)\n",
    "\n",
    "# Membagi dataset menjadi data latih dan data uji\n",
    "X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)\n",
    "\n",
    "# Ekstraksi fitur HOG pada data latih dan data uji\n",
    "X_train_hog = extract_hog_features(X_train)\n",
    "X_test_hog = extract_hog_features(X_test)\n",
    "\n",
    "# Melatih model SVM\n",
    "svm_model = SVC()\n",
    "svm_model.fit(X_train_hog, y_train)\n",
    "\n",
    "# Melakukan prediksi pada data uji\n",
    "y_pred = svm_model.predict(X_test_hog)\n",
    "\n",
    "# Evaluasi performa\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "conf_matrix = confusion_matrix(y_test, y_pred)\n",
    "precision = precision_score(y_test, y_pred, average='micro')\n",
    "classification_rep = classification_report(y_test, y_pred)\n",
    "\n",
    "# Menampilkan hasil evaluasi\n",
    "print(\"Accuracy:\", accuracy)\n",
    "print(\"Confusion Matrix:\\n\", conf_matrix)\n",
    "print(\"Precision:\", precision)\n",
    "print(\"Classification Report:\\n\", classification_rep)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
