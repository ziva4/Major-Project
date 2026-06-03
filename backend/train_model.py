import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class ViolenceDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def create_features_from_frame(self, frame):
        """Extract features from a frame for violence detection"""
        if frame is None or frame.size == 0:
            return np.zeros(128)

        # Resize frame for consistency
        frame = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow features (motion detection)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / edges.size

        # Color histogram features
        hist_b = cv2.calcHist([frame], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [8], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [8], [0, 256])

        hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        hist = hist / (hist.sum() + 1e-5)

        # Texture features (LBP-like)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_features = np.array([
            np.mean(laplacian),
            np.std(laplacian),
            np.max(np.abs(laplacian))
        ])

        # Contour features
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        # Combine all features
        features = np.concatenate([
            hist,
            texture_features,
            [edge_density, num_contours]
        ])

        return features[:128]  # Ensure 128 features

    def build_model(self):
        """Build neural network model"""
        self.model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(128,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def generate_synthetic_data(self, samples=500):
        """Generate synthetic training data for demo"""
        X = []
        y = []

        # Generate normal frames (no violence)
        for _ in range(samples // 2):
            frame = np.random.randint(0, 100, (480, 640, 3), dtype=np.uint8)
            features = self.create_features_from_frame(frame)
            X.append(features)
            y.append(0)

        # Generate violent frames (high motion, high contrast)
        for _ in range(samples // 2):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add high-frequency noise for violence
            noise = np.random.randint(0, 100, (480, 640, 3))
            frame = cv2.add(frame, noise)
            features = self.create_features_from_frame(frame)
            X.append(features)
            y.append(1)

        return np.array(X), np.array(y)

    def train(self):
        """Train the model"""
        print("Generating synthetic training data...")
        X, y = self.generate_synthetic_data(samples=1000)

        print(f"Training data shape: {X.shape}, Labels shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.build_model()

        print("Training model...")
        self.model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {accuracy*100:.2f}%")
        print(f"Test Loss: {loss:.4f}")

        self.save_model()

    def predict(self, frame):
        """Predict violence in a frame"""
        features = self.create_features_from_frame(frame)
        features = self.scaler.transform([features])
        confidence = float(self.model.predict(features, verbose=0)[0][0])
        return confidence

    def save_model(self):
        """Save trained model"""
        os.makedirs('models', exist_ok=True)
        self.model.save('models/violence_detector.h5')
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("Model saved to models/violence_detector.h5")

    def load_model(self):
        """Load trained model"""
        if os.path.exists('models/violence_detector.h5'):
            self.model = keras.models.load_model('models/violence_detector.h5')
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("Model loaded from disk")
            return True
        return False

if __name__ == "__main__":
    detector = ViolenceDetectionModel()

    if not detector.load_model():
        print("Training new model...")
        detector.train()
    else:
        print("Using existing model")
