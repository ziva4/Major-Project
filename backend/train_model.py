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
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)

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

        # Pad to 128 features
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)), mode='constant')

        return features[:128]

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

        # Generate normal frames (realistic video - people, objects, normal lighting)
        for _ in range(samples // 2):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * np.random.randint(50, 150)
            # Add natural texture
            noise = np.random.normal(0, 10, (480, 640, 3)).astype(np.uint8)
            frame = cv2.add(frame, noise)
            # Add some edges (faces, objects)
            for _ in range(np.random.randint(3, 8)):
                x, y_pos = np.random.randint(0, 640), np.random.randint(0, 480)
                cv2.circle(frame, (x, y_pos), np.random.randint(30, 80), (100, 100, 100), -1)
            features = self.create_features_from_frame(frame)
            X.append(features)
            y.append(0)

        # Generate violent frames (high motion, sharp edges, high contrast)
        for _ in range(samples // 2):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * np.random.randint(100, 180)
            # Add high-frequency edges (violence indicators)
            edges = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            edges = cv2.GaussianBlur(edges, (5, 5), 0)
            frame = cv2.addWeighted(frame, 0.6, edges, 0.4, 0)
            # Add sharp transitions
            for _ in range(np.random.randint(5, 15)):
                pt1 = (np.random.randint(0, 640), np.random.randint(0, 480))
                pt2 = (np.random.randint(0, 640), np.random.randint(0, 480))
                cv2.line(frame, pt1, pt2, (200, 50, 50), np.random.randint(2, 5))
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
        if self.model is None:
            return 0.0
        try:
            features = self.create_features_from_frame(frame)
            features = self.scaler.transform([features])
            confidence = float(self.model.predict(features, verbose=0)[0][0])
            return max(0.0, min(1.0, confidence))
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0

    def save_model(self):
        """Save trained model"""
        model_path = os.path.join(self.model_dir, 'violence_detector.h5')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        self.model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Model saved to {model_path}")

    def load_model(self):
        """Load trained model"""
        model_path = os.path.join(self.model_dir, 'violence_detector.h5')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = keras.models.load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return True
        return False

if __name__ == "__main__":
    detector = ViolenceDetectionModel()

    if not detector.load_model():
        print("Training new model...")
        detector.train()
    else:
        print("Using existing model")
