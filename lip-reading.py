import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import mediapipe as mp

class LipReader:
    def __init__(self, model_path=None):
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmarks indices for MediaPipe Face Mesh
        self.lip_landmarks = list(range(61, 69)) + list(range(48, 60))
        
        # Initialize or load the model
        self.model = self._build_model() if not model_path else self.load_model(model_path)
        
        # Sequence length for processing
        self.sequence_length = 20
        self.image_size = (88, 88)
        
    def _build_model(self):
        """Build the lip reading model architecture"""
        # Input layers
        input_shape = (self.sequence_length, *self.image_size, 1)
        inputs = layers.Input(shape=input_shape)
        
        # 3D CNN layers
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling3D((1, 2, 2))(x)
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling3D((1, 2, 2))(x)
        x = layers.Conv3D(96, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling3D((1, 2, 2))(x)
        
        # Reshape for LSTM
        x = layers.Reshape((-1, x.shape[-1] * x.shape[-2] * x.shape[-3]))(x)
        
        # LSTM layers
        x = layers.LSTM(256, return_sequences=True)(x)
        x = layers.LSTM(256)(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        
        # Output layer (vocabulary size would be adjusted based on your needs)
        outputs = layers.Dense(500, activation='softmax')(x)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def extract_lip_frames(self, video_path):
        """Extract lip region frames from video"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Extract lip coordinates
                lip_points = []
                for idx in self.lip_landmarks:
                    landmark = landmarks.landmark[idx]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    lip_points.append([x, y])
                
                # Calculate bounding box for lips
                lip_points = np.array(lip_points)
                x_min, y_min = np.min(lip_points, axis=0)
                x_max, y_max = np.max(lip_points, axis=0)
                
                # Add padding
                padding = 10
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame.shape[1], x_max + padding)
                y_max = min(frame.shape[0], y_max + padding)
                
                # Extract and preprocess lip region
                lip_frame = frame[y_min:y_max, x_min:x_max]
                lip_frame = cv2.resize(lip_frame, self.image_size)
                lip_frame = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2GRAY)
                lip_frame = lip_frame / 255.0
                
                frames.append(lip_frame)
        
        cap.release()
        return np.array(frames)
    
    def predict_sequence(self, frames):
        """Predict text from sequence of frames"""
        # Ensure we have the right number of frames
        if len(frames) < self.sequence_length:
            return "Video too short for prediction"
            
        # Prepare sequences
        sequences = []
        for i in range(0, len(frames) - self.sequence_length + 1, self.sequence_length):
            sequence = frames[i:i + self.sequence_length]
            sequences.append(sequence)
        
        sequences = np.array(sequences)
        sequences = np.expand_dims(sequences, -1)
        
        # Make predictions
        predictions = []
        for sequence in sequences:
            pred = self.model.predict(np.expand_dims(sequence, 0))
            predictions.append(pred)
            
        # Convert predictions to text (would need vocabulary mapping)
        return self._predictions_to_text(predictions)
    
    def _predictions_to_text(self, predictions):
        """Convert model predictions to text"""
        # This would need to be implemented based on your vocabulary
        # For now, return placeholder
        return "Predicted text would appear here"
    
    def save_model(self, path):
        """Save the model"""
        self.model.save(path)
    
    def load_model(self, path):
        """Load a saved model"""
        return tf.keras.models.load_model(path)

# Usage example
def main():
    # Initialize lip reader
    lip_reader = LipReader()
    
    # Process video
    video_path = "path_to_your_video.mp4"
    frames = lip_reader.extract_lip_frames(video_path)
    
    # Get prediction
    text = lip_reader.predict_sequence(frames)
    print(f"Predicted text: {text}")

if __name__ == "__main__":
    main()