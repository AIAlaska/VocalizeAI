import torch
import torchaudio
import numpy as np
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from pydub import AudioSegment
import librosa
import json
import soundfile as sf
from scipy.io import wavfile
import warnings
import logging
from typing import List, Dict, Optional, Tuple
import shutil

class VoiceProfileCreator:
    def __init__(self, output_dir: str = "voice_profile"):
        """
        Initialize the voice profile creator
        
        Args:
            output_dir (str): Directory to save the voice profile and processed audio
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = self.output_dir / "processed_audio"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Audio processing parameters
        self.target_sr = 22050  # Target sample rate
        self.min_duration = 3.0  # Minimum duration in seconds
        self.max_duration = 12.0  # Maximum duration in seconds
        
        # Initialize XTTS for voice embedding extraction
        self.initialize_xtts()
        
    def initialize_xtts(self):
        """Initialize the XTTS model for voice embedding extraction"""
        try:
            self.logger.info("Initializing XTTS model...")
            config = XttsConfig()
            config.load_json("path/to/xtts_config.json")  # You'll need the config file
            self.xtts_model = Xtts.init_from_config(config)
            self.xtts_model.load_checkpoint(config, checkpoint_dir="path/to/checkpoint")
            self.xtts_model.eval()
            if torch.cuda.is_available():
                self.xtts_model.cuda()
        except Exception as e:
            self.logger.error(f"Error initializing XTTS: {str(e)}")
            raise
            
    def preprocess_audio(self, audio_path: str) -> Optional[str]:
        """
        Preprocess a single audio file
        
        Args:
            audio_path (str): Path to input audio file
            
        Returns:
            Optional[str]: Path to processed audio file or None if processing fails
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Resample if necessary
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Check duration
            duration = librosa.get_duration(y=audio, sr=self.target_sr)
            if duration < self.min_duration or duration > self.max_duration:
                self.logger.warning(
                    f"Audio duration ({duration:.2f}s) outside acceptable range "
                    f"[{self.min_duration}-{self.max_duration}s]: {audio_path}"
                )
                return None
            
            # Remove silence
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
            
            # Save processed audio
            output_path = self.processed_dir / f"processed_{Path(audio_path).stem}.wav"
            sf.write(output_path, audio_trimmed, self.target_sr)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error processing audio file {audio_path}: {str(e)}")
            return None
            
    def extract_voice_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract voice embedding from processed audio file
        
        Args:
            audio_path (str): Path to processed audio file
            
        Returns:
            Optional[np.ndarray]: Voice embedding or None if extraction fails
        """
        try:
            # Load audio
            wav, sr = torchaudio.load(audio_path)
            
            # Convert to mono if necessary
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                wav = wav.cuda()
            
            # Extract embedding using XTTS
            with torch.no_grad():
                embedding = self.xtts_model.extract_speaker_embedding(wav)
            
            return embedding.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"Error extracting voice embedding: {str(e)}")
            return None
            
    def create_voice_profile(self, audio_files: List[str]) -> Dict:
        """
        Create a voice profile from multiple audio files
        
        Args:
            audio_files (List[str]): List of paths to audio files
            
        Returns:
            Dict: Voice profile data including embeddings and metadata
        """
        processed_files = []
        embeddings = []
        
        # Process each audio file
        for audio_file in audio_files:
            self.logger.info(f"Processing {audio_file}...")
            processed_path = self.preprocess_audio(audio_file)
            
            if processed_path:
                embedding = self.extract_voice_embedding(processed_path)
                if embedding is not None:
                    processed_files.append(processed_path)
                    embeddings.append(embedding)
        
        if not embeddings:
            raise ValueError("No valid embeddings could be extracted from the provided audio files")
        
        # Create average embedding
        average_embedding = np.mean(embeddings, axis=0)
        
        # Create profile data
        profile_data = {
            "average_embedding": average_embedding.tolist(),
            "individual_embeddings": [emb.tolist() for emb in embeddings],
            "processed_files": processed_files,
            "creation_timestamp": str(np.datetime64('now')),
            "metadata": {
                "num_samples": len(embeddings),
                "sample_rate": self.target_sr,
                "min_duration": self.min_duration,
                "max_duration": self.max_duration
            }
        }
        
        # Save profile
        profile_path = self.output_dir / "voice_profile.json"
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        self.logger.info(f"Voice profile saved to {profile_path}")
        return profile_data
    
    def validate_profile(self, profile_data: Dict) -> bool:
        """
        Validate a voice profile
        
        Args:
            profile_data (Dict): Voice profile data
            
        Returns:
            bool: True if profile is valid
        """
        required_keys = ["average_embedding", "individual_embeddings", "processed_files"]
        
        # Check required keys
        if not all(key in profile_data for key in required_keys):
            return False
        
        # Check embedding dimensions
        emb_dim = len(profile_data["average_embedding"])
        if not all(len(emb) == emb_dim for emb in profile_data["individual_embeddings"]):
            return False
            
        # Check processed files exist
        if not all(Path(f).exists() for f in profile_data["processed_files"]):
            return False
            
        return True

def main():
    # Example usage
    creator = VoiceProfileCreator(output_dir="custom_voice_profile")
    
    # List of audio files containing the target voice
    audio_files = [
        "path/to/voice_sample1.wav",
        "path/to/voice_sample2.wav",
        "path/to/voice_sample3.wav"
    ]
    
    try:
        # Create voice profile
        profile = creator.create_voice_profile(audio_files)
        
        # Validate profile
        if creator.validate_profile(profile):
            print("Voice profile created and validated successfully!")
        else:
            print("Voice profile validation failed!")
            
    except Exception as e:
        print(f"Error creating voice profile: {str(e)}")

if __name__ == "__main__":
    main()