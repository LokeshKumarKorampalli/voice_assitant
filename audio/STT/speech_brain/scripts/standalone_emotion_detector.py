# standalone_emotion_detector.py
import os
import json
import tempfile
import wave
from datetime import datetime

# Check for dependencies
try:
    import torch
    import librosa
    import numpy as np
    import pyaudio
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    DEPENDENCIES_OK = False

# Check for SpeechBrain
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

class StandaloneEmotionDetector:
    
    
    def __init__(self):
        if not DEPENDENCIES_OK:
            raise RuntimeError("Required dependencies not installed")
        
        print("Initializing Standalone Emotion Detector...")
        
        # Configuration
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        self.audio_format = pyaudio.paInt16
        self.emotions = ['angry', 'happy', 'neutral', 'sad']
        
        # Load model
        self.model = None
        self.use_speechbrain = False
        
        if SPEECHBRAIN_AVAILABLE:
            try:
                print("Loading SpeechBrain model...")
                self.model = EncoderClassifier.from_hparams(
                    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                    savedir="models/emotion_model"
                )
                self.use_speechbrain = True
                print("SpeechBrain model loaded successfully")
            except Exception as e:
                print(f"SpeechBrain loading failed: {e}")
                print("Using basic feature analysis")
        
        if not self.use_speechbrain:
            print("Using basic audio feature analysis")
        
        print("Initialization complete")
    
    def extract_audio_features(self, audio_data):
        """Extract comprehensive audio features"""
        features = {}
        
        # Energy features
        rms_energy = librosa.feature.rms(y=audio_data)[0]
        features['energy_mean'] = float(np.mean(rms_energy))
        features['energy_std'] = float(np.std(rms_energy))
        features['energy_max'] = float(np.max(rms_energy))
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
        pitch_values = pitches[pitches > 0]
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zero_crossing_rate'] = float(np.mean(zcr))
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            features['tempo'] = float(tempo)
        except:
            features['tempo'] = 0.0
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_rolloff'] = float(np.mean(rolloff))
        
        return features
    
    def analyze_with_speechbrain(self, audio_data):
        """Analyze emotion using SpeechBrain model"""
        try:
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
            out_prob, score, index, text_lab = self.model.classify_batch(audio_tensor)
            
            probabilities = torch.nn.functional.softmax(out_prob, dim=-1)
            prob_values = probabilities.squeeze().numpy()
            
            probability_dict = {}
            for i, emotion in enumerate(self.emotions):
                if i < len(prob_values):
                    probability_dict[emotion] = float(prob_values[i])
                else:
                    probability_dict[emotion] = 0.0
            
            return {
                'predicted_emotion': text_lab[0],
                'confidence': float(score.squeeze()),
                'probabilities': probability_dict,
                'method': 'speechbrain'
            }
        except Exception as e:
            print(f"SpeechBrain analysis failed: {e}")
            return None
    
    def analyze_with_basic_features(self, audio_features):
        """Analyze emotion using basic audio features"""
        probabilities = {emotion: 0.0 for emotion in self.emotions}
        
        energy_mean = audio_features.get('energy_mean', 0.0)
        energy_std = audio_features.get('energy_std', 0.0)
        pitch_mean = audio_features.get('pitch_mean', 0.0)
        pitch_std = audio_features.get('pitch_std', 0.0)
        spectral_centroid = audio_features.get('spectral_centroid_mean', 0.0)
        zcr = audio_features.get('zero_crossing_rate', 0.0)
        tempo = audio_features.get('tempo', 0.0)
        
        # Angry: high energy, high pitch variation, high spectral centroid
        angry_score = 0.0
        if energy_mean > 0.025:
            angry_score += 0.3
        if pitch_std > 60:
            angry_score += 0.25
        if spectral_centroid > 2500:
            angry_score += 0.2
        if zcr > 0.08:
            angry_score += 0.15
        if tempo > 100:
            angry_score += 0.1
        
        # Happy: moderate-high energy, higher pitch, bright voice
        happy_score = 0.0
        if 0.015 < energy_mean < 0.035:
            happy_score += 0.25
        if pitch_mean > 180:
            happy_score += 0.2
        if spectral_centroid > 2000:
            happy_score += 0.2
        if 30 < pitch_std < 80:
            happy_score += 0.15
        if 80 < tempo < 120:
            happy_score += 0.1
        if 0.04 < zcr < 0.08:
            happy_score += 0.1
        
        # Sad: low energy, lower pitch, small variation
        sad_score = 0.0
        if energy_mean < 0.015:
            sad_score += 0.3
        if pitch_mean < 150:
            sad_score += 0.25
        if pitch_std < 40:
            sad_score += 0.2
        if spectral_centroid < 1800:
            sad_score += 0.15
        if tempo < 80:
            sad_score += 0.1
        
        # Neutral: moderate values
        neutral_score = 0.0
        if 0.01 < energy_mean < 0.025:
            neutral_score += 0.2
        if 140 < pitch_mean < 200:
            neutral_score += 0.2
        if 20 < pitch_std < 60:
            neutral_score += 0.2
        if 1500 < spectral_centroid < 2500:
            neutral_score += 0.2
        if 70 < tempo < 110:
            neutral_score += 0.1
        if 0.03 < zcr < 0.07:
            neutral_score += 0.1
        
        # Assign scores with base probability
        base_prob = 0.1
        probabilities['angry'] = min(1.0, angry_score + base_prob)
        probabilities['happy'] = min(1.0, happy_score + base_prob)
        probabilities['sad'] = min(1.0, sad_score + base_prob)
        probabilities['neutral'] = min(1.0, neutral_score + base_prob)
        
        # Normalize
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {emotion: prob/total_prob for emotion, prob in probabilities.items()}
        
        predicted_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_emotion]
        
        return {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': probabilities,
            'method': 'basic_features'
        }
    
    def get_confidence_level(self, score):
        """Categorize confidence score"""
        if score >= 0.7:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def format_results(self, emotion_data, audio_features=None, metadata=None):
        """Format results with structured output"""
        primary_emotion = emotion_data.get('predicted_emotion', 'unknown')
        primary_confidence = emotion_data.get('confidence', 0.0)
        all_probabilities = emotion_data.get('probabilities', {})
        
        # Create emotion breakdown
        breakdown = []
        for emotion, score in all_probabilities.items():
            breakdown.append({
                'emotion_label': emotion,
                'confidence_score': round(score, 3),
                'confidence_level': self.get_confidence_level(score),
                'percentage': round(score * 100, 1)
            })
        
        # Sort by confidence
        breakdown.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Find secondary emotion
        sorted_emotions = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
        secondary_emotion = sorted_emotions[1] if len(sorted_emotions) > 1 else (None, 0.0)
        
        # Calculate certainty gap
        certainty_gap = primary_confidence - secondary_emotion[1] if secondary_emotion[0] else primary_confidence
        
        # Create structured results
        results = {
            'primary_emotion': {
                'label': primary_emotion,
                'confidence_score': round(primary_confidence, 3),
                'confidence_level': self.get_confidence_level(primary_confidence)
            },
            'emotion_breakdown': breakdown,
            'analysis_summary': {
                'primary_emotion': primary_emotion,
                'secondary_emotion': secondary_emotion[0] if secondary_emotion[0] else 'none',
                'certainty_gap': round(certainty_gap, 3),
                'is_ambiguous': certainty_gap < 0.2,
                'reliability': self.assess_reliability(primary_confidence, certainty_gap)
            },
            'insights': {
                'decision_confidence': self.get_confidence_level(primary_confidence),
                'requires_verification': primary_confidence < 0.6,
                'emotion_intensity': self.assess_emotion_intensity(primary_confidence),
                'recommended_actions': self.suggest_actions(primary_emotion, primary_confidence)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            results['metadata'] = metadata
        
        if audio_features:
            results['audio_features'] = audio_features
        
        return results
    
    def assess_reliability(self, confidence, certainty_gap):
        """Assess prediction reliability"""
        if confidence >= 0.8 and certainty_gap >= 0.3:
            return 'very_high'
        elif confidence >= 0.6 and certainty_gap >= 0.2:
            return 'high'
        elif confidence >= 0.4:
            return 'moderate'
        else:
            return 'low'
    
    def assess_emotion_intensity(self, confidence):
        """Assess emotion intensity"""
        if confidence >= 0.8:
            return 'very_strong'
        elif confidence >= 0.6:
            return 'strong'
        elif confidence >= 0.4:
            return 'moderate'
        else:
            return 'weak'
    
    def suggest_actions(self, emotion, confidence):
        """Suggest actions based on emotion and confidence"""
        actions = []
        
        if confidence < 0.5:
            actions.append('verify_with_additional_samples')
        
        if emotion == 'angry':
            actions.extend(['de_escalation_protocols', 'priority_handling'])
        elif emotion == 'sad':
            actions.extend(['empathetic_response', 'support_resources'])
        elif emotion == 'happy':
            actions.extend(['positive_reinforcement', 'engagement_opportunity'])
        else:
            actions.append('standard_processing')
        
        return actions
    
    def load_audio_file(self, file_path):
        """Load and preprocess audio file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            audio_data, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            if len(audio_data) == 0:
                raise ValueError("Audio file contains no data")
            
            # Normalize audio
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            return audio_data
        except Exception as e:
            raise ValueError(f"Failed to process audio file: {str(e)}")
    
    def record_audio(self, duration=5):
        """Record audio from microphone"""
        print(f"Recording for {duration} seconds...")
        print("Speak clearly into the microphone...")
        
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            
            for i in range(int(self.sample_rate / self.chunk_size * duration)):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            print("Recording completed")
            
        except Exception as e:
            raise RuntimeError(f"Audio recording failed: {str(e)}")
        finally:
            audio.terminate()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        try:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            return temp_file.name
        except Exception as e:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise RuntimeError(f"Failed to save recording: {str(e)}")
    
    def analyze_audio_file(self, file_path):
        """Analyze emotion from audio file"""
        print(f"Analyzing emotion from: {file_path}")
        
        # Load audio
        audio_data = self.load_audio_file(file_path)
        
        # Extract features
        audio_features = self.extract_audio_features(audio_data)
        
        # Analyze emotion
        if self.use_speechbrain:
            emotion_results = self.analyze_with_speechbrain(audio_data)
            if emotion_results is None:
                emotion_results = self.analyze_with_basic_features(audio_features)
        else:
            emotion_results = self.analyze_with_basic_features(audio_features)
        
        # Create metadata
        metadata = {
            'source_file': file_path,
            'analysis_method': emotion_results.get('method', 'unknown'),
            'audio_duration': len(audio_data) / self.sample_rate,
            'sample_rate': self.sample_rate
        }
        
        # Format results
        formatted_results = self.format_results(
            emotion_data=emotion_results,
            audio_features=audio_features,
            metadata=metadata
        )
        
        return formatted_results
    
    def analyze_live_recording(self, duration=5):
        """Record and analyze live audio"""
        print("Starting live audio recording and analysis...")
        
        try:
            temp_file = self.record_audio(duration)
            results = self.analyze_audio_file(temp_file)
            
            # Update metadata
            if 'metadata' in results:
                results['metadata']['source_type'] = 'live_recording'
                results['metadata']['source_file'] = 'live_microphone_input'
            
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            
            return results
            
        except Exception as e:
            return {
                'error': True,
                'error_message': f"Live recording error: {str(e)}"
            }
    
    def print_results(self, results):
        """Print formatted results"""
        if 'error' in results:
            print(f"Error: {results['error_message']}")
            return
        
        print("\n" + "="*60)
        print("EMOTION DETECTION ANALYSIS RESULTS")
        print("="*60)
        
        # Primary emotion
        primary = results['primary_emotion']
        print(f"PRIMARY EMOTION: {primary['label'].upper()}")
        print(f"Confidence Score: {primary['confidence_score']}")
        print(f"Confidence Level: {primary['confidence_level'].upper()}")
        
        # Emotion breakdown
        print(f"\nEMOTION BREAKDOWN:")
        for emotion in results['emotion_breakdown']:
            label = emotion['emotion_label'].capitalize()
            score = emotion['confidence_score']
            level = emotion['confidence_level']
            percentage = emotion['percentage']
            bar = "█" * int(score * 20)
            print(f"  {label:8} | {score:.3f} | {percentage:5.1f}% | {level:6} | {bar}")
        
        # Analysis summary
        summary = results['analysis_summary']
        print(f"\nANALYSIS SUMMARY:")
        print(f"Primary: {summary['primary_emotion']}")
        print(f"Secondary: {summary['secondary_emotion']}")
        print(f"Certainty Gap: {summary['certainty_gap']}")
        print(f"Reliability: {summary['reliability']}")
        print(f"Ambiguous: {'Yes' if summary['is_ambiguous'] else 'No'}")
        
        # Insights
        insights = results['insights']
        print(f"\nINSIGHTS:")
        print(f"Decision Confidence: {insights['decision_confidence']}")
        print(f"Emotion Intensity: {insights['emotion_intensity']}")
        print(f"Requires Verification: {'Yes' if insights['requires_verification'] else 'No'}")
        
        # Actions
        print(f"\nRECOMMENDED ACTIONS:")
        for action in insights['recommended_actions']:
            print(f"  - {action.replace('_', ' ').title()}")
        
        print("="*60)
    
    def export_to_json(self, results, file_path=None):
        """Export results to JSON"""
        json_output = json.dumps(results, indent=2, ensure_ascii=False)
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_output)
            return file_path
        else:
            return json_output

def main():
    """Main function"""
    print("Standalone Emotion Detection System")
    print("Structured Output with Labels and Confidence Scores")
    print("="*60)
    
    if not DEPENDENCIES_OK:
        print("Error: Required dependencies not installed")
        print("Install with: pip install torch librosa numpy pyaudio")
        return
    
    try:
        detector = StandaloneEmotionDetector()
    except Exception as e:
        print(f"Failed to initialize detector: {str(e)}")
        return
    
    print(f"SpeechBrain Available: {SPEECHBRAIN_AVAILABLE}")
    print(f"Using SpeechBrain: {detector.use_speechbrain}")
    
    while True:
        print(f"\nOptions:")
        print("1. Record and analyze live audio")
        print("2. Analyze audio file")
        print("3. Exit")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == '1':
            duration_input = input("Recording duration in seconds (default 5): ").strip()
            try:
                duration = float(duration_input) if duration_input else 5.0
            except ValueError:
                duration = 5.0
            
            results = detector.analyze_live_recording(duration)
            detector.print_results(results)
            
            if 'error' not in results:
                export_choice = input("\nExport to JSON? (y/n): ").strip().lower()
                if export_choice == 'y':
                    filename = f"emotion_analysis_{results['timestamp'].replace(':', '-')}.json"
                    detector.export_to_json(results, filename)
                    print(f"Results exported to: {filename}")
        
        elif choice == '2':
            file_path = input("Enter audio file path: ").strip()
            
            if not os.path.exists(file_path):
                print("File not found!")
                continue
            
            results = detector.analyze_audio_file(file_path)
            detector.print_results(results)
            
            if 'error' not in results:
                export_choice = input("\nExport to JSON? (y/n): ").strip().lower()
                if export_choice == 'y':
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    filename = f"emotion_analysis_{base_name}.json"
                    detector.export_to_json(results, filename)
                    print(f"Results exported to: {filename}")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()