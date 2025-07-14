import torch
import torchaudio
import os
import datetime
import numpy as np
import whisper
from scipy.signal import spectrogram, butter, filtfilt
from scipy.fft import fft, fftfreq
from pathlib import Path
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

class AudioProcessor:
    def __init__(self):
        self.model_filename = 'silero_vad_model.pt'
        self.local_repo_dir = os.path.join(os.path.dirname(__file__), 'silero-vad')
        self.vad_model, self.vad_utils = self.load_silero_vad()
        self.whisper_model = self.load_whisper_model()

        self.morse_freq_range = (800, 1500)  
        self.morse_sample_rate = 16000       
        self.morse_silence_threshold = 0.01  
        self.dot_dash_ratio = 3.0           
        self.min_symbol_samples = int(0.05 * self.morse_sample_rate)  
        
        self.dot_duration = None
        self.dash_duration = None
        self.symbol_gap = None
        self.letter_gap = None
        self.word_gap = None

    def load_silero_vad(self):
        print("Loading Silero VAD model via torch hub...")
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            model.eval()
            return model, utils
        except Exception as e:
            print(f"Error loading Silero VAD: {e}")
            raise

    def _download_and_save_model(self, model_to_save, path):
        try:
            torch.save(model_to_save.state_dict(), path)
            print(f"Веса модели успешно сохранены в: {path}")
        except Exception as e:
            print(f"Не удалось сохранить веса модели в {path}: {e}")

    def load_whisper_model(self):
        model_path = "/app/whisper-models/small.pt"
        print(f"Loading Whisper model from {model_path}")
        try:
            return whisper.load_model(model_path, device="cpu")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise

    def read_wav(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform.squeeze().numpy(), sample_rate

    def save_wav(self, file_path, data, sample_rate):
        tensor = torch.from_numpy(data).unsqueeze(0) if len(data.shape) == 1 else torch.from_numpy(data)
        torchaudio.save(file_path, tensor, sample_rate)

    def resample_audio(self, data, original_rate, target_rate):
        if original_rate == target_rate:
            return data
        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
        tensor = torch.from_numpy(data).float()
        return resampler(tensor).numpy()

    def bandpass_filter(self, data, lowcut, highcut, sample_rate, order=5):
        nyq = 0.5 * sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def decode_morse_segment(self, segment_data, sample_rate):
        try:
            segment_data = self.resample_audio(segment_data, sample_rate, self.morse_sample_rate)
            sample_rate = self.morse_sample_rate

            filtered = self.bandpass_filter(segment_data, *self.morse_freq_range, sample_rate)
            
            max_amp = np.max(np.abs(filtered))
            if max_amp < 1e-6: 
                return ""
            filtered = filtered / max_amp

            window_size = max(16, int(0.05 * sample_rate)) 
            energy = np.convolve(filtered**2, np.ones(window_size)/window_size, mode='same')
            
            med = np.median(energy)
            mad = np.median(np.abs(energy - med))
            threshold = max(self.morse_silence_threshold, med + 5 * mad)
            
            active = energy > threshold
            active_diff = np.diff(active.astype(int))
            
            starts = np.where(active_diff == 1)[0]
            ends = np.where(active_diff == -1)[0]

            if len(starts) == 0 or len(ends) == 0:
                return ""

            if starts[0] > ends[0]:
                ends = ends[1:]
            if len(starts) > len(ends):
                starts = starts[:len(ends)]
            elif len(ends) > len(starts):
                ends = ends[:len(starts)]

            if len(starts) == 0 or len(ends) == 0:
                return ""

            durations = [(end - start)/sample_rate for start, end in zip(starts, ends)]
            
            if not self.dot_duration:  
                median_duration = np.median(durations)
                self.dot_duration = median_duration
                self.dash_duration = median_duration * self.dot_dash_ratio
                self.symbol_gap = self.dot_duration * 1.5  
                self.letter_gap = self.dot_duration * 3.5 
                self.word_gap = self.dot_duration * 7.0   
            
            morse_code = []
            prev_end = ends[0]
            
            for i, (start, end, duration) in enumerate(zip(starts, ends, durations)):
                if i > 0:
                    gap = (start - prev_end)/sample_rate
                    if gap > self.word_gap:
                        morse_code.append(' / ')
                    elif gap > self.letter_gap:
                        morse_code.append(' ')
                
                if duration < self.dot_duration * 1.5:
                    morse_code.append('.')
                else:
                    morse_code.append('-')
                
                prev_end = end

            return ''.join(morse_code).strip()

        except Exception as e:
            print(f"Ошибка декодирования морзянки: {str(e)}")
            return ""

    def get_dominant_frequency(self, segment, sample_rate):
        n = len(segment)
        if n < 10: 
            return 0

        try:
            window = np.hamming(n)
            windowed = segment * window

            fft_result = fft(windowed)
            freqs = fftfreq(n, 1/sample_rate)

            mask = (freqs >= self.morse_freq_range[0]) & (freqs <= self.morse_freq_range[1])
            if not np.any(mask):  
                return 0

            psd = np.abs(fft_result[mask])**2
            if len(psd) == 0: 
                return 0

            peak_freq = freqs[mask][np.argmax(psd)]
            return peak_freq
        except:
            return 0  

    def detect_morse_segments(self, data, sample_rate):
        try:
            data = self.resample_audio(data, sample_rate, self.morse_sample_rate)
            sample_rate = self.morse_sample_rate

            filtered = self.bandpass_filter(data, *self.morse_freq_range, sample_rate)
            
            max_amp = np.max(np.abs(filtered))
            if max_amp == 0:
                return []
            filtered = filtered / max_amp

            window_size = max(16, int(0.05 * sample_rate))  
            energy = np.convolve(filtered**2, np.ones(window_size)/window_size, mode='same')
            
            rms = np.sqrt(np.mean(filtered**2))
            med = np.median(energy)
            mad = np.median(np.abs(energy - med))
            threshold = max(self.morse_silence_threshold, 0.5 * rms + 3 * mad)
            
            active = energy > threshold
            active_diff = np.diff(active.astype(int))
            
            starts = np.where(active_diff == 1)[0]
            ends = np.where(active_diff == -1)[0]

            if len(ends) == 0 or (len(starts) > 0 and starts[0] > ends[0]):
                starts = np.insert(starts, 0, 0)
            if len(starts) == 0 or (len(ends) > 0 and ends[-1] < starts[-1]):
                ends = np.append(ends, len(active)-1)

            if len(starts) == 0 or len(ends) == 0:
                return []

            merged_segments = []
            current_start = starts[0]
            current_end = ends[0]
            current_freq = self.get_dominant_frequency(filtered[current_start:current_end], sample_rate)

            for i in range(1, len(starts)):
                gap = starts[i] - ends[i-1]
                next_freq = self.get_dominant_frequency(filtered[starts[i]:ends[i]], sample_rate)

                if gap <= 0.02 * sample_rate and abs(current_freq - next_freq) < 50:
                    current_end = ends[i]
                else:
                    merged_segments.append({
                        'start': current_start,
                        'end': current_end,
                        'type': 'morse',
                        'duration': (current_end - current_start) / sample_rate,
                        'frequency': current_freq
                    })
                    current_start = starts[i]
                    current_end = ends[i]
                    current_freq = next_freq

            if current_end > current_start:
                merged_segments.append({
                    'start': current_start,
                    'end': current_end,
                    'type': 'morse',
                    'duration': (current_end - current_start) / sample_rate,
                    'frequency': current_freq
                })

            return [seg for seg in merged_segments if seg['duration'] >= 0.05]

        except Exception as e:
            print(f"Ошибка при обнаружении сегментов с морзянкой: {str(e)}")
            return []

    def merge_segments(self, segments, sample_rate, min_segment_duration=1.0, min_silence_duration=30.0):
        if not segments:
            return []

        min_segment_samples = int(min_segment_duration * sample_rate)
        min_silence_samples = int(min_silence_duration * sample_rate)

        segments = sorted(segments, key=lambda x: x['start'])

        merged_segments = []
        current_start = segments[0]['start']
        current_end = segments[0]['end']
        current_type = segments[0]['type']

        for seg in segments[1:]:
            if seg['start'] - current_end < min_silence_samples and seg['type'] == current_type:
                current_end = seg['end']
            else:
                if current_end - current_start >= min_segment_samples or current_type == 'morse':
                    merged_segments.append({
                        'start': current_start,
                        'end': current_end,
                        'type': current_type
                    })
                current_start = seg['start']
                current_end = seg['end']
                current_type = seg['type']

        if current_end - current_start >= min_segment_samples or current_type == 'morse':
            merged_segments.append({
                'start': current_start,
                'end': current_end,
                'type': current_type
            })

        return merged_segments

    def process_with_whisper(self, segment_data, sample_rate):
        temp_file = "temp_segment.wav"
        self.save_wav(temp_file, segment_data, sample_rate)

        audio = whisper.load_audio(temp_file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)

        _, probs = self.whisper_model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)

        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(self.whisper_model, mel, options)

        if os.path.exists(temp_file):
            os.remove(temp_file)

        return detected_lang, result.text

    def process_audio_file(self, input_file, output_dir="segments"):
        if input_file is None:
            input_file = os.path.join(project_root, "input.wav")
        if output_dir is None:
            output_dir = os.path.join(project_root, "segments")

        print(f"\nНачало обработки аудиофайла: {input_file}")
        
        try:
            data, sample_rate = self.read_wav(input_file)
            print("Аудиофайл успешно загружен.")
        except Exception as e:
            print(f"Ошибка при чтении аудиофайла: {e}")
            return

        print("Поиск сегментов с морзянкой...")
        morse_segments = self.detect_morse_segments(data, sample_rate)
        print(f"Найдено {len(morse_segments)} сегментов с морзянкой.")

        occupied = np.zeros(len(data), dtype=bool)
        for seg in morse_segments:
            occupied[seg['start']:seg['end']] = True

        speech_segments = []
        if len(morse_segments) < len(data) / (sample_rate * 0.1):
            print("Поиск речевых сегментов...")
            clean_data = data.copy()
            clean_data[occupied] = 0
            waveform = torch.from_numpy(clean_data).unsqueeze(0)

            get_speech_timestamps = self.vad_utils[0]
            speech_timestamps = get_speech_timestamps(
                waveform,
                self.vad_model,
                sampling_rate=sample_rate,
                min_speech_duration_ms=1000,
                min_silence_duration_ms=30000,
                speech_pad_ms=0,
            )

            speech_segments = [{
                'start': seg['start'],
                'end': seg['end'],
                'type': 'speech'
            } for seg in speech_timestamps]
            print(f"Найдено {len(speech_segments)} речевых сегментов.")
        else:
            print("Пропуск поиска речи - слишком много сегментов с морзянкой.")

        all_segments = morse_segments + speech_segments
        merged_segments = self.merge_segments(all_segments, sample_rate)
        print(f"Всего сегментов после объединения: {len(merged_segments)}")

        os.makedirs(output_dir, exist_ok=True)
        print(f"Сохранение результатов в директорию: {output_dir}")

        for i, segment in enumerate(merged_segments):
            start_sample = segment['start']
            end_sample = segment['end']
            segment_data = data[start_sample:end_sample]

            start_time_sec = start_sample / sample_rate
            duration_sec = (end_sample - start_sample) / sample_rate

            base_time = datetime.datetime(1970, 1, 1)
            segment_time = base_time + datetime.timedelta(seconds=start_time_sec)
            start_str = segment_time.strftime("%Y-%m-%d-%H-%M-%S")

            hours, rem = divmod(duration_sec, 3600)
            minutes, seconds = divmod(rem, 60)
            duration_str = f"{int(hours):02d}-{int(minutes):02d}-{int(seconds):02d}"

            if segment['type'] == 'morse':
                print(f"\nОбработка сегмента с морзянкой {i+1}/{len(merged_segments)}...")
                morse_code = self.decode_morse_segment(segment_data, sample_rate)
                detected_lang = "morse"
                recognized_text = f"[МОРЗЯНКА: {morse_code}]"
            else:
                print(f"\nОбработка речевого сегмента {i+1}/{len(merged_segments)}...")
                detected_lang, recognized_text = self.process_with_whisper(segment_data, sample_rate)

            base_filename = f"start_{start_str}_dur_{duration_str}_{detected_lang}"

            audio_output_path = os.path.join(output_dir, f"{base_filename}.wav")
            self.save_wav(audio_output_path, segment_data, sample_rate)

            text_output_path = os.path.join(output_dir, f"{base_filename}.txt")
            with open(text_output_path, 'w', encoding='utf-8') as text_file:
                text_file.write(recognized_text)

            print(f"Сохранен файл: {audio_output_path}")
            print(f"Длительность: {duration_sec:.2f} сек, Тип: {segment['type']}, Язык: {detected_lang}")
            print(f"Распознанный текст: {recognized_text[:100]}...") 

        print("\nОбработка файла завершена успешно!")

if __name__ == "__main__":
    print("=== Анализатор аудиофайлов ===")
    print("Программа для обнаружения морзянки и речи в аудиозаписях")
    
    processor = AudioProcessor()
    input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.wav")

    if os.path.exists(input_file):
        print(f"\nНайден входной файл: {input_file}")
        processor.process_audio_file(input_file)
    else:
        print(f"\nОшибка: файл {input_file} не найден.")
        print("Пожалуйста, поместите аудиофайл в формате WAV в ту же директорию и назовите его 'input.wav'")