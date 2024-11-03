from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import webrtcvad

def process_audio(py, stream, chunk_length=2):
    frames = []
    num_frames = int(16000 * chunk_length / 1024)
    for _ in range(num_frames):
        try:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        except Exception as e:
            print(f"Erro na leitura do stream: {e}")
            continue

    audio_data = b''.join(frames)
    return audio_data

def is_speech(audio_data, sample_rate=16000):
    vad = webrtcvad.Vad(3)
    frame_duration = 30
    frame_size = int(sample_rate * frame_duration / 1000) * 2 

    is_speech_detected = False
    for i in range(0, len(audio_data) - frame_size + 1, frame_size):
        frame = audio_data[i:i + frame_size]
        if len(frame) < frame_size:
            continue
        if vad.is_speech(frame, sample_rate):
            is_speech_detected = True
            break
    return is_speech_detected

def transcribe_chunk(model, audio_data):
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = model.transcribe(audio_np, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += "%s\n" % (segment.text)
        print("Transcrição: %s" % segment.text)
    return transcription

def main():
    model_size = "small.en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    py = pyaudio.PyAudio()
    print("Microfones disponíveis:")
    for i in range(py.get_device_count()):
        print(f"{i}: {py.get_device_info_by_index(i).get('name')}")

    mic_index = int(input("Digite o índice do microfone que deseja usar: "))
    stream = py.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=mic_index)

    a_transcription = ""

    try:
        while True:
            audio_data = process_audio(py, stream, chunk_length=2)

            if audio_data:
                if is_speech(audio_data):
                    transcription = transcribe_chunk(model, audio_data)
                    if transcription.strip():
                        print(transcription)
                        a_transcription += transcription + " "
                else:
                    print("Silêncio detectado")
            else:
                print("Nenhum dado de áudio recebido")
    except KeyboardInterrupt:
        print("Transcrição finalizada")
    finally:
        print("Transcrição final: %s" % a_transcription)
        stream.stop_stream()
        stream.close()
        py.terminate()

if __name__ == "__main__":
    main()
