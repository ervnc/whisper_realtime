from faster_whisper import WhisperModel
import wave
import pyaudio
import os

def process_audio(py, stream, file_path, chunk_length=5):
    frames = []
    num_frames = int(16000 * chunk_length / 1024)
    print(num_frames)
    for _ in range(0, num_frames):
        try:
            data = stream.read(1024)
            frames.append(data)
        except Exception as e:
            print(f"Erro na leitura do stream: {e}")
            continue
    print(f"Número de frames capturados: {len(frames)}")

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(py.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_chunk(model, chunk_file):
    segments, info = model.transcribe(chunk_file, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    transcription = ""
    for segment in segments:
        transcription += "%s\n" % (segment.text)
        print("Transcription: %s" % segment.text)
    return transcription

def main():
    model_size = "tiny.en"
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
            chunk_file = "chunk.wav"
            process_audio(py, stream, chunk_file)

            if os.path.exists(chunk_file):
                transcription = transcribe_chunk(model, chunk_file)
                print(transcription)
                os.remove(chunk_file)

                a_transcription += transcription + " "
            else:
                print("Arquivo não encontrado")
    except KeyboardInterrupt:
        print("Transcrição finalizada")
    finally:
        print("Transcrição final: %s" % a_transcription)
        stream.stop_stream()
        stream.close()
        py.terminate()

if __name__ == "__main__":
    main()
