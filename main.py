import argparse
import numpy as np
import speech_recognition as sr
import whisper
import torch
from queue import Queue
from time import sleep
import os
from threading import Thread

def process_audio(audio_model, data_queue):
  transcription = ""
  print("Modelo carregado. Transcrição em tempo real...\n")

  while True:
      if not data_queue.empty():
        audio_data = data_queue.get()

        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcrição
        result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
        text = result['text'].strip()

        transcription += text + " "
        os.system('cls' if os.name == 'nt' else 'clear')
        print(transcription)

      sleep(0.25)

def main(model_name):
    audio_model = whisper.load_model(model_name)

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 1000
    recognizer.dynamic_energy_threshold = False

    # Listar os dispositivos de microfone disponíveis
    print("Microfones disponíveis:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{index}: {name}")
    
    mic_index = int(input("Digite o índice do microfone que deseja usar: "))
    mic = sr.Microphone(sample_rate=16000, device_index=mic_index)

    # Fila para armazenar os dados de áudio
    data_queue = Queue()

    # Função de callback para capturar o áudio
    def record_callback(_, audio: sr.AudioData):
        data = audio.get_raw_data()
        data_queue.put(data)

    with mic:
        recognizer.adjust_for_ambient_noise(mic)

    # Captura de áudio em segundo plano
    recognizer.listen_in_background(mic, record_callback)    

    processing_thread = Thread(target=process_audio, args=(audio_model, data_queue))
    processing_thread.daemon = True
    processing_thread.start()

    try:
      while True:
        sleep(1)
    except KeyboardInterrupt:
      print("\nTranscrição finalizada.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Transcrição de áudio em tempo real")
  # Modelo inicial: tiny
  parser.add_argument("--model", type=str, default="tiny", help="Nome do modelo (ex: tiny, base, small)")

  args = parser.parse_args()

  main(args.model)
