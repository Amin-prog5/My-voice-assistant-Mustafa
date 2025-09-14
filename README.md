# My-voice-assistant-Mustafa

1. Imports

Brings in libraries for voice recognition, text-to-speech, face recognition, web requests, and large language models (LLMs).

Examples: cv2 for camera, speech_recognition for microphone input, kokoro for TTS, llama_cpp for running a local LLM.

2. Setup

Loads API key for weather data (OpenWeatherMap).

Initializes Kokoro TTS pipeline for generating speech.

Defines load_voice() to load voice models.

Loads a local LLM model (Mistral-7B) with memory and GPU settings.

3. Functions

f_r() → Face Recognition
Uses the webcam to detect a face and compare it with a known image (amin.jpg). Returns either "ameen" or "unknown".

g_a() → Voice Input
Listens to microphone input, converts speech to text using Google Speech Recognition, and returns the recognized string.

get_weather(keyword) → Weather Info
Fetches live weather data for a city using OpenWeatherMap API and returns condition + temperature.

open_youtube_search(query) → YouTube Search
Opens YouTube with a search for the given query.

sad(response) → Speak Response
Uses Kokoro TTS to convert text into voice and play it with sounddevice.
