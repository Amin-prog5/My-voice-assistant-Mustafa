import os
import webbrowser
from urllib.parse import quote_plus
import cv2
import face_recognition
import numpy as np
import requests
import sounddevice as sd
import speech_recognition as sr
import torch
import wikipedia
from AppOpener import open
from kokoro import KPipeline
from llama_cpp import Llama

api_key = 'e5bf8b5f17793c783f96aa387038bea1'

pipeline = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")

voices_path = r"C:\Users\Amin\.cache\huggingface\hub\models--hexgrad--Kokoro-82M\snapshots\offline\voices"
def load_voice(v): return torch.load(os.path.join(voices_path, f"{v}.pt"), map_location="cpu")
pipeline.load_voice = load_voice

# load llm and configr
llm = Llama(
    model_path="models/Mistral-7B-Instruct-v0.1.Q3_K_M.gguf",

    n_ctx=512,
    n_gpu_layers=20,
    n_batch=512,
    f16_kv=True,
    use_mlock= True,
    low_vram=True,
)


# face reco funn
def f_r():

    amin_image = face_recognition.load_image_file(r"C:\Users\Amin\PycharmProjects\amin.jpg")

    amin_face_encoding = face_recognition.face_encodings(amin_image)[0]

    known_face_encodings = [amin_face_encoding]
    known_face_names = ["ameen"]

    id_map = {"ameen": 5}

    video_capture = cv2.VideoCapture(0)
    name = "unknown"

    while True:
        ret, frame = video_capture.read()
        if not ret:
            name = "unknown"
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1].astype(np.uint8)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                else:
                    name = "unknown"

            break

    video_capture.release()
    return name


# listening fun
def g_a():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1.2)
        print("Say something sirr")
        audio = r.listen(source)

        try:
           # audio = r.listen(source, timeout=5, phrase_time_limit=8)
            text = r.recognize_google(audio)
            print("You said :", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, Come again ! ")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results ; {e}")
            return ""


# get weather fun
def get_weather(keyword):
    user_city = keyword

    weather_data = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?q={user_city}&units=metric&APPID={api_key}")

    weather = weather_data.json()['weather'][0]['main']
    tem = round(weather_data.json()['main']['temp'])
    temp = "And the temperature is "+str(tem) + " celsius"
    print(temp)
    print(weather)
    return weather, temp

def open_youtube_search(query):
    url = "https://www.youtube.com/results?search_query=" + quote_plus(query)
    print("Opening:", url)
    webbrowser.open(url, new=2)


# Speakkk fun
def sad(response):
    for _, _, audio in pipeline(response, "am_santa"):
        sd.play(audio, 24000)
        sd.wait()

ssp = "やっほ〜アミンさんっ！今日も元気いっぱいかな？会えてうれしいよ〜！"
intro = "Hello!- [Ameen](+1), my name is [Mustafa](/ˈmostɑfɑ/), I'm your voice assistant. How can I help you ?"
intro1 =  "Hello!- Ameen, I'm your voice assistant . my name is [Mustafa](/ˈmostɑfɑ/), call me if you need any thing  ?"

sad(intro1)
while True:

    keyword = g_a().lower()
    if ("bye" in keyword or "see you later" in keyword):
        sad("it was honour to help you and, see you later")
        break

# the key word
    while ( "mustafa" in keyword or "moustafa" in keyword):

        o=""
        print("welcome back amin ")
        sad("how can i help you?")

        o = g_a()

        if "weather" in o or "today" in o:
            sad("in which  city?")
            city = g_a()
            t = get_weather(city)
            sad(t)
            break

        if "look " in o or "something " in o or "search" in o:
            sad("what should i search for ")
            sh = g_a()
            res1 = wikipedia.summary(sh, sentences=2)
            print(res1)
            sad(res1)
            break

        if "open" in o or "app " in o:
            sad("what should i open?")
            app = g_a()
            open(app,match_closest=True)
            break

        if "favorite" in  o or "of all time" in o:
            url = "https://www.youtube.com/watch?v=KD5aqhUytbk"
            webbrowser.open_new_tab(url)
            break

        if "play" in o:
            sad(" what should i play ?")
            play = g_a()
            open_youtube_search(play)
            break

        if "chat " in o or "mode" in o or "with you" in o:
            sad("in which subject ")
            o1 = g_a()
            prompt = "<s>[INST]" +o1+" [/INST]"

            output = llm(prompt, max_tokens=70, stop=["</s>"])

            print( output["choices"][0]["text"].strip())

            res = output["choices"][0]["text"].strip()
            sad(res)
            break

        if ("goodbye" in o):
            sad("bye bye")
            break

        if "who am" in o or "i" in o:
            print("you are : " + f_r())
            me = ",!"+f_r()
            sad(me +".!"+ " the guy, who made me")
            break




