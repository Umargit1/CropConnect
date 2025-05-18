# ————————————————————————————————
# ALL HELPER FUNCTIONS
# ————————————————————————————————
import os
import requests
from groq import Groq
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import gradio as gr

# — Groq client setup
API_KEY = "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"
client = Groq(api_key=API_KEY)

def farmer_chatbot(prompt, model="meta-llama/llama-4-scout-17b-16e-instruct",
                   temperature=1.0, max_completion_tokens=200, top_p=1.0):
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        stream=True,
    )
    text = ""
    for chunk in completion:
        delta = chunk.choices[0].delta.content
        if delta: 
            text += delta
    return text.strip()

# — Image model setup
processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
image_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
image_model.eval()

def classify_crop_health(image_path):
    image = Image.fromarray(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = image_model(**inputs)
    idx = outputs.logits.argmax(-1).item()
    return image_model.config.id2label[idx]

def get_crop_health_advice(image_path):
    disease = classify_crop_health(image_path)
    prompt = f"My crop shows signs of {disease}. What should I do?"
    return farmer_chatbot(prompt)

# — Weather (OpenWeatherMap)
OWM_API_KEY = "f405deec8dcfd711e4596069769a4de4"
def get_weather_update(location):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": OWM_API_KEY, "units": "metric", "lang": "en"}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return f"Error {r.status_code}: {r.json().get('message','')}"
    d = r.json()
    return (
        f"Weather in {d['name']},{d['sys']['country']}:\n"
        f" • {d['weather'][0]['description'].title()}, {d['main']['temp']}°C\n"
        f" • Humidity {d['main']['humidity']}%, Wind {d['wind']['speed']} m/s"
    )

def get_weather_chat(location):
    info = get_weather_update(location)
    prompt = info + "\nAs an expert farm advisor, what should the farmer do given this weather?"
    return farmer_chatbot(prompt)

# — Pest & Disease
def get_pest_disease_advice(issue):
    prompt = (f"I am dealing with {issue} on my farm. "
              "Please provide practical, low-cost, and environmentally friendly management strategies.")
    return farmer_chatbot(prompt)

# — Soil Health
def get_soil_health_suggestions():
    prompt = ("As an expert agronomist, what are the best low-cost, "
              "sustainable practices a farmer can follow to improve and maintain soil health?")
    return farmer_chatbot(prompt, max_completion_tokens=250)

# — Market Price
def get_market_price_update(crop):
    prompt = (f"As a market analyst specializing in agriculture, "
              f"what is the current market price per quintal of {crop}? Please answer in PKR and give the date.")
    return farmer_chatbot(prompt, max_completion_tokens=150)

# — Government Schemes
def get_government_schemes(region="Punjab, Pakistan"):
    prompt = (f"As an agricultural policy expert, list the active "
              f"government schemes and subsidies in {region}, including eligibility and application steps.")
    return farmer_chatbot(prompt, max_completion_tokens=250)

# ————————————————————————————————
# GRADIO UI SETUP
# ————————————————————————————————
with gr.Blocks() as demo:
    gr.Markdown("## 🌾 Farmer Assistant Chatbot")

    with gr.Tab("Crop Health Analysis"):
        img_input = gr.Image(label="Upload Crop Image")
        img_output = gr.Textbox(label="Diagnosis & Advice")
        img_button = gr.Button("Analyze Crop Health")
        img_button.click(get_crop_health_advice, inputs=img_input, outputs=img_output)

    with gr.Tab("Weather Information"):
        loc_input = gr.Textbox(label="Enter Location")
        loc_output = gr.Textbox(label="Weather Information & Advice")
        loc_button = gr.Button("Get Weather Info")
        loc_button.click(get_weather_chat, inputs=loc_input, outputs=loc_output)

    with gr.Tab("Pest & Disease Management"):
        pest_input = gr.Textbox(label="Enter Pest or Disease")
        pest_output = gr.Textbox(label="Management Strategies")
        pest_button = gr.Button("Get Pest/Disease Advice")
        pest_button.click(get_pest_disease_advice, inputs=pest_input, outputs=pest_output)

    with gr.Tab("Soil Health Improvement"):
        soil_output = gr.Textbox(label="Soil Health Suggestions")
        soil_button = gr.Button("Get Soil Health Advice")
        soil_button.click(get_soil_health_suggestions, inputs=None, outputs=soil_output)

    with gr.Tab("Market Prices for Crops"):
        crop_input = gr.Textbox(label="Enter Crop Name")
        crop_output = gr.Textbox(label="Market Price Information")
        crop_button = gr.Button("Get Market Price")
        crop_button.click(get_market_price_update, inputs=crop_input, outputs=crop_output)

    with gr.Tab("Government Schemes and Subsidies"):
        region_input = gr.Textbox(label="Enter Region (Default: Punjab, Pakistan)")
        region_output = gr.Textbox(label="Government Schemes & Subsidies")
        region_button = gr.Button("Get Schemes Info")
        region_button.click(get_government_schemes, inputs=region_input, outputs=region_output)

demo.launch(share=True)
