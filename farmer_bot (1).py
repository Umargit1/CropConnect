# ————————————————————————————————
# ALL HELPER FUNCTIONS
# ————————————————————————————————
import os, requests
from groq import Groq
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image
import torch

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
        if delta: text += delta
    return text.strip()

# — Image model setup
processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
image_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
image_model.eval()

def classify_crop_health(image_path):
    image = Image.open(image_path).convert("RGB")
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

# farmer_bot.py
# ————————————————————————————————
# (all of your existing imports + helper functions go here)
# ————————————————————————————————

if __name__ == "__main__":
    import os
    os.environ["GROQ_API_KEY"] = "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"
    # 1) LangChain imports
    from langchain.agents import Tool, initialize_agent
    from langchain_groq import ChatGroq

    # 2) Instantiate your Groq LLM
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


    # 3) Wrap your functions as LangChain Tools
    tools = [
        Tool(
            name="CropHealth",
            func=lambda path: get_crop_health_advice(path),
            description="Upload an image path to diagnose crop disease and get advice."
        ),
        Tool(
            name="Weather",
            func=lambda loc: get_weather_chat(loc),
            description="Get current weather and farm-specific advice for a location."
        ),
        Tool(
            name="PestDisease",
            func=lambda issue: get_pest_disease_advice(issue),
            description="Get management strategies for a pest or crop disease."
        ),
        Tool(
            name="SoilHealth",
            func=lambda _: get_soil_health_suggestions(),
            description="Get best practices to improve and maintain soil health."
        ),
        Tool(
            name="MarketPrice",
            func=lambda crop: get_market_price_update(crop),
            description="Get the current market price for a given crop."
        ),
        Tool(
            name="Schemes",
            func=lambda region="Punjab, Pakistan": get_government_schemes(region),
            description="List government schemes & subsidies for farmers in a region."
        ),
    ]

    # 4) Create the LangChain Agent
    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    # 5) Choose your interface:

    # ─── a) Console REPL ─────────────────────────────────────────────────
    print("👩‍🌾 Farmer Assistant is online. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ("exit", "quit"):
            break
        print(agent.run(query))

    # ─── b) (Optional) Gradio Web UI ──────────────────────────────────────
    # Uncomment this block if you’d rather launch a simple web interface:
    #
    # import gradio as gr
    #
    # def chat_fn(user_input):
    #     return agent.run(user_input)
    #
    # demo = gr.Interface(fn=chat_fn,
    #                     inputs=gr.Textbox(lines=2, placeholder="Ask anything…"),
    #                     outputs="text",
    #                     title="Farmer Assistant Chatbot")
    # demo.launch(share=True)
