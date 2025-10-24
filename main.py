import discord
from discord.ext import commands
import asyncio

import cv2
from imageai.Detection import ObjectDetection
import os


TOKEN = ""
MODEL_PATH = "yolo.h5"
INPUT_IMAGE = "Imagenes\imagen.jpg"  
OUTPUT_IMAGE = "Imagenes\imagen_detectada.jpg"


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

def detect_objects_on_farm(input_image, output_image, model_path):
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()
    detections = detector.detectObjectsFromImage(
        input_image=input_image,
        output_image_path=output_image,
        minimum_percentage_probability=30
    )
    return detections

def analyze_objects(detections):
    PRODUCT_CLASSES = ['apple', 'orange', 'banana', 'broccoli', 'carrot', 'pear', 'watermelon', 'tomato']
    return [d for d in detections if d['name'] in PRODUCT_CLASSES]

@bot.event
async def on_ready():
    print(f"✅ Bot conectado como {bot.user}")

@bot.command()
async def detectar(ctx):
    await ctx.send("🔍 Iniciando detección de frutas y vegetales...")

    detections = detect_objects_on_farm(INPUT_IMAGE, OUTPUT_IMAGE, MODEL_PATH)
    product_objects = analyze_objects(detections)

    if not product_objects:
        await ctx.send("🚫 No se detectaron frutas ni vegetales relevantes.")
        return

    resumen = "\n".join([f"- {obj['name']}: {obj['percentage_probability']:.1f}%" for obj in product_objects])
    await ctx.send(f"✅ Productos detectados:\n{resumen}")
    await ctx.send(file=discord.File(OUTPUT_IMAGE))
    await ctx.send("🧠 Análisis completado por AgroAI 🌿")

bot.run(TOKEN)

