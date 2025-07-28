from flask import Flask, request, jsonify
import torch
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
import io
import os
from datetime import datetime
import logging

# Configuration
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Variables globales pour le modèle
processor = None
model = None

def load_model():
    """Charger le modèle BLIP pour la description d'images"""
    global processor, model
    
    print("🔄 Chargement du modèle BLIP...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Utiliser CPU par défaut (GPU si disponible)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Modèle chargé sur {device}")

# Charger le modèle au démarrage
load_model()

@app.route('/')
def home():
    return jsonify({
        "status": "active",
        "service": "WhatsApp Vision API",
        "endpoints": {
            "/analyze": "POST - Analyser une image",
            "/health": "GET - État du service"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Endpoint principal pour analyser les images"""
    try:
        data = request.get_json()
        
        # Récupérer l'image en base64
        if 'image' not in data:
            return jsonify({
                "success": False,
                "error": "Aucune image fournie"
            }), 400
        
        # Décoder l'image base64
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Paramètres optionnels
        language = data.get('language', 'fr')  # Français par défaut
        detail_level = data.get('detail_level', 'normal')  # normal ou detailed
        
        # Générer la description
        description = generate_description(image, language, detail_level)
        
        # Détecter les objets principaux
        objects = detect_objects(image)
        
        # Analyser les couleurs dominantes
        colors = analyze_colors(image)
        
        # Réponse complète
        response = {
            "success": True,
            "description": description,
            "details": {
                "objects": objects,
                "colors": colors,
                "image_size": f"{image.width}x{image.height}"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Message formaté pour WhatsApp
        whatsapp_message = format_whatsapp_response(description, objects, colors)
        response["whatsapp_message"] = whatsapp_message
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Erreur: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def generate_description(image, language='fr', detail_level='normal'):
    """Générer une description de l'image"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Préparer l'image
    inputs = processor(image, return_tensors="pt").to(device)
    
    # Générer la description en anglais d'abord
    if detail_level == 'detailed':
        # Description détaillée
        text = "a photography of"
        inputs_text = processor(image, text, return_tensors="pt").to(device)
        out = model.generate(**inputs_text, max_length=100, num_beams=5)
    else:
        # Description normale
        out = model.generate(**inputs, max_length=50, num_beams=3)
    
    description_en = processor.decode(out[0], skip_special_tokens=True)
    
    # Traduire si nécessaire (ici on fait une traduction simple)
    if language == 'fr':
        description = translate_to_french(description_en)
    else:
        description = description_en
    
    return description

def detect_objects(image):
    """Détecter les objets principaux dans l'image"""
    # Pour cet exemple, on utilise des catégories générales
    # En production, vous pourriez utiliser YOLOv5 ou un autre modèle
    
    # Analyse basique avec BLIP
    text = "What objects are in this image?"
    inputs = processor(image, text, return_tensors="pt")
    out = model.generate(**inputs, max_length=30)
    objects_text = processor.decode(out[0], skip_special_tokens=True)
    
    # Parser les objets (simplifié)
    common_objects = ['person', 'car', 'animal', 'food', 'building', 'nature', 'object']
    detected = []
    
    for obj in common_objects:
        if obj in objects_text.lower():
            detected.append(obj)
    
    return detected if detected else ['objet non identifié']

def analyze_colors(image):
    """Analyser les couleurs dominantes"""
    # Redimensionner pour l'analyse
    image_small = image.resize((150, 150))
    pixels = image_small.getdata()
    
    # Compter les couleurs (simplifié)
    color_counts = {}
    for pixel in pixels:
        # Catégoriser par couleur principale
        r, g, b = pixel
        if r > g and r > b:
            color = "rouge"
        elif g > r and g > b:
            color = "vert"
        elif b > r and b > g:
            color = "bleu"
        elif r > 200 and g > 200 and b > 200:
            color = "blanc"
        elif r < 50 and g < 50 and b < 50:
            color = "noir"
        else:
            color = "mixte"
        
        color_counts[color] = color_counts.get(color, 0) + 1
    
    # Top 3 couleurs
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    return [color[0] for color in sorted_colors[:3]]

def translate_to_french(text):
    """Traduction simple anglais → français"""
    # Dictionnaire de traduction basique
    translations = {
        "a photo of": "une photo de",
        "an image of": "une image de",
        "person": "personne",
        "people": "personnes",
        "man": "homme",
        "woman": "femme",
        "car": "voiture",
        "dog": "chien",
        "cat": "chat",
        "food": "nourriture",
        "building": "bâtiment",
        "street": "rue",
        "tree": "arbre",
        "sky": "ciel",
        "water": "eau",
        "sitting": "assis",
        "standing": "debout",
        "walking": "marchant",
        "on": "sur",
        "in": "dans",
        "with": "avec",
        "and": "et",
        "the": "le/la",
        "a": "un/une"
    }
    
    result = text.lower()
    for en, fr in translations.items():
        result = result.replace(en, fr)
    
    # Capitaliser la première lettre
    return result.capitalize()

def format_whatsapp_response(description, objects, colors):
    """Formater la réponse pour WhatsApp"""
    message = f"""📸 *Analyse de votre image*

📝 *Description:*
{description}

🔍 *Éléments détectés:*
{', '.join(objects)}

🎨 *Couleurs principales:*
{', '.join(colors)}

🤖 _Analyse générée par IA_
⏰ _{datetime.now().strftime('%H:%M')}_"""
    
    return message

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)