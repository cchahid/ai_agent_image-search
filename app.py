# =======================================================================
# ÉTAPE 1: (Handled by running 'pip install' in the PyCharm Terminal)
# =======================================================================

# =======================================================================
# ÉTAPE 2: IMPORTATIONS
# =======================================================================
print("\n🧠 Chargement du modèle CLIP...")
import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import chromadb
import gradio as gr
import requests
from PIL import Image
from io import BytesIO
import os

# --- Google Drive connection is REMOVED for local execution ---

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation du périphérique : {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# =======================================================================
# ÉTAPE 3: DÉFINITION DES CHEMINS PERSISTANTS LOCAUX
# =======================================================================
print("\n🗄️ Configuration des chemins de sauvegarde locaux...")

# MODIFIED: Use the current directory for the project folder
project_folder = "."
db_path = os.path.join(project_folder, "vector_database")
dataset_cache_path = os.path.join(project_folder, "dataset_cache")

# Create the directories if they don't exist
os.makedirs(db_path, exist_ok=True)
os.makedirs(dataset_cache_path, exist_ok=True)

# Initialize ChromaDB client to use the persistent local path
client = chromadb.PersistentClient(path=db_path)
collection_name = "unsplash_full_persistent"
collection = client.get_or_create_collection(name=collection_name)

# =======================================================================
# ÉTAPE 4: CHARGEMENT DU DATASET ET INDEXATION
# =======================================================================

# Helper function to download an image from a URL
def get_image_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None

# We only run the long indexing process if the database is empty
if collection.count() == 0:
    print("\n📥 Téléchargement du dataset COMPLET. Il sera sauvegardé localement...")
    # --- MISE À JOUR : On utilise 'cache_dir' pour sauvegarder le dataset localement ---
    dataset = load_dataset("1aurent/unsplash-lite", split="train", cache_dir=dataset_cache_path)

    print(f"\n⚙️  Indexation de {len(dataset)} images. Ce processus long n'aura lieu qu'une seule fois.")

    batch_size = 100
    images_to_embed = []
    metadatas_to_add = []
    ids_to_add = []

    for i, item in enumerate(dataset):
        image_display_url = item['photo']['image_url']
        image_download_url = image_display_url + "?w=640"
        image = get_image_from_url(image_download_url)

        if image:
            images_to_embed.append(image)
            ids_to_add.append(item['photo']['id'])
            metadatas_to_add.append({
                'url': image_display_url,
                'description': item['photo']['description']
            })

            if len(images_to_embed) >= batch_size or (i == len(dataset) - 1 and len(images_to_embed) > 0):
                if images_to_embed:
                    inputs = processor(images=images_to_embed, return_tensors="pt", padding=True, truncation=True).to(device)
                    with torch.no_grad():
                        image_embeddings = model.get_image_features(**inputs)
                    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

                    collection.upsert(
                        ids=ids_to_add,
                        embeddings=image_embeddings.cpu().numpy().tolist(),
                        metadatas=metadatas_to_add
                    )
                    print(f"   -> Lot {i // batch_size + 1} inséré. Total: {collection.count()}")
                    images_to_embed, metadatas_to_add, ids_to_add = [], [], []

    print("✅ Indexation terminée et sauvegardée localement !")
else:
    print(f"\n✅ Base de données chargée depuis le dossier local avec {collection.count()} images.")


# =======================================================================
# ÉTAPE 5: FONCTION DE RECHERCHE ET INTERFACE GRADIO
# =======================================================================
print("\n🚀 Lancement de l'interface web Gradio...")

def search(text_query, top_k=9):
    inputs = processor(text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    query_vector = text_embedding.cpu().numpy().tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=int(top_k)
    )

    image_urls = [item['url'] for item in results['metadatas'][0]]
    return image_urls

iface = gr.Interface(
    fn=search,
    inputs=[
        gr.Textbox(label="Décrivez l'image que vous cherchez...", placeholder="Ex: a cinematic shot of a forest..."),
        gr.Slider(minimum=1, maximum=12, step=1, value=9, label="Nombre de résultats")
    ],
    outputs=gr.Gallery(label="Résultats de la recherche", columns=3, object_fit="contain"),
    title="Recherche d'Images avec Unsplash 📸",
    description="Moteur de recherche sémantique utilisant une base de données persistante locale.",
    examples=[["a group of friends laughing"], ["a surfer on a big wave"], ["a cat sleeping on a laptop"] , ["a blue truck"], ["a frog in the wild"], ["a plane in the sky"] , ["a car on the road"] , ["a cat in the grass"] , ["a big ship on the ocean"] , ["a horse in a field"]]
)

# MODIFIED: Launch Gradio locally without the 'share' option
iface.launch()