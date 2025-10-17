# 📸 Moteur de Recherche d'Images Sémantique avec CLIP

Ce projet est un moteur de recherche d'images intelligent qui utilise le modèle **CLIP** d'OpenAI pour trouver des images à partir d'une description textuelle en langage naturel.

Il télécharge un sous-ensemble du dataset Unsplash, génère des représentations vectorielles (*embeddings*) pour chaque image et les stocke dans une base de données locale et persistante (**ChromaDB**). L'interface utilisateur est construite avec **Gradio**, permettant une interaction simple et intuitive.



## ✨ Fonctionnalités

-   **Recherche Sémantique :** Trouvez des images en décrivant ce que vous cherchez (ex: "un chat dormant sur un ordinateur portable").
-   **Modèle CLIP :** Utilise le puissant modèle `openai/clip-vit-base-patch32` pour comprendre à la fois le texte et les images.
-   **Base de Données Persistante :** Grâce à ChromaDB, l'indexation complète des images n'est effectuée **qu'une seule fois**. Les lancements ultérieurs sont quasi instantanés.
-   **Interface Web Simple :** Une interface utilisateur propre et fonctionnelle créée avec Gradio, accessible depuis votre navigateur.

---

## ⚙️ Installation et Lancement

### Prérequis
- Python 3.8 ou supérieur
- `pip` (gestionnaire de paquets Python)

### Étapes

1.  **Structure des Fichiers**
    Assurez-vous que votre fichier de script (par exemple, `app.py`) et le fichier `requirements.txt` sont dans le même dossier.

2.  **Créez un environnement virtuel (Fortement recommandé)**
    Ouvrez un terminal dans le dossier de votre projet et exécutez :
    ```bash
    # Créer l'environnement
    python -m venv venv

    # Activer l'environnement
    # Sur Windows:
    .\venv\Scripts\activate
    # Sur macOS/Linux:
    source venv/bin/activate
    ```

3.  **Installez les dépendances**
    Avec votre environnement virtuel activé, installez toutes les bibliothèques requises :
    ```bash
    pip install -r requirements.txt
    ```

4.  **Lancez l'application**
    Exécutez le script principal :
    ```bash
    python app.py
    ```

---

## 🚀 Utilisation

-   **⚠️ La première exécution sera longue.** Le script doit télécharger le dataset (~25 000 images) et les indexer. Ce processus peut prendre plusieurs minutes en fonction de votre connexion internet et de la puissance de votre ordinateur (CPU/GPU). Vous verrez la progression dans le terminal.

-   **Les lancements suivants seront très rapides.** Le script détectera la base de données locale et la chargera directement.

-   Une fois que le message `Running on local URL: http://...` s'affiche dans le terminal, ouvrez ce lien dans votre navigateur pour commencer à chercher des images !

### Structure du projet après exécution

Après le premier lancement, de nouveaux dossiers seront créés pour stocker les données de manière persistante :

```
.
├── app.py                  # Le script principal de l'application
├── requirements.txt        # Les dépendances Python
├── vector_database/        # Dossier créé par ChromaDB pour la base de données
└── dataset_cache/          # Dossier créé par 'datasets' pour stocker le dataset Unsplash
```