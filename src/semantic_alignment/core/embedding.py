import torch
from sentence_transformers import SentenceTransformer


def format_words_for_model(words, model_name):
    if "e5" in model_name.lower():
        return [f"query: {w}" for w in words]
    return words


def embed_words(words, model_name, device):
    model = SentenceTransformer(model_name, device=str(device))
    formatted_words = format_words_for_model(words, model_name)
    emb = model.encode(
        formatted_words,
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return torch.tensor(emb, dtype=torch.float32, device=device)


def precompute_embeddings(words, model_names, device):
    embeddings = {}
    for model_name in model_names:
        print("\n" + "=" * 100)
        print(f"Embedding model: {model_name}")
        print("=" * 100)
        embeddings[model_name] = embed_words(words, model_name, device)
        print("E:", tuple(embeddings[model_name].shape), embeddings[model_name].device)
    return embeddings
