from fastapi import FastAPI, UploadFile, File
import shutil
import os
import torch
import clip
from PIL import Image
from pymilvus import MilvusClient

app = FastAPI()

COLLECTION_NAME = "BookCovers"

# ---------------------------
# INIT MILVUS
# ---------------------------
client = MilvusClient("http://localhost:19530")

client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=512,
    auto_id=True,
    enable_dynamic_field=True
)

# ---------------------------
# LOAD MODEL (CLIP)
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        vec = model.encode_image(image)
        vec = vec / vec.norm(dim=-1, keepdim=True)

    return vec.cpu().numpy().flatten()


# ---------------------------
# API 1: ADD BOOK
# ---------------------------
@app.post("/add-book")
async def add_book(file: UploadFile = File(...), title: str = ""):
    file_path = f"uploads/{file.filename}"

    os.makedirs("uploads", exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    vector = get_embedding(file_path)
    print("vector --->>", vector)

    data = [
        {
            "vector": vector,
            "title": title,
            "image_path": file_path
        }
    ]
    print("data --->>", data)

    client.insert(
        collection_name=COLLECTION_NAME,
        data=data
    )

    return {"status": "book added"}


# ---------------------------
# API 2: SEARCH BOOK
# ---------------------------
@app.post("/search-book")
async def search_book(file: UploadFile = File(...)):
    print("file.filename -->>", file.filename)

    file_path = f"uploads/query_{file.filename}"
    print("file_path -->>", file_path)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    query_vector = get_embedding(file_path)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=3,
        output_fields=["title", "image_path"],
        search_params={"metric_type": "COSINE", "params": {}}
    )
    print("results -->>", results)

    output = []
    for hit in results[0]:
        output.append({
            "id": hit["id"],
            "distance": hit["distance"],
            "title": hit["entity"].get("title"),
            "image": hit["entity"].get("image_path")
        })
    print("output --->>", output)

    return {"results": output}