import cv2
import numpy as np
import insightface
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import weaviate
import weaviate.classes.config as wvc
from weaviate.classes.query import MetadataQuery

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Weaviate
client = weaviate.WeaviateClient(
    connection_params=weaviate.connect.ConnectionParams.from_params(
        http_host="localhost",
        http_port=8081,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
    )
)
client.connect()
if client.is_live():
    print("✅ Weaviate connected on port 8081!")

# Init collection
def init_collection():
    if not client.collections.exists("UserFaces"):
        client.collections.create(
            name="UserFaces",
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            properties=[
                wvc.Property(name="user_id", data_type=wvc.DataType.TEXT),
                wvc.Property(name="face_type", data_type=wvc.DataType.TEXT),
            ],
        )
        print("Collection 'UserFaces' initialized.")
init_collection()

# ✅ Use FaceAnalysis app from InsightFace (easier and robust)
app_face = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app_face.prepare(ctx_id=-1)  # CPU, ctx_id=0 for GPU

# Detect faces and get embedding
def detect_faces(frame):
    faces = app_face.get(frame)  # returns list of faces with bbox & embedding
    crops = []
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        crop = frame[y1:y2, x1:x2].copy()
        crops.append((crop, face.embedding))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame, crops

# Check if face exists
async def is_face_exist(embedding, threshold=0.5):
    try:
        coll = client.collections.get("UserFaces")
        res = coll.query.near_vector(
            near_vector=embedding.tolist(),
            limit=1,
            return_metadata=True
        )
        if res.objects:
            if res.objects[0].metadata.distance < threshold:
                return True
        return False
    except Exception as e:
        print("⚠️ Weaviate query error:", e)
        return False

# Store face
async def store_faces_in_weaviate(face_img, embedding, face_type="Straight"):
    if await is_face_exist(embedding):
        print("Face already exists, skipping store")
        return
    try:
        client.collections.add(
            collection_name="UserFaces",
            objects=[{"face_type": face_type}],
            vector=embedding.tolist()
        )
        print("✅ Stored new face in Weaviate")
    except Exception as e:
        print("⚠️ Error storing face:", e)

# Video track processing
class ModifiedTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        img_with_boxes, crops = detect_faces(img)

        for crop, embedding in crops:
            await store_faces_in_weaviate(crop, embedding)

        new_frame = frame.from_ndarray(img_with_boxes, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

@app.post("/webrtc-offer")
async def webrtc_offer(data: dict):
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    pc = RTCPeerConnection()

    @pc.on("track")
    def on_track(track):
        pc.addTrack(ModifiedTrack(track))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}