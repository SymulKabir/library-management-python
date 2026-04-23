import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole
import weaviate
import weaviate.classes.config as wvc
from weaviate.classes.query import MetadataQuery
from weaviate.collections.classes.filters import Filter


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use the manual connection for port 8081
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


# Initialize the Collection (Schema) if it doesn't exist
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


async def face_with_box(frame):

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    face_crops = []
    # Draw red boxes around detected faces
    for x, y, w, h in faces:

        # Crop face region
        face_img = frame[y : y + h, x : x + w].copy()
        face_crops.append(face_img)

        # Draw red box on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    return {"face_with_box": frame, "face_crops": face_crops}


# Convert face image (numpy array) to a normalized vector
# def face_to_vector(face_img):
#     vec = face_img.flatten().astype(float)
#     vec = vec / (np.linalg.norm(vec) + 1e-6)
#     return vec.tolist()

def face_to_vector(face_img):
    # Resize to a fixed size (e.g., 128x128)
    standard_size = cv2.resize(face_img, (128, 128))
    
    # Flatten: 128 * 128 * 3 = 49152 dimensions
    vec = standard_size.flatten().astype(float)
    
    # Normalize
    vec = vec / (np.linalg.norm(vec) + 1e-6)
    return vec.tolist()


# Check if a similar face already exists in Weaviate
async def is_face_exist(face_img, similarity_threshold=0.8):
    user_faces_coll = client.collections.get("UserFaces")
    vector = face_to_vector(face_img)

    try:
        response = user_faces_coll.query.near_vector(
            near_vector=vector,
            limit=1,
            # filters=user_filter,
            return_metadata=MetadataQuery(distance=True),
            # include_vector=True
        )
        # `results` contains a list of objects
        if response.objects and response.objects[0].metadata.distance < 0.15:
            # face_match = await match_faces(straight_vec, response.objects[0].vector)
            face_id = str(response.objects[0].uuid)
            print("face_id:->", face_id)
            # print("current vector:->", response.objects[0].vector)

    except Exception as e:
        print("⚠️ Weaviate query error:", e)
        return False 
    
    
# Store exactly one face in Weaviate
async def store_faces_in_weaviate(face_img, face_type="Straight"):
    # if len(face_img) != 1:
    #     print("⚠️ Skipping storing: not exactly one face detected")
    #     return
    try:

        vector = face_to_vector(face_img)

        client.collections.add(
            collection_name="UserFaces", objects=[{"face_type": face_type}], vector=vector
        )
        print(f"✅ Stored 1 face for user ")
    except Exception as e:
        print("⚠️ Error storing face in Weaviate:", e)  


class ModifiedTrack(VideoStreamTrack):
    """
    Receives video → modifies → sends back
    """

    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()

        img = frame.to_ndarray(format="bgr24")

        # Run face detection and get frame with boxes
        result = await face_with_box(img)
        img_with_boxes = result["face_with_box"]
        img_crops = result["face_crops"]
        img = img_with_boxes

        if len(img_crops) == 1:
            print(f"Detected {len(img_crops)} faces")
            img = img_crops[0]
            # face_exist = await is_face_exist(img)
            # print(f"Face exist in Weaviate: {face_exist}")
            await store_faces_in_weaviate(img)

        # 🔥 APPLY MODIFICATION HERE
        # Example: draw red box
        # cv2.rectangle(img, (50, 50), (300, 300), (0, 0, 255), 3)

        # convert back to VideoFrame
        new_frame = frame.from_ndarray(img, format="bgr24")
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
