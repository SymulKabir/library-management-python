from pymilvus import connections
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from datetime import datetime, timedelta

authentication_secession = {}




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

print("✅ Milvus connected successfully")





class ModifiedTrack(VideoStreamTrack):
    """
    Receives video → modifies → sends back
    """

    def __init__(self, track, pc):
        super().__init__()
        self.track = track
        self.pc = pc

    async def recv(self):
        frame = await self.track.recv()

        img = frame.to_ndarray(format="bgr24")

        # Run face detection and get frame with boxes
        result = await face_with_box(img)
        img_with_boxes = result["face_with_box"]
        img_crops = result["face_crops"]
        img = img_with_boxes
        face_id = None
        

        if len(img_crops) == 1 and self.pc.blink_count >= 5:
            img = img_crops[0]
            print("Hello form if block ---->>>")
            print("self.pc.blink_count ---->>>", self.pc.blink_count)

            straight = await is_straight_face(img)

            if straight:
                face_exist = await is_face_exist(img)
                print("face_exist -->", face_exist)

                if not self.pc.face_id:
                    if face_exist.get("face_id"):
                        face_id = face_exist.get("face_id")
                        self.pc.face_id = face_exist.get("face_id")
                    elif self.pc.failed_match_count >= 50:
                        result = await store_faces_in_weaviate(img)
                        if result:
                            face_id = result
                            self.pc.face_id = face_id
                    elif face_exist.get("status") == "face not matched":
                        self.pc.failed_match_count = self.pc.failed_match_count + 1



        if len(img_crops) == 1 and self.pc.blink_count < 5:
                blink = await detect_blink(img)
                if blink:
                    print("self.pc.blink_count ===========>>>>>", self.pc.blink_count)
                    self.pc.blink_count = self.pc.blink_count + 1

        # Send face_id via DataChannel
        if self.pc.face_auth_channel:
            payload = {}

            # payload["warning"] = False
            # if len(img_crops) == 1:
            #     payload["warning"] = False
            # else:
            #     payload["warning"] = True

            if datetime.now() - self.pc.initiate_time > timedelta(seconds=200000):
                payload["status"] = "timeout"
                payload["face_id"] = None
            elif face_id:
                payload["status"] = "recognized"
                payload["face_id"] = str(face_id)

            # print("self.pc.face_id -->>", self.pc.face_id)
            # print("payload -->>", payload)
            if len(payload) > 0:
                self.pc.face_auth_channel.send(json.dumps(payload))

        # convert back to VideoFrame
        new_frame = frame.from_ndarray(img_with_boxes, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame



@app.post("/video-stream")
async def webrtc_offer(data: dict):
    offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
    pc = RTCPeerConnection()

    pc.face_id = None
    pc.failed_match_count = 0
    pc.blink_count = 0
    pc.secession_id = data.get("secessionId")
    pc.initiate_time = datetime.now()

    if not authentication_secession.get(pc.secession_id):
        authentication_secession[pc.secession_id] = pc
    else:
        print("Existing secession found for ID:", pc.secession_id)

    print("pc.secession_id --->>>", pc.secession_id)

    print("authentication_secession --->>>", authentication_secession)

    @pc.on("datachannel")
    def on_datachannel(channel):
        print("channel.label -------------->>>>", channel.label)
        pc.face_auth_channel = channel
        if channel.label == "face-auth-channel":
            pc.face_auth_channel = channel
        # elif channel.label == "other-channel":
        #     print("Other data channel connected")

    @pc.on("track")
    def on_track(track):
        print("Track received:", track.kind)
        pc.addTrack(ModifiedTrack(track, pc))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}