import grpc
import cv2
import numpy as np

from proto.api.common.ar_session_pb2 import ArCameraFrame
from proto.api.services.ar_session.ar_session_service_pb2 import (
    EndToEndReconstructionRequest,
)

from proto.api.services.ar_session.ar_session_service_pb2_grpc import (
    EndToEndServiceStub,
)
from proto.api.vision.image_pb2 import Image

from common import dust3r_logger


def cv_image_to_proto(image: cv2.Mat, format_extension: str = "") -> Image:
    image_proto = Image()
    if len(format_extension) != 0:
        success, encoded_img = cv2.imencode(format_extension, image)
        if not success:
            dust3r_logger.error("Failed to encode image")
            return image_proto

        image_proto.encoded_str = encoded_img.tobytes()

        return image_proto

    height, width, channels = image.shape
    if channels == 1:
        image_proto.gray_image.row = height
        image_proto.gray_image.column = width
        image_proto.gray_image.data = image.data.tobytes()
        return image_proto

    if channels == 3:
        image_proto.rgb_image.height = height
        image_proto.rgb_image.width = width
        image_proto.rgb_image.data = image.data.tobytes()
        return image_proto


if __name__ == "__main__":
    channel = grpc.insecure_channel(
        "localhost:10001",
        options=[
            ("grpc.so_reuseport", 1),
            ("grpc.use_local_subchannel_pool", 1),
        ],
    )
    stub = EndToEndServiceStub(channel)

    img = cv2.imread("/home/arthurycliu/Documents/Alpha/Data/image_0.png")
    img2 = cv2.imread("/home/arthurycliu/Documents/Alpha/Data/image_355.png")

    request = EndToEndReconstructionRequest()
    request.frames.append(
        ArCameraFrame(
            image=cv_image_to_proto(img, ".png"),
        )
    )
    request.frames.append(
        ArCameraFrame(
            image=cv_image_to_proto(img2),
        )
    )
    response = stub.EndToEndReconstruction(request)
    print(len(response.image_depths))

    for depth_map_proto in response.image_depths:
        assert depth_map_proto.WhichOneof("image_type") == "gray_image"

        depth_map = np.frombuffer(
            depth_map_proto.gray_image.data, dtype=np.float32
        ).reshape((depth_map_proto.gray_image.row, depth_map_proto.gray_image.column))

        max_depth = np.max(depth_map)
        min_depth = np.min(depth_map)
        print(max_depth)
        print(min_depth)
        cv2.imshow("depth", (depth_map - min_depth) / (max_depth - min_depth))
        cv2.waitKey()
