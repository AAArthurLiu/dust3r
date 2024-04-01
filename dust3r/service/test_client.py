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
        # "192.168.10.150:7999",
        "0.0.0.0:10001",
        options=[
            ("grpc.so_reuseport", 1),
            ("grpc.use_local_subchannel_pool", 1),
        ],
    )
    stub = EndToEndServiceStub(channel)

    imgs = []
    imgs.append(cv2.imread("/home/arthurycliu/Documents/Alpha/Data/image_0.png"))
    imgs.append(cv2.imread("/home/arthurycliu/Documents/Alpha/Data/image_355.png"))

    request = EndToEndReconstructionRequest()
    request.frames.append(
        ArCameraFrame(
            image=cv_image_to_proto(imgs[0], ".png"),
        )
    )
    request.frames.append(
        ArCameraFrame(
            image=cv_image_to_proto(imgs[1]),
        )
    )
    response = stub.EndToEndReconstruction(request)

    pcl_list = []
    col_list = []
    print(len(response.point_maps))
    for i, point_map in enumerate(response.point_maps):
        points = np.frombuffer(point_map.data, dtype=np.float32).reshape(
            (point_map.row, point_map.column, 3)
        )

        image_resize = cv2.resize(imgs[i], (points.shape[1], points.shape[0]))
        image_rgb = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)

        # Convert the image to float64
        image_float64 = image_rgb.astype(np.float64) / 255.0

        assert point_map.row == image_float64.shape[0], f"{image_float64.shape}"
        assert point_map.column == image_float64.shape[1], f"{image_float64.shape}"
        for pts_row, col_row in zip(points, image_float64):
            for pts, cols in zip(pts_row, col_row):
                if np.linalg.norm(pts) == 0:
                    continue
                pcl_list.append(pts.reshape(1, 3))
                col_list.append(cols.reshape(1, 3))

    pcl = np.concatenate(pcl_list)
    col = np.concatenate(col_list)

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    pcd.colors = o3d.utility.Vector3dVector(col)

    o3d.visualization.draw_geometries([pcd])
