import cv2
import grpc
import numpy as np
import signal
from scipy.spatial import transform
from concurrent import futures
from common import dust3r_logger

from dust3r.inference import inference, load_model
from dust3r.utils.image import convert_cv_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.service.utils import to_masked_point_map

from proto.api.services.ar_session.ar_session_service_pb2 import (
    EndToEndReconstructionRequest,
    EndToEndReconstructionResponse,
)
from proto.api.services.ar_session.ar_session_service_pb2_grpc import (
    EndToEndServiceServicer,
    add_EndToEndServiceServicer_to_server,
)
from proto.api.math.pose_pb2 import Pose


def GetCvImage(image_proto):
    image_type_case = image_proto.WhichOneof("image_type")

    if image_type_case == "encoded_str":
        image_data = np.frombuffer(image_proto.encoded_str, dtype=np.uint8)
        return cv2.imdecode(image_data, cv2.IMREAD_ANYCOLOR)

    if image_type_case == "gray_image":
        return np.frombuffer(image_proto.gray_image.data, dtype=np.uint8).reshape(
            (image_proto.gray_image.row, image_proto.gray_image.column)
        )

    if image_type_case == "rgb_image":
        img_array = np.frombuffer(image_proto.rgb_image.data, dtype=np.uint8).reshape(
            (image_proto.rgb_image.height, image_proto.rgb_image.width, 3)
        )
        return img_array

    if image_type_case == "jpeg_image":
        image_data = np.frombuffer(image_proto.jpeg_image.data, dtype=np.uint8)
        return cv2.imdecode(image_data, cv2.IMREAD_ANYCOLOR)

    else:
        return cv2.Mat()


class EndToEndService(EndToEndServiceServicer):
    def __init__(self):
        self.model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        self.device = "cuda"
        self.batch_size = 1
        self.schedule = "cosine"
        self.lr = 0.01
        self.niter = 300
        self.model = load_model(self.model_path, self.device)

    def EndToEndReconstruction(
        self, request: EndToEndReconstructionRequest, context
    ) -> EndToEndReconstructionResponse:
        print("EndToEndReconstruction called")
        assert len(request.frames) == 2, "Only two frames are supported"

        images = convert_cv_images(
            [GetCvImage(frame.image) for frame in request.frames],
            size=512,
        )

        pairs = make_pairs(
            images, scene_graph="complete", prefilter=None, symmetrize=True
        )
        output = inference(pairs, self.model, self.device, batch_size=self.batch_size)
        scene = global_aligner(
            output, device=self.device, mode=GlobalAlignerMode.PairViewer
        )
        # # Not necessary for just one pair
        # _ = scene.compute_global_alignment(
        #     init="mst", niter=self.niter, schedule=self.schedule, lr=self.lr
        # )

        # retrieve useful values from scene:
        pts3d = scene.get_pts3d()
        depths = scene.get_depthmaps(raw=True)
        confidence_masks = scene.get_masks()
        poses = scene.get_im_poses()

        response = EndToEndReconstructionResponse()
        for i in range(len(depths)):
            point_map = to_masked_point_map(pts3d[i], confidence_masks[i])

            assert point_map.dtype == np.float32, f"Actual type: {point_map.dtype}"

            point_map_proto = EndToEndReconstructionResponse.PointMap()
            point_map_proto.row, point_map_proto.column, _ = point_map.shape
            point_map_proto.data = point_map.tobytes()
            response.point_maps.append(point_map_proto)

            pose_proto = Pose()
            print(poses[i])
            pose_proto.translation.x = poses[i][0, 3].item()
            pose_proto.translation.y = poses[i][1, 3].item()
            pose_proto.translation.z = poses[i][2, 3].item()
            quaternion = transform.Rotation.from_matrix(
                poses[i][:3, :3].cpu().numpy().reshape(3, 3)
            ).as_quat()
            pose_proto.rotation.x = quaternion[0]
            pose_proto.rotation.y = quaternion[1]
            pose_proto.rotation.z = quaternion[2]
            pose_proto.rotation.w = quaternion[3]
            response.poses.append(pose_proto)

        return response


if __name__ == "__main__":
    service_addresses = ["localhost:10001"]
    channel_opt = [
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=channel_opt)
    servicer = EndToEndService()
    add_EndToEndServiceServicer_to_server(
        servicer,
        server,
    )
    for service_address in service_addresses:
        server.add_insecure_port(service_address)
        dust3r_logger.info(
            f"Mapping schedule service set to listen to {service_address}"
        )
    server.start()
    dust3r_logger.info("Mapping schedule service started.")

    def sigterm_handler(signum, frame):
        dust3r_logger.info("Termination signal received. Gracefully terminating...")
        # Termination code here
        server.stop(None)

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)
    server.wait_for_termination()
