import cv2
import grpc
import numpy as np
import signal
from concurrent import futures
from common import dust3r_logger

from dust3r.inference import inference, load_model
from dust3r.utils.image import convert_cv_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from proto.api.services.ar_session.ar_session_service_pb2 import (
    EndToEndReconstructionRequest,
    EndToEndReconstructionResponse,
)
from proto.api.services.ar_session.ar_session_service_pb2_grpc import (
    EndToEndServiceServicer,
    add_EndToEndServiceServicer_to_server,
)
from proto.api.vision.image_pb2 import Image


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

        print("I am running. yeah")

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
        # Not necessary for just one pair
        # _ = scene.compute_global_alignment(
        #     init="mst", niter=self.niter, schedule=self.schedule, lr=self.lr
        # )

        # retrieve useful values from scene:
        imgs = scene.imgs
        depths = scene.get_depthmaps(raw=True)
        confidence_masks = scene.get_masks()

        def to_depth_map(depth, img, mask):
            def to_numpy(x):
                return x.detach().cpu().numpy()

            depth = to_numpy(depth).reshape(img.shape[0], img.shape[1])
            mask = to_numpy(mask)

            for p, m in zip(depth, mask):
                p[~m] = 0

            return depth

        response = EndToEndReconstructionResponse()
        for i in range(len(depths)):
            depth_map = to_depth_map(depths[i], imgs[i], confidence_masks[i])
            assert depth_map.dtype == np.float32, f"Actual type: {depth_map.dtype}"
            depth_proto = Image()
            depth_proto.gray_image.row = depth_map.shape[0]
            depth_proto.gray_image.column = depth_map.shape[1]
            depth_proto.gray_image.data = depth_map.tobytes()
            response.image_depths.append(depth_proto)

        return response

        # pts3d = scene.get_pts3d()
        # def to_pcl(pts3d, color, mask):
        #     def to_numpy(x):
        #         return x.detach().cpu().numpy()

        #     def clamp_0_1(colors):
        #         if not isinstance(colors, np.ndarray):
        #             colors = colors.astype(float) / 255
        #         if np.issubdtype(colors.dtype, np.floating):
        #             pass
        #         assert (
        #             0 <= colors.min() and colors.max() <= 1
        #         ), f"{colors.min()} {colors.max()}"
        #         return colors

        #     pts3d = to_numpy(pts3d)
        #     mask = to_numpy(mask)
        #     if mask is None:
        #         mask = [slice(None)] * len(pts3d)

        #     pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        #     col = np.concatenate([p[m] for p, m in zip(color, mask)])
        #     # for i in range(10):
        #     #     print(mask[i * 30, i * 30])
        #     # print(f"color.shape {color.shape}")
        #     # print(f"pts.shape {pts.shape}")
        #     # print(f"col.shape {col.shape}")
        #     return pts.reshape(-1, 3), clamp_0_1(col.reshape(-1, 3))

        # pcl_list = []
        # col_list = []
        # for i in range(len(pts3d)):
        #     pcl, col = to_pcl(pts3d[i], imgs[i], confidence_masks[i])
        #     pcl_list.append(pcl)
        #     col_list.append(col)
        # pcl = np.concatenate(pcl_list)
        # col = np.concatenate(col_list)

        # print(pcl.shape)
        # print(col.shape)
        # # print(col[::1000, :])

        # import open3d as o3d

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcl)
        # pcd.colors = o3d.utility.Vector3dVector(col)

        # o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    service_addresses = ["localhost:10001"]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
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
