import grpc

from proto.api.services.ar_session.ar_session_service_pb2 import (
    EndToEndReconstructionRequest,
)

from proto.api.services.ar_session.ar_session_service_pb2_grpc import (
    EndToEndServiceStub,
)


from common import dust3r_logger

if __name__ == "__main__":
    channel = grpc.insecure_channel(
        "localhost:10001",
        options=[
            ("grpc.so_reuseport", 1),
            ("grpc.use_local_subchannel_pool", 1),
        ],
    )
    stub = EndToEndServiceStub(channel)

    request = EndToEndReconstructionRequest()
    response = stub.EndToEndReconstruction(request)
    print(response)
