from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

if __name__ == "__main__":
    model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    device = "cuda"
    batch_size = 1
    schedule = "cosine"
    lr = 0.01
    niter = 300

    model = load_model(model_path, device)
    # load_images can take a list of images or a directory
    # images = load_images(
    #     ["croco/assets/Chateau1.png", "croco/assets/Chateau2.png"], size=512
    # )
    images = load_images(
        [
            "/home/arthurycliu/Documents/Alpha/Data/image_0.png",
            "/home/arthurycliu/Documents/Alpha/Data/image_355.png",
        ],
        size=512,
    )
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output["view1"], output["pred1"]
    view2, pred2 = output["view2"], output["pred2"]
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer
    )
    loss = scene.compute_global_alignment(
        init="mst", niter=niter, schedule=schedule, lr=lr
    )

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()

    import numpy as np

    def to_pcl(pts3d, color, mask):
        def to_numpy(x):
            return x.detach().cpu().numpy()

        def clamp_0_1(colors):
            if not isinstance(colors, np.ndarray):
                colors = colors.astype(float) / 255
            if np.issubdtype(colors.dtype, np.floating):
                pass
            assert (
                0 <= colors.min() and colors.max() <= 1
            ), f"{colors.min()} {colors.max()}"
            return colors

        pts3d = to_numpy(pts3d)
        mask = to_numpy(mask)
        if mask is None:
            mask = [slice(None)] * len(pts3d)

        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(color, mask)])
        # for i in range(10):
        #     print(mask[i * 30, i * 30])
        # print(f"color.shape {color.shape}")
        # print(f"pts.shape {pts.shape}")
        # print(f"col.shape {col.shape}")
        return pts.reshape(-1, 3), clamp_0_1(col.reshape(-1, 3))

    confidence_masks = scene.get_masks()

    pcl_list = []
    col_list = []
    for i in range(len(pts3d)):
        pcl, col = to_pcl(pts3d[i], imgs[i], confidence_masks[i])
        pcl_list.append(pcl)
        col_list.append(col)
    pcl = np.concatenate(pcl_list)
    col = np.concatenate(col_list)

    print(pcl.shape)
    print(col.shape)
    # print(col[::1000, :])

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    pcd.colors = o3d.utility.Vector3dVector(col)

    o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud("/home/arthurycliu/Documents/dust3r/data_desktop.ply", pcd)
    # print("dump done")

    # confidence_masks = scene.get_masks()

    # # visualize reconstruction
    # scene.show()

    # # find 2D-2D matches between the two images
    # from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

    # pts2d_list, pts3d_list = [], []
    # for i in range(2):
    #     conf_i = confidence_masks[i].cpu().numpy()
    #     pts2d_list.append(
    #         xy_grid(*imgs[i].shape[:2][::-1])[conf_i]
    #     )  # imgs[i].shape[:2] = (H, W)
    #     pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    # reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    # print(f"found {num_matches} matches")
    # matches_im1 = pts2d_list[1][reciprocal_in_P2]
    # matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # # visualize a few matches
    # import numpy as np
    # from matplotlib import pyplot as pl

    # pcl1, pcl2 = pts3d_list
    # print(type(pcl1))
    # print(type(pcl2))
    # # Dump pcl1 matrix into a txt file
    # np.savetxt("/home/arthurycliu/Documents/dust3r/pcl1.txt", pcl1)
    # np.savetxt("/home/arthurycliu/Documents/dust3r/pcl2.txt", pcl2)

    # n_viz = 10
    # match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    # viz_matches_im0, viz_matches_im1 = (
    #     matches_im0[match_idx_to_viz],
    #     matches_im1[match_idx_to_viz],
    # )

    # H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    # img0 = np.pad(
    #     imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), "constant", constant_values=0
    # )
    # img1 = np.pad(
    #     imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), "constant", constant_values=0
    # )
    # img = np.concatenate((img0, img1), axis=1)
    # pl.figure()
    # pl.imshow(img)
    # cmap = pl.get_cmap("jet")
    # for i in range(n_viz):
    #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
    #     pl.plot(
    #         [x0, x1 + W0],
    #         [y0, y1],
    #         "-+",
    #         color=cmap(i / (n_viz - 1)),
    #         scalex=False,
    #         scaley=False,
    #     )
    # pl.show(block=True)
