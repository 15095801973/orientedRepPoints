import os.path as osp
import pickle
import random
import shutil
import tempfile

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.ops.minarearect import minaerarect
from mmcv.image import imread, imwrite
import cv2
from mmcv.visualization.color import Color, color_val
from mmdet.core import get_classes, tensor2imgs, rbbox2result
import matplotlib.pyplot as plt

def single_gpu_test(model, data_loader, show=False):
    # load_results = True
    load_results = False
    show = True
    # show = False
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    # for i, data in enumerate(data_loader):
    #     with torch.no_grad():
    #         result = model(return_loss=False, rescale=True, **data)
    #     results.append(result)
    # for i, data in enumerate(dataset):
    for i, data in enumerate(data_loader):
        if load_results:
            break
        with torch.no_grad():
            # abc_list = np.array(data['img'][0])
            # img = torch.from_numpy(abc_list)
            #
            # abc = np.array(data['img'])
            # img = torch.from_numpy(abc)
            # img_t = img.permute(0, 3, 1, 2)
            # r = model.module.simple_test(img_t.cuda(), [data])
            # result( [0:18]:reppoints 9个点  [18:26]bbox 4个顶点   最后一个是score(阈值0.05) )
            # torch.cuda.empty_cache() #第二张还重新分配?
            # result = model(return_loss=False, rescale=True, img = [img_t], img_metas= [[data]])
            # 为什么只有第一张不需要重映射?因为rescale(即show)变量发生了变化
            # result_temp = model(return_loss=False, rescale=not show, **data)
            result = model(return_loss=False, rescale=True, **data)
            # 结果是以设置中的scale为基准,没有还原到各个图片shape上
            # dataloader将ori_shape转换到norm_scale, 现在从norm_scale映射到ori_shape, 默认保持长宽比例
            # ori_shape = data['img_metas'][0].data[0][0]['ori_shape']
            # img_shape = data['img_metas'][0].data[0][0]['img_shape']
            # scale_factor = data['img_metas'][0].data[0][0]['scale_factor']
            # result = result_temp.copy()
            # for item in result:
            #     item[:, 0:26] = item[:, 0:26] / scale_factor
            # print('result', result)
        # results.append(result)



        if show:
            pts = annotations[i]['bboxes']
            pts = pts.reshape((-1, 4, 2)).astype(int)
            # cv2.fillConvexPoly(img, pts, (255, 0, 0))
            # 可以一次性画多个, 所以可以跳出循环
            path = data['img_metas'][0].data[0][0]['filename']
            img = cv2.imread(path)
            # img = imread(path)
            img = np.ascontiguousarray(img)
            cv2.polylines(img, pts, isClosed=True, color=(0, 0, 255), thickness=2)
            plt.title("oriented GT")
            # plt.imshow(img)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

        if show:
            # model.module.show_result(data, result)
            # img_tensor = img_t
            # img_metas = [data]
            # imgs = tensor2imgs(img_tensor)
            # assert len(imgs) == len(img_metas)

            if dataset is None:
                class_names = model.module.CLASSES
            elif isinstance(dataset, str):
                class_names = get_classes(dataset)
            elif isinstance(dataset, (list, tuple)):
                class_names = dataset
            else:
                class_names = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
                 'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor',
                 'swimming-pool', 'helicopter')
                # raise TypeError(
                #     'dataset must be a valid dataset name or a sequence'
                #     ' of class names, not {}'.format(type(dataset)))

            # for img, img_meta in zip(imgs, img_metas):
            #     pass
                # plt.title("tensor2img")
                # plt.imshow(img)
                # plt.show()
                # plt.title("data_img")
                # plt.imshow(data["img"][0])
                # plt.show()
                # h, w, _ = img_meta['img_shape'][0]
                # img_show = img[:h, :w, :]
                # show_one_v = np.zeros(img_meta["img_shape"][0])
                # o_img = data["img"][0]
                # show_points_v = np.zeros([h+100, w+100, 3])
                # for idx in range(len(result[5])):
                #     if idx >1:
                #         break
                #     one_large_v = result[5][idx]
                #     # print(f'score = {one_large_v[-1]}')
                #     x_points=[]
                #     y_points=[]
                #     # if idx == 0 :
                #     rd = random.randint(0,6)
                #     if rd == 0:
                #         rgb = [1., 1., 1.]
                #     elif rd == 1:
                #         rgb = [0., 0., 1.]
                #     elif rd == 2:
                #         rgb = [0., 1., 0.]
                #     elif rd == 3:
                #         rgb = [1., 0., 0.]
                #     elif rd == 4:
                #         rgb = [1., 1., 0.]
                #     elif rd == 5:
                #         rgb = [1., 0., 1.]
                #     elif rd == 6:
                #         rgb = [0., 1., 1.]
                #     # r = random.randint(0,2)
                #     # g = random.randint(0,2)
                #     # b = random.randint(0,2)
                #     # rgb = [r, g, b]
                #     for i in range(0, len(one_large_v) - 1, 2):
                #         x = one_large_v[i].astype(int)
                #         y = one_large_v[i+1].astype(int)
                #         x_points.append(x)
                #         y_points.append(y)
                #         if (x < w + 100) and (y < h + 100):
                #             show_points_v[y, x, 0:3] = rgb
                #     min_y = max(np.min(y_points), 0)
                #     min_x = max(np.min(x_points), 0)
                #     max_y = min(np.max(y_points), h-1)
                #     max_x = min(np.max(x_points), w-1)
                #     h, w, c = img_meta["img_shape"][0]
                #     for j in range(min_y, max_y):
                #         for i in range(min_x, max_x):
                #             show_one_v[j, i, 0] = 255.0
                #             o_img[j, i, 0] = o_img[j, i, 0] * 0.5 + 255.0 * 0.5
                # # show_one_v[x, y, 0] = 255.0
                #
                # plt.figure(dpi=300, figsize=(5, 6))
                # plt.title("show_points_v")
                # plt.imshow(show_points_v, interpolation='none')
                # plt.show()
                # plt.title("one large v")
                # plt.imshow(show_one_v)
                # plt.show()
                # plt.title("mix sample")
                # plt.imshow(o_img)
                # plt.show()

                bbox_result, segm_result = result, None
                bboxes_result = np.vstack(bbox_result)

                # draw bounding boxes
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)
                # mmcv.imshow_det_bboxes(
                #     img_show,
                #     bboxes[:, 18:27], #前面18个是9个rep点
                #     labels,
                #     class_names=class_names,
                #     score_thr=0.5)

                show_init = bboxes_result.shape[1] == 9+18+18
                if bboxes_result.shape[1] == 9 + 18:
                    bboxes = bboxes_result[:, 18:27]
                    scatters = bboxes_result[:, 0:18]
                    init_points = None
                elif show_init:
                    bboxes = bboxes_result[:, 36:45]
                    scatters = bboxes_result[:, 0:18]
                    init_points = bboxes_result[:, 18:36]
                # scatters[:,1::2] =

                class_names = None
                # path = data['filename'][0]
                path = data['img_metas'][0].data[0][0]['filename']

                # img = data["img"][0]
                img = cv2.imread(path)
                # img = imread(path)
                img = np.ascontiguousarray(img)
                # score_thr = 0.7
                # score_thr = 0.5
                score_thr = 0.3
                # score_thr = 0.2
                if score_thr > 0.01:
                    assert bboxes.shape[1] == 5 or bboxes.shape[1] == 9
                    scores = bboxes[:, -1]
                    inds = scores > score_thr
                    bboxes = bboxes[inds, :]
                    scatters = scatters[inds, :]
                    labels = labels[inds]
                    if show_init:
                        init_points = init_points[inds, :]
                bbox_color = text_color = 'green'
                bbox_color = color_val(bbox_color)
                text_color = color_val(text_color)

                for bbox, label in zip(bboxes, labels):
                    bbox_int = bbox.astype(np.int32)
                    left_top = (bbox_int[0], bbox_int[1])
                    right_bottom = (bbox_int[2], bbox_int[3])
                    thickness = 1
                    if len(bbox) == 5 or len(bbox) == 4:
                        cv2.rectangle(
                            img, left_top, right_bottom, bbox_color, thickness=thickness)
                        label_text = class_names[
                            label] if class_names is not None else f'cls {label}'
                    if len(bbox) == 5:
                        label_text += f'|{bbox[-1]:.03f}'
                    if len(bbox) == 8 or len(bbox) == 9:
                        # 绘制未填充的多边形
                        # cv2.polylines(img, [bbox], isClosed=True, color=(0, 0, 255), thickness=1)

                        # 绘制填充的多边形
                        # pts = bbox[0:8]
                        # pts = pts.reshape((-1, 4, 2)).astype(int)
                        #
                        # triangle = np.array([[0, 0], [1000, 800], [0, 800]]).reshape((-1, 1, 2))
                        # cv2.fillConvexPoly(img, pts, (255, 0, 0))
                        # 可以一次性化多个
                        # cv2.polylines(img, pts, isClosed=True, color=(0, 0, 255), thickness=5)

                        # plt.title("dead v")
                        # plt.imshow(img)
                        # plt.show()
                        # cv2.fillPoly(img, triangle, color=(255, 255, 255))
                        label_text = class_names[
                            label] if class_names is not None else f'cls {label}'

                    if len(bbox) == 9:
                        label_text += f'|{bbox[-1]:.03f}'

                    # pts = bbox[0:8]
                    # pts = pts.reshape((-1, 4, 2)).astype(int)

                    # triangle = np.array([[0, 0], [1000, 800], [0, 800]]).reshape((-1, 1, 2))
                    # cv2.fillConvexPoly(img, pts, (255, 0, 0))
                    # 可以一次性画多个, 所以可以跳出循环
                    # cv2.polylines(img, pts, isClosed=True, color=(0, 0, 255), thickness=2)

                    font_scale = 0.5
                    cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                                cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
                pts = bboxes[:, 0:8]
                pts = pts.reshape((-1, 4, 2)).astype(int)
                # 绘制矩形框, 可以一次性画多个, 所以可以跳出循环
                cv2.polylines(img, pts, isClosed=True, color=(0, 0, 255), thickness=2)

                # 可以用上面画多边形的函数画散点图

                # 一个点的多边形就是散点
                scatters = scatters.reshape((-1, 1, 2)).astype(int)
                cv2.polylines(img, scatters, isClosed=True, color=(0, 255, 255), thickness=3)
                if show_init:
                    # init即细化前的自适应点
                    scatters_init = init_points.reshape((-1, 1, 2)).astype(int)
                    cv2.polylines(img, scatters_init, isClosed=True, color=(255, 0, 0), thickness=3)
                    # 将细化前后的两点连线
                    line = np.concatenate((scatters_init, scatters), axis=1)
                    cv2.polylines(img, line, isClosed=True, color=(0, 255, 0), thickness=1)

                win_name = ''
                wait_time = 0
                out_file = f"F:\\360downloads\OrientedRepPoints-main\data\dota_1024\\trainval_split\\res_{i}.png" #None
                if show:
                    # cv2.imshow(win_name, imread(img))
                    plt.title("oriented v")
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    # plt.imshow(img)
                    plt.show()
                    cv2.imshow(win_name, img)
                    if wait_time == 0:  # prevent from hanging if windows was closed
                        while True:
                            ret = cv2.waitKey(1)

                            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
                            # if user closed window or if some key pressed
                            if closed or ret != -1:
                                break
                    else:
                        ret = cv2.waitKey(wait_time)
                if out_file is not None:
                    imwrite(img, out_file)
        if show:
            pass
            # img_metas = data
            # bbox_inputs = result + (img_metas, model.module.test_cfg, False)
            # bbox_list = model.module.bbox_head.get_bboxes(*bbox_inputs)
            # bbox_results = [
            #     rbbox2result(det_bboxes, det_labels, model.module.bbox_head.num_classes)
            #     for det_bboxes, det_labels in bbox_list
            # ]
            # wrap = {'img': [img_t], 'img_metas': [data]}
            # model.module.simple_test([img_t], [[data]])
            # model.module.show_result(data, r)

        new_result = []
        for item in result:
            bboxes = item [:, 36:45]
            scatters = item[:, 0:18]
            init_points = item[:, 18:36]
            new_result.append(np.concatenate([scatters, bboxes], axis=1))
            # new_result.append(np.concatenate([init_points, bboxes], axis=1))
        results.append(new_result)
        batch_size = len(data['img'])
        # batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    # np.save(f"F:\\360downloads\OrientedRepPoints-main\\results.npy", results)
    # results = np.load(f"F:\\360downloads\OrientedRepPoints-main\\results.npy", allow_pickle=True)
    if load_results:
        import _pickle
        f = open('F:\\360downloads\\OrientedRepPoints-main\\work_dirs\\orientedreppoints_r50_demo\\results.pkl', 'rb+')
        results = _pickle.load(f)
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
