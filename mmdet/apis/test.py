import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmdet.core import  get_classes, tensor2imgs
import matplotlib.pyplot as plt

def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # for i, data in enumerate(data_loader):
    for i, data in enumerate(dataset):
        with torch.no_grad():
            abc = np.array(data['img'])
            img = torch.from_numpy(abc)
            img_t = img.permute(0,3,1,2)
            result = model(return_loss=False, rescale=not show, img = [img_t], img_metas= [[data]])
            print('result', result)
        results.append(result)

        show = True
        if show:
            # model.module.show_result(data, result)
            bbox_result, segm_result = result, None

            img_tensor = img_t
            img_metas = [data]
            imgs = tensor2imgs(img_tensor)
            assert len(imgs) == len(img_metas)

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

            for img, img_meta in zip(imgs, img_metas):
                plt.imshow(img)
                plt.title("tensor2img")
                plt.imshow(data["img"][0])
                plt.show()
                plt.title("data_img")
                plt.show()
                h, w, _ = img_meta['img_shape'][0]
                img_show = img[:h, :w, :]
                one_large_v = result[5][0]
                show_one_v = np.zeros(img_meta["img_shape"][0])
                min_y = min_x = 9999
                max_y = max_x = 0
                x_points=[]
                y_points=[]
                for i in range(0,len(one_large_v) -1, 2):
                    x = one_large_v[i].astype(int)
                    y = one_large_v[i+1].astype(int)
                    x_points.append(x)
                    y_points.append(y)
                min_y = np.min(y_points)
                min_x = np.min(x_points)
                max_y = np.max(y_points)
                max_x = np.max(x_points)
                h, w, c = img_meta["img_shape"][0]
                o_img = data["img"][0]
                for j in range(min_y, max_y + 1):
                    for i in range(min_x, max_x + 1):
                        show_one_v[j, i, 0] = 255.0
                        o_img[j, i, 0] = o_img[j, i, 0] * 0.5 + 255.0 * 0.5
                # show_one_v[x, y, 0] = 255.0
                print(f'score = {one_large_v[-1]}')
                plt.imshow(show_one_v)
                plt.title("one large v")
                plt.imshow(o_img)
                plt.show()
                plt.title("mix sample")
                plt.show()
                bboxes = np.vstack(bbox_result)

                # draw bounding boxes
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)
                mmcv.imshow_det_bboxes(
                    img_show,
                    bboxes,
                    labels,
                    class_names=class_names,
                    score_thr=0.5)

        batch_size = len(data['img'])
        # batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
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
