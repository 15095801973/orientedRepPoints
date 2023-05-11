windows平台,torch大于1.4,遇到过的一些问题
# 问题1: AT_CHECK
* setup报错: mmdet\ops\nms\src/nms_cuda.cpp(9): error C3861:“AT_CHECK”: 找不到标识符
* 方案: torch1.5及之后版本弃用了AT_CHECK, 将AT_CHECK全改成TORCH_CHECK就行了.

# 问题2: eps
* setup报错: mmdet/ops/nms/src/rnms_kernel.cu(18): error: identifier "eps" is undefined in device code
* 方案: 不能识别定义在函数外的eps, 因为不熟悉cu编程,不知道是什么原因. 但是把eps的定义放在函数里面就行了.

# 问题3: mmcv
* 使用requirements安装库报错, 错误发生在mmcv==0.6.2 (from -r requirements.txt (line 5))
* 因为mmcv只支持linux平台(存疑), windows下可以替换为安装mmcv-full==1.6.0(亲测)

# 问题4: _filter_imgs
* 训练时数据莫名其妙少了很多(1/3),是_filter_imgs出现了问题
* 原代码coco.py中
```python
 def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds
```
改为
```python
def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['id'] for _ in self.img_infos)
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds
```