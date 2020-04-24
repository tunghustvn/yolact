$:  command
[]: error
{}: fixed

#### 1: 

```
Computing validation mAP (this may take a while)...

Traceback (most recent call last):
  File "train.py", line 504, in <module>
    train()
  File "train.py", line 371, in train
    compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
  File "train.py", line 492, in compute_validation_map
    val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
  File "/home/hanguyen/Tung_yolact/yolact/eval.py", line 956, in evaluate
    prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)
  File "/home/hanguyen/Tung_yolact/yolact/eval.py", line 435, in prep_metrics
    mask_iou_cache = _mask_iou(masks, gt_masks)
  File "/home/hanguyen/Tung_yolact/yolact/eval.py", line 378, in _mask_iou
    ret = mask_iou(mask1, mask2, iscrowd)
  File "/home/hanguyen/Tung_yolact/yolact/layers/box_utils.py", line 108, in mask_iou
    masks_b = masks_b.view(masks_b.size(0), -1)
```

`RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1] because the unspecified dimension size -1 can be any value and is ambiguous`

[
id error: frankfurt_000001_062793_leftImg8bit.png
"munster_000051_000019_leftImg8bit.png"
]

{
Just skip all empty images. Changed code in **modified_eval.py** to fix this error.
}

#### 2:

```

Traceback (most recent call last):
  File "train.py", line 504, in <module>
    train()
  File "train.py", line 307, in train
    losses = net(datum)
  File "/home/hanguyen/Tung_deep_snake/snake_gpu6/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/hanguyen/Tung_deep_snake/snake_gpu6/lib/python3.7/site-packages/torch/nn/parallel/data_parallel.py", line 152, in forward
    outputs = self.parallel_apply(replicas, inputs, kwargs)
  File "/home/hanguyen/Tung_deep_snake/snake_gpu6/lib/python3.7/site-packages/torch/nn/parallel/data_parallel.py", line 162, in parallel_apply
    return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
  File "/home/hanguyen/Tung_deep_snake/snake_gpu6/lib/python3.7/site-packages/torch/nn/parallel/parallel_apply.py", line 85, in parallel_apply
    output.reraise()
  File "/home/hanguyen/Tung_deep_snake/snake_gpu6/lib/python3.7/site-packages/torch/_utils.py", line 394, in reraise
    raise self.exc_type(msg)
RuntimeError: Caught RuntimeError in replica 0 on device 0.
Original Traceback (most recent call last):
  File "/home/hanguyen/Tung_deep_snake/snake_gpu6/lib/python3.7/site-packages/torch/nn/parallel/parallel_apply.py", line 60, in _worker
    output = module(*input, **kwargs)
  File "/home/hanguyen/Tung_deep_snake/snake_gpu6/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "train.py", line 146, in forward
    losses = self.criterion(self.net, preds, targets, masks, num_crowds)
  File "/home/hanguyen/Tung_deep_snake/snake_gpu6/lib/python3.7/site-packages/torch/nn/modules/module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/hanguyen/Tung_yolact/yolact/layers/modules/multibox_loss.py", line 159, in forward
    ret = self.lincomb_mask_loss(pos, idx_t, loc_data, mask_data, priors, proto_data, masks, gt_box_t, score_data, inst_data, labels)
  File "/home/hanguyen/Tung_yolact/yolact/layers/modules/multibox_loss.py", line 546, in lincomb_mask_loss
    pos_idx_t = idx_t[idx, cur_pos]
RuntimeError: copy_if failed to synchronize: device-side assert triggered

```

#### 3:

When evaluate model with **trained model=yolact_plus_tung_84_30000.pth** and **config=yolact_plus_base_config** **dataset=cityscapes_traffic_no_pole**


```

Traceback (most recent call last):
  File "modified_eval.py", line 1113, in <module>
    evaluate(net, dataset)
  File "modified_eval.py", line 964, in evaluate
    prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)
  File "modified_eval.py", line 471, in prep_metrics
    ap_obj = ap_data[iou_type][iouIdx][_class]
IndexError: list index out of range

```

{https://github.com/dbolya/yolact/issues/31}

