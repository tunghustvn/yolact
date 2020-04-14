#### 1: 
`RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1] because the unspecified dimension size -1 can be any value and is ambiguous`

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


