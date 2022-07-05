import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
from args import args


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.readers.File(file_root=data_dir, shard_id=0, num_shards=1, random_shuffle=True)
        self.decode = ops.decoders.Image(device="mixed")
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.random.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)

        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.readers.File(file_root=data_dir, shard_id=0, num_shards=1, random_shuffle=True)
        # Note that random_shuffle should be True for generate dynamic graphs
        # default: False for test
        self.decode = ops.decoders.Image(device="mixed")
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, crop, val_size=256):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                    data_dir=image_dir + '/train',
                                    crop=crop)
        pip_train.build()
        print(f'pip_train.epoch_size("Reader"):{pip_train.epoch_size("Reader")}')
        dali_iter_train = DALIClassificationIterator(
            pip_train,
            size=pip_train.epoch_size("Reader")
        )
        return dali_iter_train
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                data_dir=image_dir + '/val',
                                crop=crop, size=val_size)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(
            pip_val,
            size=pip_val.epoch_size("Reader")
        )

        return dali_iter_val


class ImageNetDali:
    def __init__(self):
        super(ImageNetDali, self).__init__()
        self.train_loader = get_imagenet_iter_dali(
            type='train',
            image_dir='/public/MountData/dataset/ImageNet50/', # You should construct the ImageNet50 dataset according to the content of the supplementary materials
            batch_size=args.batch_size,
            num_threads=16,
            crop=224,
            device_id=args.gpu
        )
        self.val_loader = get_imagenet_iter_dali(
            type='val',
            image_dir='/public/MountData/dataset/ImageNet50/',  # You should construct the ImageNet50 dataset according to the content of the supplementary materials
            batch_size=args.batch_size,
            num_threads=16,
            crop=224,
            device_id=args.gpu
        )
