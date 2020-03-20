import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from torchvision.transforms import transforms


def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    image_list = sorted(glob.glob('{}/*'.format(args.images_dir)))
    patch_idx = 0

    for i, image_path in enumerate(image_list):
        hr = pil_image.open(image_path).convert('RGB')

        for hr in transforms.FiveCrop(size=(hr.height // 2, hr.width // 2))(hr):
            hr = hr.resize(((hr.width // args.scale) * args.scale, (hr.height // args.scale) * args.scale), resample=pil_image.BICUBIC)
            lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)

            hr = np.array(hr)
            lr = np.array(lr)

            lr_group.create_dataset(str(patch_idx), data=lr)
            hr_group.create_dataset(str(patch_idx), data=hr)

            patch_idx += 1
    print(len(h5_file['lr']))

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)

        hr = np.array(hr)
        lr = np.array(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

        print(i)

    print(len(h5_file['lr']))
    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='data/DIV2K_train_HR/DIV2K_train_HR')
    parser.add_argument('--output-path', type=str, default='train_DIV_neww.h5')
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--eval', type=bool, default=False)
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)