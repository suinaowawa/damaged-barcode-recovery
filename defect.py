import torchvision.transforms.functional as FL
from PIL import Image, ImageDraw
import numpy as np



class DefectAdder(object):
    def __init__(self, mode='geometry', size_range=(0.05, 0.5), defect_shape=('line',), normal_only=False):
        self.mode = mode
        self.size_range = size_range
        self.defect_shape = defect_shape
        self.normal_only = normal_only

    def __call__(self, input):
        # assert isinstance(input, Image)
        # assert len(input.shape) == 3
        input_ = input.copy()
        if self.normal_only:
            return [input, input, input]
        if self.mode == 'geometry':
            output, target = self.add_defect(input_)
        else:
            raise NotImplementedError
        return [input, output, target]

    def add_defect(self, input):
        w, h = input.size
        draw = ImageDraw.Draw(input)
        shape = np.random.choice(self.defect_shape)
        size_ratio = np.random.uniform(self.size_range[0], self.size_range[1])
        x = int(np.random.random() * w)
        y = int(np.random.random() * h)
        size = int(size_ratio * min(w, h))
        color = tuple(np.random.randint(0, 255, 3))
        if shape == 'circle':
            draw.ellipse([x, y, x + size, y + size], fill=color)
            target = self.generate_target(input.size, shape, [x, y, x + size, y + size])
        elif shape == 'square':
            draw.rectangle([x, y, x + size, y + size], fill=color)
            target = self.generate_target(input.size, shape, [x, y, x + size, y + size])
        elif shape == 'line':
            while True:
                x1 = int(np.random.random() * w)
                y1 = int(np.random.random() * h)
                if x1 == x or y1 == y:
                    continue
                draw.line([x, y, x1, y1], fill=color, width=size)
                break
            target = self.generate_target(input.size, shape, [x, y, x1, y1], size)
        return input, target

    @staticmethod
    def generate_target(input_size, mode, xy, size=None):
        target = np.zeros(input_size)
        target = Image.fromarray(target)
        draw = ImageDraw.Draw(target)
        if mode == 'circle':
            draw.ellipse(xy, fill=1)
        elif mode == 'square':
            draw.rectangle(xy, fill=1)
        elif mode == 'line':
            draw.line(xy, fill=1, width=size)
        return target

    def __repr__(self):
        return self.__class__.__name__ + 'mode={}'.format(self.mode)


class NormalizeList(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensors):
        """
        Args:
            tensors (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensors: Normalized Tensor image.
        """
        for i in range(len(tensors) - 1):
            tensors[i] = FL.normalize(tensors[i], self.mean, self.std, self.inplace)
        return tensors

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensorList(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pics):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        for i in range(len(pics)):
            pics[i] = FL.to_tensor(pics[i])
        return pics

    def __repr__(self):
        return self.__class__.__name__ + '()'
