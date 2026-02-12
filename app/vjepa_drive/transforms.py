# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from app.vjepa.transforms import make_transforms as _make_transforms


def make_transforms(
    random_resize_aspect_ratio=(3 / 4, 4 / 3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
):
    """Make driving video transforms.

    Same as V-JEPA2 transforms but with random_horizontal_flip=False,
    since flipping driving videos would invert the trajectory semantics
    (left turns become right turns, etc.).
    """
    return _make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
        normalize=normalize,
    )
