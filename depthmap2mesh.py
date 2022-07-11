#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def depthmap2mesh(
        depth, obj_file, img_path=None, mtl_file=None, mtl_name='colord',
        grad_mask=None, fov=(np.pi/2), min_val=0, max_val=np.inf):

    if isinstance(obj_file, str):
        obj_file = open(obj_file, 'w')

    if img_path is not None:
        if mtl_file is not None:
            mtl_file = open(mtl_file, 'w')

        obj_file.write(f'mtllib {mtl_file.name}\n')
        obj_file.write(f'usemtl {mtl_name}\n')

    h, w = depth.shape
    d = max(h, w) / 2 / np.tan(fov / 2)

    grid = np.mgrid[:h, :w].transpose(1, 2, 0)
    grid = grid - np.array((h, w)) / 2

    valid = (depth > min_val) & (depth < max_val)
    v_inds = np.arange(1, depth.size + 1).reshape(h, w)
    v_inds[~valid] = 0
    depth[~valid] = 0

    # v_grid, v_depth = grid[valid], depth[valid]
    v_grid, v_depth = grid.reshape(-1, 2), depth.reshape(-1)
    v_grid *= v_depth[:, None] / d
    v_grid = np.stack((v_grid[:, 0], -v_grid[:, 1], -v_depth), -1)

    obj_file.write('\n'.join(
        ' '.join(['v', str(x), str(y), str(z)]) for (y, x, z) in v_grid))
    obj_file.write('\n')

    # if img_path is not None:
    vt_grid = (
        np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
        / np.array((h, w)))
    obj_file.write('\n'.join(
        f'vt {str(x)} {str(y)}' for y, x in vt_grid))
    obj_file.write('\n')

    l_grid = np.stack(
        (v_inds[:-1, :-1], v_inds[1:, :-1], v_inds[:-1, 1:], v_inds[1:, 1:]))
    top_left, bottom_right = np.all(l_grid[:3], 0), np.all(l_grid[1:], 0)
    if grad_mask is not None:
        top_left, bottom_right = (
            top_left & grad_mask[:-1, :-1, 0] & grad_mask[:-1, :-1, 1],
            bottom_right & grad_mask[1:, 1:, 0] & grad_mask[1:, 1:, 1])
    invalid = ~(top_left | bottom_right)

    l_list = l_grid.reshape(4, -1)
    f_list = np.concatenate((
        l_list[:3, top_left.reshape(-1)],
        l_list[1:, bottom_right.reshape(-1)]), 1)

    if np.any(invalid):
        top_right, bottom_left = (
            invalid & np.all(l_grid[[0, 1, 3]]),
            invalid & np.all(l_grid[[0, 2, 3]]))
        if grad_mask is not None:
            top_right, bottom_left = (
                top_right & grad_mask[:-1, 1:, 0] & grad_mask[1:, :-1, 1],
                bottom_left & grad_mask[1:, :-1, 0] & grad_mask[:-1, 1:, 1])
        if np.any(top_right):
            f_list = np.concatenate((
                f_list, l_list[[0, 1, 3], top_right.reshape(-1)]), 1)
        if np.any(bottom_left):
            f_list = np.concatenate((
                f_list, l_list[[0, 2, 3], bottom_left.reshape(-1)]), 1)

    obj_file.write('\n'.join(
        ' '.join(['f'] + [f'{x}/{x}' for x in nums])
        for nums in f_list.transpose()))
    obj_file.write('\n')

    obj_file.close()
    if mtl_file is not None:
        mtl_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert depth map to mesh as .obj file')
    parser.add_argument('depth', help=(
        'Input depth map. '
        'Either in 16 bit png format in millimeter scale (kinect), '
        'or .npy file in meter scale.'))
    parser.add_argument('-i', '--image', help=(
        'Image corresponding to depth map to generate material, if not givin, '
        'no material file would be created'))
    parser.add_argument('-o', '--output', help=(
        'Output .obj file, defaults to [DEPTH_PATH].obj'))
    parser.add_argument('-m', '--material', help=(
        'Output .mtl file, defaults to [DEPTH_PATH].mtl'))
    args = parser.parse_args()

    depth_path = Path(args.depth)
    if depth_path.suffix == '.png':
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        depth = depth.astype(float)
    else:
        depth = np.load(str(depth_path)).astype(float) * 1000

    if args.output is None:
        args.output = str(depth_path.parent / (depth_path.stem + '.obj'))

    if args.material is None and args.image is not None:
        args.material = str(depth_path.parent / (depth_path.stem + '.mtl'))

    depthmap2mesh(
        depth, args.output, img_path=args.image, mtl_file=args.material)
