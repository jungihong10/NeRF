import os

import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle

def get_grid(args):
    """
    Output:
    - grid (res, res, res): A boolean-type grid that represents the presence of an object.
    - grid_coord (res, res, res, 3): The xyz coordinates representing the center of each grid cell.
    """
    grid = np.ones((args.res, args.res, args.res), dtype=np.bool_)
    grid_coord = np.zeros((args.res, args.res, args.res, 3))
    grid_range = (np.arange(args.res) + 0.5) * 2.0 * args.range / args.res - args.range
    grid_coord[..., 0] = grid_range.reshape((-1, 1, 1))
    grid_coord[..., 1] = grid_range.reshape((1, -1, 1))
    grid_coord[..., 2] = grid_range.reshape((1, 1, -1))

    return grid, grid_coord


def load_data(frame, data_path, camera_angle_x):
    """
    Input:
        - frame (Dict):
            - file_path (str): The relative path of the RGBA image, excluding the extension. The extension is '.png'.
            - transform_matrix (List of List of float): A 4x4 list containing the camera-to-world transformation matrix.
        - data_path (str): The path to the dataset.
        - camera_angle_x (float): The horizontal field of view in the camera.

    Output:
        - img (h, w): A boolean-type numpy array representing the alpha mask. Pixels with nonzero alpha values should be marked as <True>.
        - K (3, 3): The camera intrinsic matrix. It can be assumed that f_x = f_y, and the camera center (c_x, c_y) aligns with the image center.
        - pose (4, 4): The world-to-camera transformation matrix. Note that for a blender-based dataset, the blender2opencv transformation matrix must be multiplied.
    """
    # Load the image and its alpha mask
    image_path = os.path.join(data_path, frame['file_path'] + '.png')
    img = np.array(Image.open(image_path).convert('RGBA'))

    img_alpha = img[..., 3] > 0  # Mark pixels with nonzero alpha values as True

    # Extract camera intrinsic parameters
    f = 0.5 * img.shape[1] / np.tan(0.5 * camera_angle_x)  # Focal length
    K = np.array([[f, 0, 0.5 * img.shape[1]], [0, f, 0.5 * img.shape[0]], [0, 0, 1]])

    # Extract camera-to-world transformation matrix
    transform_matrix = frame['transform_matrix']
    pose = np.array(transform_matrix)
    pose = np.linalg.inv(pose)

    # Apply blender2opencv transformation if needed
    blender2opencv = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    pose = np.dot(blender2opencv, pose)

    return img_alpha, K, pose

def project_to_image(xyz, K, pose):
    
    # Transform points from world to camera coordinates
    xyz = xyz.reshape(-1, 3)
    #print("xyz: ", xyz)
    xyz = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=-1)
    #print("xyz1: ", xyz)

    xyz_cam = (pose @ xyz.T).T[:, :3]
    #print("xyz_cam shape:", xyz_cam.shape)
    #print("k shape:", K.shape)
    
    # Project points onto the image plane
    uv = (K @ xyz_cam.T).T
    uv /= uv[:, 2:3]  # divide by z to get pixel coordinates
    #print("uv shape:", uv.shape)

    pixel_coord = uv[:, :2]
    #print("pixel_coord:", pixel_coord.reshape(-1, 2))

    return pixel_coord.reshape(-1, 2)


def invalid_check(img, pixel_coord):
    """
    Input:
        - img (h, w): Alpha mask.
        - pixel_coord (..., 2): XY coordinates in the pixel coordinate system.
    Output:
        - mask (...): Boolean type array indicating whether the pixel_coord corresponds to an empty pixel or not. mask.shape should be equal to pixel_coord.shape[:-1].
    """
    
    h, w = img.shape
    x, y = np.round(pixel_coord[..., 0]).astype(int), np.round(pixel_coord[..., 1]).astype(int)
    
    # Check if the coordinates are within the image dimensions
    valid_coords = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    # Initialize mask with False values
    mask = np.zeros_like(valid_coords, dtype=bool)

    # Update the mask only for invalid coordinates
    mask[valid_coords] = img[y[valid_coords], x[valid_coords]] == 0
    return mask

def get_rays(img_size, f, pose):
    """
    Input:
        - img_size (int): Size of the image for both height and width.
        - f (float): Focal length.
        - pose (4, 4): World-to-camera transformation matrix.
    Output:
        - rays_o (3,): XYZ values where all the camera rays originate.
        - rays_d (img_size, img_size, 3): XYZ directions for each camera ray towards each pixel.
    """
    # Inverse pose is used to transform rays from camera space to world space.
    inv_pose = np.linalg.inv(pose)
    
    # The camera's origin in world coordinates (camera-to-world)
    camera_origin = np.array([0, 0, 0, 1])
    rays_o = inv_pose @ camera_origin  # Matrix-vector multiplication
    rays_o = rays_o[:3]  # Discard the homogeneous coordinate

    # Create pixel grid
    i, j = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing='xy')
    
    # Compute direction for each ray in camera space
    x = (i - img_size / 2 ) / f
    y = (j - img_size / 2 ) / f
    z = np.ones((img_size, img_size))
    rays_d_cam = np.stack((x, y, z), axis=-1)
    
    # Rotate ray directions to world space
    rays_d_world = np.einsum('ij,klj->kli', inv_pose[:3, :3], rays_d_cam)
    
    return rays_o, rays_d_world


def render_depth(args, grid, rays_o, rays_d):
    """
    Input:
        - args: Additional arguments or configurations, if needed.
        - grid (res, res, res): A boolean grid indicating presence of an object.
        - rays_o (3,): The origin of the rays in world coordinate.
        - rays_d (img_size, img_size, 3): The direction of the rays in world coordinate.
        - near (float): The near plane distance.
        - far (float): The far plane distance.
        - num_steps (int): The number of steps to check along each ray.
    Output:
        - depth_map (img_size, img_size): Depth map for given rays_o and rays_d.
    """
    img_size = rays_d.shape[:2]
    depth_map = np.full(img_size, args.far)  # Initialize depth map with 'far' value.
    
    # Create a linspace for the depth values to check
    depth_values = np.linspace(args.near, args.far, args.num_step)
    
    for i in range(args.num_step):
        # Calculate the 3D points for this depth
        points = rays_o + rays_d * depth_values[i]
        
        # Transform these points to grid coordinates
        grid_coords = ((points + args.range) / (2 * args.range) * args.res).astype(int)
        
        # Ensure we don't go out of the grid bounds
        grid_coords = np.clip(grid_coords, 0, args.res - 1)
        
        # Check if these points intersect with an object in the grid
        intersects = grid[grid_coords[:, :, 0], grid_coords[:, :, 1], grid_coords[:, :, 2]]
        
        # Update the depth map wherever an intersection is found with a smaller depth value
        depth_map[intersects] = np.minimum(depth_map[intersects], depth_values[i])
        
    return depth_map


def IoU(mask1, mask2):
    I = (mask1 & mask2).sum()
    U = (mask1 | mask2).sum()
    return I / max(1, U)

def train(args):
    grid, grid_coord = get_grid(args)
    with open(os.path.join(args.datadir, args.scene, 'transforms_train.json'), 'r') as f:
        data = json.load(f)
    camera_angle_x = data['camera_angle_x']
    frames = data['frames']

    os.makedirs(args.traindir, exist_ok=True)
    train_tqdm = tqdm(frames, desc=args.scene + ' train')
    for frame in train_tqdm:
        img, K, pose = load_data(frame, os.path.join(args.datadir, args.scene), camera_angle_x)

        pixel_coord = project_to_image(grid_coord[grid], K, pose)
        grid[grid] = ~invalid_check(img, pixel_coord)
        volume = grid.sum() * ((2.0 * args.range) ** 3) / (args.res ** 3)
        train_tqdm.set_description(args.scene + ' train - volume: %.3f'%volume)

    with open(os.path.join(args.traindir, '%s.grid'%args.scene), 'wb') as f:
        pickle.dump(grid, f)

def test(args):
    with open(os.path.join(args.traindir, '%s.grid'%args.scene), 'rb') as f:
        grid = pickle.load(f)
    with open(os.path.join(args.datadir, args.scene, 'transforms_test.json'), 'r') as f:
        data = json.load(f)
    camera_angle_x = data['camera_angle_x']
    frames = data['frames']
    os.makedirs(os.path.join(args.testdir, args.scene), exist_ok=True)
    ious = []
    depth_maps = []
    test_tqdm = tqdm(frames[::2], desc=args.scene + ' test') # test 100 images not 200
    for frame in test_tqdm:
        img, K, pose = load_data(frame, os.path.join(args.datadir, args.scene), camera_angle_x)
        
        rays_o, rays_d = get_rays(img.shape[1], K[0, 0], pose)

        depth_map = render_depth(args, grid, rays_o, rays_d)

        mask = depth_map < args.far
        iou = IoU(img, mask)
        ious.append(iou)
        test_tqdm.set_description(args.scene + ' test - iou: %.3f - iou mean: %.3f'%(iou, sum(ious) / len(ious)))

        depth_map = 1.0 - (depth_map - args.near) / (args.far - args.near)
        depth_map = (depth_map * 255.0).astype(np.uint8)

        depth_map = Image.fromarray(depth_map)
        depth_map.save(os.path.join(args.testdir, args.scene, frame['file_path'].split('/')[-1] + '.png'))
        depth_maps.append(depth_map)
    
    depth_maps[0].save(os.path.join(args.testdir, '%s.gif'%args.scene), save_all=True, append_images=depth_maps[1:], loop=0)
    print(args.scene, 'test IoU mean:', sum(ious) / len(ious))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='nerf_synthetic')
    parser.add_argument('--traindir', type=str, default='train')
    parser.add_argument('--testdir', type=str, default='test')
    parser.add_argument('--scene', type=str, default='drums')

    # grid
    parser.add_argument('--res', type=int, default=512)
    parser.add_argument('--range', type=float, default=1.5)

    # other args
    parser.add_argument('--near', type=float, default=2.0)
    parser.add_argument('--far', type=float, default=6.0)
    parser.add_argument('--num_step', type=int, default=256)
    args = parser.parse_args()

    if args.scene == 'all':
        for args.scene in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']:
            train(args)
            test(args)
    else:
        train(args)
        test(args)