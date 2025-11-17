import cv2
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt
from superpoint import SuperPointNet


# Initialize SuperPoint model

# Use GPU if availble
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SuperPointNet().to(device)
state_dict = torch.load("superpoint_v1.pth", map_location=device)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
model.eval()
print(f"[INFO] Loaded SuperPointNet. missing={missing}, unexpected={unexpected}")

def extract_superpoint_features(img_gray, model, thresh=0.3):
    """Run SuperPoint on grayscale image and return keypoints + descriptors."""
    with torch.no_grad():
        inp = torch.from_numpy(img_gray / 255.).float()[None, None].to(device)
        output = model(inp)

        if isinstance(output, (list, tuple)):
            semi, desc = output[0], output[1]
        else:
            semi, desc = output["semi"], output["desc"]

        heatmap = torch.nn.functional.softmax(semi, dim=1)[:, :-1]
        heatmap = heatmap.squeeze().cpu().numpy().sum(axis=0)

        keypoints = np.argwhere(heatmap > thresh)
        scores = heatmap[keypoints[:, 0], keypoints[:, 1]]
        keypoints = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 1) for pt in keypoints]

        desc = desc.squeeze().cpu().numpy()
        desc = desc.reshape(desc.shape[0], -1).T
        return keypoints, desc

# Camera intrinsics
fx, fy, cx, cy = 320, 320, 160, 120
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
poses = [np.eye(4)]

# Load data
rgb_files = sorted(glob.glob("processed/*.png")) # Raw images
depth_files = sorted(glob.glob("depth_output/*.png")) # Depath maps
mask_files = sorted(glob.glob("segmentation_output/*.png")) # Segmented images
min_len = min(len(rgb_files), len(depth_files), len(mask_files))
print(f"[INFO] Using {min_len} synchronized frames.")
bf = cv2.BFMatcher(cv2.NORM_L2)

# SLAM loop
valid_frames = 0
for i in range(min_len - 1):
    print(f"\n[Frame {i}/{min_len-1}] ----------------------")
    img1 = cv2.imread(rgb_files[i], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(rgb_files[i + 1], cv2.IMREAD_GRAYSCALE)
    d1 = cv2.imread(depth_files[i], cv2.IMREAD_GRAYSCALE)
    mask1 = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None or d1 is None or mask1 is None:
        print("⚠️ Missing file, skipping.")
        continue

    d1 = cv2.resize(d1, (img1.shape[1], img1.shape[0]))
    mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY_INV)[1]
    mask1 = cv2.resize(mask1, (img1.shape[1], img1.shape[0]))

    img1_masked = cv2.bitwise_and(img1, mask1)

    # extract features
    kp1, des1 = extract_superpoint_features(img1_masked, model, thresh=0.3)
    kp2, des2 = extract_superpoint_features(img2, model, thresh=0.3)
    print(f"   Keypoints: {len(kp1)} → {len(kp2)}")

    if len(kp1) == 0 or len(kp2) == 0:
        print("⚠️ No valid keypoints.")
        continue

    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.trainIdx < len(kp2) and m.queryIdx < len(kp1):
            if m.distance < 0.9 * n.distance:
                good.append(m)
    print(f"   Matches found: {len(good)}")
    if len(good) < 8:
        print("⚠️ Too few matches.")
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # compute depth
    x_idx = np.clip(np.int32(pts1[:, 0]), 0, d1.shape[1] - 1)
    y_idx = np.clip(np.int32(pts1[:, 1]), 0, d1.shape[0] - 1)
    z = (255 - d1[y_idx, x_idx]) / 255.0 * 0.8  # inverted & scaled
    X = (pts1[:, 0] - cx) * z / fx
    Y = (pts1[:, 1] - cy) * z / fy
    obj_pts = np.vstack((X, Y, z)).T

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, pts2, K, None)
        if not success:
            print("⚠️ PnP failed.")
            continue
    except cv2.error:
        print("⚠️ PnP exception.")
        continue

    # Normalize translation
    tvec = tvec / (np.linalg.norm(tvec) + 1e-6) * 0.05  # 5cm step

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, tvec.squeeze()

    poses.append(poses[-1] @ np.linalg.inv(T))
    valid_frames += 1
    print(f"Pose updated. | tvec_norm={np.linalg.norm(tvec):.4f}")

# Trajectory processing
traj = np.array([p[:3, 3] for p in poses])
traj -= traj[0]  # start from (0,0,0)

# Normalize scale for display
scale = np.max(np.linalg.norm(traj - traj[0], axis=1))
if scale > 0:
    traj /= scale / 5.0  # scale to ~5m range

# Moving average smoothing
window = 5
traj_smooth_x = np.convolve(traj[:, 0], np.ones(window)/window, mode='valid')
traj_smooth_z = np.convolve(traj[:, 2], np.ones(window)/window, mode='valid')

# Plot estimated trajectory
plt.figure(figsize=(8, 6))
plt.plot(traj[:, 0], traj[:, 2], 'r--', alpha=0.4, label='Raw Trajectory')
plt.plot(traj_smooth_x, traj_smooth_z, 'b-', linewidth=2, label='Smoothed Trajectory')
plt.title("Estimated UAV Trajectory")
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

print(f"[SUMMARY] Valid frames: {valid_frames}/{min_len}")
