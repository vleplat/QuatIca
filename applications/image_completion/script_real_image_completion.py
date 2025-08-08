import numpy as np
from PIL import Image
import quaternion
import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'core'))
from utils import quat_matmat
from solver import NewtonSchulzPseudoinverse
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time

def rgb_to_quaternion(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    h, w = r.shape
    quat_data = np.stack([np.zeros((h, w)), r, g, b], axis=-1)
    return quaternion.as_quat_array(quat_data)

def quaternion_to_rgb(Q):
    comp = quaternion.as_float_array(Q)
    r = comp[..., 1]
    g = comp[..., 2]
    b = comp[..., 3]
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

def gen_mask(shape, missing_rate):
    return (np.random.rand(*shape) > missing_rate).astype(np.float32)

def MRQ(A, max_iter=100):
    solver = NewtonSchulzPseudoinverse(gamma=1.0, max_iter=max_iter, tol=1e-6, verbose=False)
    A_pinv, _, _ = solver.compute(A)
    return A_pinv

def CURT(X, R1, R2, max_it=100):
    m, n = X.shape
    I = np.random.choice(n, R1, replace=False)
    J = np.random.choice(m, R2, replace=False)
    C = X[:, I]
    R = X[J, :]
    C_pinv = MRQ(C, max_it)
    R_pinv = MRQ(R, max_it)
    U = quat_matmat(quat_matmat(C_pinv, X), R_pinv)
    output = quat_matmat(quat_matmat(C, U), R)
    return output

def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

if __name__ == "__main__":
    # Load image and convert to quaternion
    #img_path = os.path.join('../.../../data/images', 'sample_image.jpg')
    img_path = os.path.join('../../data/images', 'kodim16.png')
    img = Image.open(img_path).convert('RGB')
    img = img.resize((256, 256))  # Optional: resize for speed
    img_np = np.array(img)
    B = rgb_to_quaternion(img_np)
    M, N = B.shape

    # Generate mask (missing 70%)
    mr = 0.70
    Q = gen_mask(B.shape, mr)
    B_Miss = B * Q

    X = B_Miss.copy()
    n_iter = 100
    rank_col = 60
    rank_row = 60
    max_it_pinv = 100

    # Track PSNR over iterations
    psnr_history = []
    
    # Start timing
    start_time = time.time()
    
    plt.ion()
    for i in range(n_iter):
        print(f"Iteration {i+1}/{n_iter}")
        Y = CURT(X, rank_col, rank_row, max_it_pinv)
        X = B_Miss + (1 - Q) * Y  # Fill in missing entries with CUR approx

        # Convert to RGB for visualization and denoising
        X_im = quaternion_to_rgb(X)
        
        # Calculate PSNR for this iteration
        current_psnr = psnr(X_im, img_np)
        psnr_history.append(current_psnr)
        
        plt.clf()
        plt.imshow(X_im)
        plt.title(f"Iteration {i+1} - PSNR: {current_psnr:.2f} dB")
        plt.pause(0.01)

        # Gaussian blur (denoising)
        X_im_blur = gaussian_filter(X_im, sigma=0.5)
        X = rgb_to_quaternion(X_im_blur)

    # End timing
    total_time = time.time() - start_time
    
    plt.ioff()
    
    # Convert masked image to RGB for visualization
    B_Miss_im = quaternion_to_rgb(B_Miss)
    
    # Get the final reconstruction BEFORE denoising for accurate PSNR
    # We need to reconstruct one more time without applying denoising
    Y_final = CURT(X, rank_col, rank_row, max_it_pinv)
    X_final = B_Miss + (1 - Q) * Y_final  # Fill in missing entries with CUR approx
    X_im_final = quaternion_to_rgb(X_final)
    
    # Calculate final PSNR on the non-denoised reconstruction
    psnr_val = psnr(X_im_final, img_np)
    print(f"Final PSNR (non-denoised): {psnr_val:.2f} dB")
    print(f"Total computation time: {total_time:.2f} seconds")
    
    # For visualization, we can still use the denoised version
    X_im = quaternion_to_rgb(X)
    
    # Create figure with subplots: images and PSNR evolution
    fig = plt.figure(figsize=(20, 8))
    
    # Top row: Original, Masked, Reconstructed images
    plt.subplot(2, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(B_Miss_im)
    plt.title('Masked Image (70% missing)')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(X_im_final)  # Show the non-denoised version for accurate PSNR
    plt.title(f'Reconstructed Image\nPSNR: {psnr_val:.2f} dB')
    plt.axis('off')
    
    # Bottom row: PSNR evolution
    plt.subplot(2, 3, 4)
    plt.plot(range(1, len(psnr_history) + 1), psnr_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR Evolution\nFinal: {psnr_val:.2f} dB')
    plt.grid(True, alpha=0.3)
    
    # Add computation time info
    plt.subplot(2, 3, 5)
    plt.text(0.1, 0.5, f'Computation Time: {total_time:.2f} s\n\nAlgorithm Parameters:\n• Rank: {rank_col}\n• Iterations: {n_iter}\n• Missing Rate: {mr*100:.0f}%\n• Max MP iterations: {max_it_pinv}', 
             fontsize=12, verticalalignment='center', transform=plt.gca().transAxes)
    plt.axis('off')
    
    # Add convergence info
    plt.subplot(2, 3, 6)
    if len(psnr_history) > 1:
        improvement = psnr_history[-1] - psnr_history[0]
        plt.text(0.1, 0.5, f'Convergence Info:\n\nInitial PSNR: {psnr_history[0]:.2f} dB\nFinal PSNR: {psnr_history[-1]:.2f} dB\nImprovement: {improvement:.2f} dB\n\nAvg time/iter: {total_time/n_iter:.3f} s', 
                 fontsize=12, verticalalignment='center', transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save high-resolution figure
    output_dir = "../../output_figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with parameters
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"quaternion_completion_{timestamp}_rank{rank_col}_iter{n_iter}_missing{mr*100:.0f}.png"
    save_path = os.path.join(output_dir, filename)
    
    # Save in high resolution (300 DPI)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"High-resolution figure saved to: {save_path}")
    
    plt.show()