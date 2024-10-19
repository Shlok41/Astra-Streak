import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

def gaussian_streak(image_size, center, length, width, direction, angle_deg, intensity):
    """
    Generate a synthetic Gaussian streak.
    """
    x0, y0 = center
    sigma_x = length / 2.355
    sigma_y = width / 2.355

    x = np.arange(0, image_size[1])
    y = np.arange(0, image_size[0])
    x, y = np.meshgrid(x, y)

    if direction == 'horizontal':
        x_rot = x - x0
        y_rot = y - y0
    elif direction == 'vertical':
        x_rot = x - x0
        y_rot = y - y0
    elif direction == 'diagonal':
        angle_rad = np.deg2rad(angle_deg)
        x_rot = (x - x0) * np.cos(angle_rad) + (y - y0) * np.sin(angle_rad)
        y_rot = -(x - x0) * np.sin(angle_rad) + (y - y0) * np.cos(angle_rad)
    else:
        raise ValueError("Invalid direction. Must be 'horizontal', 'vertical', or 'diagonal'.")

    streak = intensity * np.exp(-((x_rot)**2 / (2 * sigma_x**2) + (y_rot)**2 / (2 * sigma_y**2)))

    return streak

# Parameters
image_size = (256, 256)  # Increased image size for multiple streaks
num_streaks = 10

# Directory to save the images
save_dir = r"C:\Users\avsk1\OneDrive\Desktop\coding\more code\Synthetic Streaks"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Generate multiple streaks
for i in range(num_streaks):
    # Random parameters for each streak
    center = (np.random.randint(0, image_size[0]), np.random.randint(0, image_size[1]))
    length = np.random.randint(30, 150)
    width = np.random.randint(5, 60)
    direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
    angle_deg = np.random.randint(0, 180) if direction == 'diagonal' else 0
    intensity = np.random.uniform(1.0, 3.0)

    # Generate streak
    streak = gaussian_streak(image_size, center, length, width, direction, angle_deg, intensity)

    # Add Gaussian noise
    noise_level = 0.1
    noise = np.random.normal(0, noise_level, image_size)
    noisy_streak = streak + noise

    # Save the synthetic FMA streaks in FITS format
    hdu = fits.PrimaryHDU(streak)
    hdu.writeto(os.path.join(save_dir, f'synthetic_fma_streak_{i+1:02d}.fits'), overwrite=True)

    hdu = fits.PrimaryHDU(noisy_streak)
    hdu.writeto(os.path.join(save_dir, f'noisy_synthetic_fma_streak_{i+1:02d}.fits'), overwrite=True)

    # Plot the results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(streak, cmap='gray')
    plt.title(f'Synthetic FMA Streak {i+1}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(noisy_streak, cmap='gray')
    plt.title(f'Noisy Synthetic FMA Streak {i+1}')
    plt.axis('off')

    plt.tight_layout()
    plt .show()