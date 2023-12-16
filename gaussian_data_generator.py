from scipy.stats import norm, multivariate_normal
import numpy as np


def generate_2d_gaussian_data_with_noise(mean, cov, num_samples, noise_std=None):
    """
    Generate 2D Gaussian data with optional Gaussian noise and calculate the probability density for each point.

    :param mean: List of means for the Gaussian distribution.
    :param cov: 2x2 covariance matrix.
    :param num_samples: Number of samples to generate.
    :param noise_std: Standard deviation of Gaussian noise to add to each point. If None, no noise is added.
    :return: Tuple of (X, y) where X is the input features and y is the target variable (probability density).
    """
    # Generate the 2D Gaussian data
    data_2d = np.random.multivariate_normal(mean, cov, num_samples)

    # Add noise if specified
    if noise_std is not None:
        noise = np.random.normal(0, noise_std, data_2d.shape)
        data_2d += noise

    # Calculate the probability density for each point
    rv = multivariate_normal(mean, cov)
    prob_density = rv.pdf(data_2d)

    X = data_2d  # The input features (X and Y coordinates)
    y = prob_density  # The target variable

    return X, y



    
def generate_1d_gaussian_data_with_noise(mean, std_dev, num_points, noise_std=None):
    """
    Generate 1D Gaussian data with optional Gaussian noise.

    :param mean: Mean of the Gaussian distribution.
    :param std_dev: Standard deviation of the Gaussian distribution.
    :param num_points: Number of points to generate.
    :param noise_std: Standard deviation of Gaussian noise to add to each point. If None, no noise is added.
    :return: Array of generated data points.
    """
    # Generate the 1D Gaussian data
    data_1d = np.random.normal(mean, std_dev, num_points)

    # Add noise if specified
    if noise_std is not None:
        noise = np.random.normal(0, noise_std, data_1d.shape)
        data_1d += noise

    return data_1d
