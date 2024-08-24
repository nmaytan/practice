import numpy as np
from matplotlib import pyplot as plt


def plot_beam_and_noise(
    particle_distribution, H, H_noised, H_filtered, noise_raw, noise_filter, X, Y
):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Components of beam image simulation")

    axs[0, 0].set_title("Raw particle distribution")
    axs[0, 0].scatter(particle_distribution[:, 0], particle_distribution[:, 1])

    axs[0, 1].set_title("Simulated Detector Image (Gaussian Spot) - Raw")
    axs[0, 1].set_xlabel("width")
    axs[0, 1].set_ylabel("height")
    axs[0, 1].axis("equal")
    c2 = axs[0, 1].contourf(X, Y, H, cmap="plasma", levels=100)
    cbar2 = fig.colorbar(c2, label="Beam Intensity")
    # ^contourf coordinates could be specified with 1D or 2D arrays and should be equivalent here,
    #  we use the 2D arrays signature for an excuse to use meshgrid()

    axs[1, 0].set_title("Simulated Detector Image (Gaussian Spot) - Noised")
    axs[1, 0].set_xlabel("width")
    axs[1, 0].set_ylabel("height")
    axs[1, 0].axis("equal")
    c3 = axs[1, 0].contourf(X, Y, H_noised, cmap="plasma", levels=100)
    cbar3 = fig.colorbar(c3, label="Beam Intensity")

    axs[1, 1].set_title("Simulated Detector Image (Gaussian Spot) - Filtered")
    axs[1, 1].set_xlabel("width")
    axs[1, 1].set_ylabel("height")
    axs[1, 1].axis("equal")
    c4 = axs[1, 1].contourf(X, Y, H_filtered, cmap="plasma", levels=100)
    cbar4 = fig.colorbar(c4, label="Beam Intensity")

    axs[0, 2].set_title("Simulated Noise - Raw")
    axs[0, 2].set_xlabel("width")
    axs[0, 2].set_ylabel("height")
    axs[0, 2].axis("equal")
    c5 = axs[0, 2].contourf(X, Y, noise_raw, cmap="magma", levels=100)
    # c5.set_clim(-1, 1)
    cbar5 = fig.colorbar(c5, label="Noise Intensity")

    axs[1, 2].set_title("Simulated Noise - Filter")
    axs[1, 2].set_xlabel("width")
    axs[1, 2].set_ylabel("height")
    axs[1, 2].axis("equal")
    c6 = axs[1, 2].contourf(X, Y, noise_filter, cmap="magma", levels=100)
    # c6.set_clim(-1, 1)
    cbar6 = fig.colorbar(c6, label="Noise Intensity")

    plt.show()


def main():
    rng = np.random.default_rng(0)
    seed = 55  # set a different seed to fuzz noise
    mean = [0, 0]  # center the beam
    cov = [[1, 0.2], [0.2, 1]]  # shape of the beam, matrix should be symmetric
    # ^slightly positive covariance to make simulated beam slightly out-of-round.
    size = 1000000  # how many samples to take
    axes_max = 100  # unit-length of distribution axes
    histogram_bins = 100  # number of histogram bins to use
    noise_value = 0.15  # noise magnitude
    fuzz_value = 0.05  # fuzz magnitude

    # generate 2D distribution
    particle_distribution = rng.multivariate_normal(mean, cov, size)

    # make positive and normalize
    particle_distribution += np.absolute(particle_distribution.min())
    particle_distribution /= particle_distribution.max()
    particle_distribution *= axes_max

    x_min, x_max = 0, axes_max
    y_min, y_max = 0, axes_max
    x_bin_edges = np.linspace(x_min, x_max, histogram_bins + 1)
    y_bin_edges = np.linspace(y_min, y_max, histogram_bins + 1)
    # edges must be bins + 1

    # create the 2D histogram (density/intensity)
    # there are multiple ways to specify bins which should be equivalent here,
    # but we use the [array, array] signature for linspace() for practice
    # H, xedges, yedges = np.histogram2d(beam_clean[:, 0], beam_clean[:, 1], bins=100, density=True)
    H, xedges, yedges = np.histogram2d(
        particle_distribution[:, 0],
        particle_distribution[:, 1],
        bins=[x_bin_edges, y_bin_edges],
        density=False,
    )
    # ^density setting (return pdf vs raw count) doesn't matter; we are going to normalize the histogram
    #  so that we can apply noise

    H /= H.max()  # Normalize the histogram
    H = H.T
    # ^per NumPy docs, the returned histogram reverses X and Y axes,
    #  so transpose before visualizing.

    # Generate white noise to apply to image
    # also generate slightly different noise to make image filter imperfect
    # noise can additive or subtractive
    noise_raw = rng.uniform(noise_value * -1, noise_value, H.shape)
    noise_fuzz = np.random.default_rng(seed).uniform(
        fuzz_value * -1, fuzz_value, H.shape
    )

    # add noise source to image
    H_noised_unclipped = H + noise_raw
    # clip to display image within sensor limits
    H_noised = np.clip(H_noised_unclipped, 0, 1)
    # discoverable noise filter will be limited by sensor range
    # also fuzz the filter to guarantee imperfection
    noise_filter = H_noised - H + noise_fuzz
    # subtract noise filter from image
    H_filtered_unclipped = H_noised_unclipped - noise_filter
    # clip to display image within sensor limits
    H_filtered = np.clip(H_filtered_unclipped, 0, 1)

    X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    #  ^this is setting the bin positions, so must be edges - 1
    #   "ij" indexing is needed to get expected axes convention

    plot_beam_and_noise(
        particle_distribution, H, H_noised, H_filtered, noise_raw, noise_filter, X, Y
    )


if __name__ == "__main__":
    main()
