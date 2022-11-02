import numpy as np
import torch


def get_alpha_bar(variance_schedule):
    """Return the noise variance sequence from the variance schedule."""
    return np.cumprod(1 - np.asarray(variance_schedule))


def forward_noise(img, t, alpha_bar, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.standard_normal(img.shape, dtype=np.float32)

    l1 = np.sqrt(alpha_bar[t])
    l2 = np.sqrt(1 - alpha_bar[t])

    return (l1*img + l2*noise, t), noise


class ShuffleColorsTransform:
    def __init__(self, n_channels=3, seed=None):
        self.rng = np.random.default_rng(seed)
        self.p = np.arange(n_channels)

    def __call__(self, x):
        self.rng.shuffle(self.p)
        return x[self.p]


class DiffusionDataset(torch.utils.data.Dataset):
    """Custom Dataset for generating noisy images for
    the forward diffusion process,
    given a dataset and a variance schedule.

    Parameters
    ----------
    data : Dataset
        it must support ``__getitem__``.
    variance_schedule : array or array-like, optional
        sequence of increases of the noise variance.
        Either this or ``alpha_t`` must be provided.
    alpha_t : array or array-like, optional
        sequence of the noise variance.
        Either this or ``variance_schedule`` must be provided.
    transform : torchvision.transforms.Transform, optional
    random_seed : optional
        random seed to be passed to the numpy random number generator.
    *args, **kwargs
        ``passed to torch.utils.data.Dataset``.
    """

    def __init__(self, data, variance_schedule=None, alpha_bar=None, transform=None, random_seed=None, *args, **kwargs):
        if variance_schedule is None and alpha_bar is None:
            raise TypeError("Either variance_schedule or alpha_t must be provided.")
        if variance_schedule is not None and alpha_bar is not None:
            raise TypeError("variance_schedule and alpha_t are both provided.")
        if alpha_bar is None:
            alpha_bar = np.cumprod(1 - np.asarray(variance_schedule))

        super().__init__(*args, **kwargs)
        self.data = data
        self.alpha_bar = alpha_bar
        self.T = len(self.alpha_bar)
        self.transform = transform
        self.rng = np.random.default_rng(random_seed)

    def __getitem__(self, idx):
        img, _ = self.data[idx]

        if self.transform:
            img = self.transform(img)

        t = self.rng.integers(1, self.T)
        return forward_noise(img, t, self.alpha_bar, self.rng)

    def __len__(self):
        return len(self.data)
