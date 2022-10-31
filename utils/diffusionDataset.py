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


class DiffusionDataLoader(torch.utils.data.DataLoader):
    """Custom DataLoader for also sampling the timestep.
    Maybe not needed, let's see.
    """
    def __init__(self, timestep_batching_function, timestep_batch_size, variance_schedule, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestep_batching_function = timestep_batching_function
        self.timestep_batch_size = timestep_batch_size
        self.variance_schedule = variance_schedule
        self.alpha_t = np.cumprod(1-variance_schedule)
        print()

    def __forward_noise(self, image_batch, timestep_batch):
        extended_image_batch = np.concatenate([image_batch for i in range(self.timestep_batch_size)])
        print(extended_image_batch.shape)
        noisy_image_batch = np.zeros(extended_image_batch.shape)

        for i, t in enumerate(timestep_batch):        # add noise to image batch
            alpha_t = self.alpha_t[t]
            loc = np.sqrt(alpha_t)*image_batch
            scale = np.sqrt(1-alpha_t)

            start_index = i*self.timestep_batch_size
            end_index = start_index + self.batch_size
            noisy_image_batch[start_index:end_index] = np.random.normal(loc=loc, scale=scale, size=image_batch.shape)

        extended_image_batch = torch.tensor(extended_image_batch)
        noisy_image_batch = torch.tensor(noisy_image_batch)

        return extended_image_batch, noisy_image_batch

    def __iter__(self):
        def augment_iter(old_iter):
            for image_batch, _ in old_iter:

                timestep_batch = self.timestep_batching_function(self.timestep_batch_size)

                extended_image_batch, noisy_image_batch = self.__forward_noise(image_batch, timestep_batch)
                yield noisy_image_batch, extended_image_batch, timestep_batch

        return augment_iter(super().__iter__())
