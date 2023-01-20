import numpy as np
import torch


def get_alpha_bar(variance_schedule):
    """Return the noise variance sequence 'alpha_bar' from the variance schedule 'beta'."""
    return np.cumprod(1 - np.asarray(variance_schedule))


def forward_noise(img, t, alpha_bar, rng=None):
    """Give an image and a timestep, return the couple `(x_noisy, t), noise`.
    
    Basically, it generates the noise and add it to the original image.

    Parameters
    ----------
    img : np.array
        Original image, i.e. x_0
    t : int
        Timestep
    alpha_bar : array or array-like
        sequence of the noise variance.
    rng : np.random.Generator, optional
        random generator, by default None

    Returns
    -------
    (x_noisy, t), noise
    """
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
    """Custom Dataset for generating noisy images for the forward diffusion process, given a dataset and a variance schedule.

    More precisely, we start from a standard dataset of images. Given and index 'idx', this dataset returns the sample image 
    with that index: 'img' (this is basically our original image x_0).
    Then, we enrich this, creating a new dataset. Given and index 'idx', this diffusion datset returns a couple: 
    - The couple (x_noisy, t), where 't' is the timestep and 'x_noisy' is the image plus the noise at that timestep (i.e. 
      x_t).
      This couple can be thought as the input data of the network.
    - The noise 'noise' which has been added to the image at that timestep 't'.
      This can be thought of as the target label for the network.

    If this dataset is used for training, then each time an image 'img' is sampled, a different random timestep 't' is used
    for generating 'noise' and 'x_noisy'. So, across different batches, the same image is affected to different degrees of
    noise.
    If instead this datset is used for validation, then each image 'img' has a random but fixed timestep 't', which is the 
    same each time that image is sampled, for generating the same 'noise' and 'x_noisy'. So, across different batches, the 
    same image is affected by the same degree of noise.  

    Parameters
    ----------
    data : Dataset
        Dataset containing the images.
        It must support ``__getitem__``.
    variance_schedule : array or array-like, optional
        sequence of increases of the noise variance. (In the paper, this is referred to beta_t)
        Either this or ``alpha_bar`` must be provided.
    alpha_bar : array or array-like, optional
        sequence of the noise variance.
        Either this or ``variance_schedule`` must be provided.
    validation : bool, optional
        whether this dataset will be used for testing or training, by default False.
        If the dataset is used for training, the generation of the timestep 't' for each image is randomic and changes each 
        time this image is sampled: so, across different batches, this same image has different 't'.
        Instead, if the dataset is used for testing, the generation of the timestep 't' for an image is still randomic but it 
        is fixed across different sampling/batches: it is done only one time at the beginning.
    transform : torchvision.transforms.Transform, optional
        Optional transformation to apply to the image data, by default None
    random_seed : optional
        random seed to be passed to the numpy random number generator.
    *args, **kwargs
        ``passed to torch.utils.data.Dataset``.
    """

    def __init__(self, data, variance_schedule=None, alpha_bar=None, transform=None, validation=False, random_seed=None,  
                 *args, **kwargs):
        if variance_schedule is None and alpha_bar is None:
            raise TypeError("Either variance_schedule or alpha_bar must be provided.")
        if variance_schedule is not None and alpha_bar is not None:
            raise TypeError("variance_schedule and alpha_bar are both provided.")
        if alpha_bar is None:
            alpha_bar = get_alpha_bar(variance_schedule)

        super().__init__(*args, **kwargs)
        self.data = data 
        self.alpha_bar = alpha_bar
        self.T = len(self.alpha_bar)
        self.transform = transform
        self.rng = np.random.default_rng(random_seed)
        self.validation = validation

        if validation:
            # Fixed timesteps t for the data images 
            n_samples = len(data)
            self.fixed_t = self.rng.integers(low=1, high=self.T, size=(n_samples,))        

    def __getitem__(self, idx):
        # Ge the original dataset image
        img, _ = self.data[idx]

        if self.transform:
            # Apply the optional transformation
            img = self.transform(img)

        # Sample the timestep t
        if not self.validation:
            t = self.rng.integers(1, self.T)
        else:
            t = self.fixed_t[idx]

        # Generate and return the triple `(x_noisy, t), noise``
        return forward_noise(img, t, self.alpha_bar, self.rng)

    def __len__(self):
        return len(self.data)
