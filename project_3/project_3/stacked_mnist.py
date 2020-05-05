import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from enum import auto, Enum


class DataMode(Enum):
    """
    MONO_BINARY_COMPLETE:
    Standard one-channel MNIST dataset. All classes represented. Binarized.
    Use for learning standard generative models, check coverage, etc.
    """
    MONO_BINARY_COMPLETE = auto()

    """
    MONO_BINARY_MISSING:
    Standard one-channel MNIST dataset, but one class taken out.
    Use for testing "anomaly detection". Binarized.
    """
    MONO_BINARY_MISSING = auto()

    """
    MONO_FLOAT_COMPLETE:
    Standard one-channel MNIST dataset, All classes there.
    Use for testing coverage etc. Data represented by their float values (not binarized).
    Can be easier to learn, but does not give as easy a probabilistic understanding.
    """
    MONO_FLOAT_COMPLETE = auto()

    """
    MONO_FLOAT_MISSING:
    Standard one-channel MNIST dataset, but one class taken out.
    Use for testing anomaly detection use-case. Data represented by their float values (not binarized).
    Can be easier to learn, but does not give as easy a probabilistic understanding.
    """
    MONO_FLOAT_MISSING = auto()

    """
    COLOR_<WHATEVER>:
    These are *STACKED* versions of MNIST, i.e., three color channels with one digit in each channel.
    Subgroups [BINARY|FLOAT]_[COMPLETE|MISSING]: As above for the MONO versions
    """
    COLOR_BINARY_COMPLETE = auto()
    COLOR_BINARY_MISSING = auto()
    COLOR_FLOAT_COMPLETE = auto()
    COLOR_FLOAT_MISSING = auto()


class StackedMNISTData:
    """
    The class will provide examples of data by sampling uniformly from MNIST data. We can do this one-channel
    (black-and-white images) or multi-channel (*STACKED* data), in which the last dimension will be the
    "color channel" of the image. In this case, 3 channels is the most natural, in which case each channel is
    one color (e.g. RGB).

    In the RGB-case we use channel 0 counting the ones for the red channel,
    channel 1 counting the tens for the green channel, and channel 2 counting the hundreds for the blue.
    """

    def __init__(self, mode: DataMode, default_batch_size: np.int = 256) -> None:
        #def __init__(self, default_batch_size: np.int = 256, channels: np.int = 1, make_binary: bool = False) -> None:
        # Load MNIST and put in internals
        self.default_batch_size = default_batch_size

        # Color or not
        if mode in [DataMode.MONO_BINARY_COMPLETE,
                    DataMode.MONO_BINARY_MISSING,
                    DataMode.MONO_FLOAT_COMPLETE,
                    DataMode.MONO_FLOAT_MISSING]:
            self.channels = 1
        else:
            self.channels = 3

        # Drop digit eight?
        if mode in [DataMode.MONO_BINARY_COMPLETE,
                    DataMode.MONO_FLOAT_COMPLETE,
                    DataMode.COLOR_BINARY_COMPLETE,
                    DataMode.COLOR_FLOAT_COMPLETE]:
            self.remove_class = None
        else:
            self.remove_class = 8

        # Binarize the data?
        if mode in [DataMode.MONO_BINARY_COMPLETE,
                    DataMode.MONO_BINARY_MISSING,
                    DataMode.COLOR_BINARY_COMPLETE,
                    DataMode.COLOR_BINARY_MISSING]:
            self.make_binary = True
        else:
            self.make_binary = False

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
        self.train_images = np.expand_dims(self.train_images, axis=-1)
        self.test_images = np.expand_dims(self.test_images, axis=-1)
        self.train_images, self.train_labels = self.__prepare_data_set(training=True)
        self.test_images, self.test_labels = self.__prepare_data_set(training=False)

    def get_full_data_set(self, training: bool = True) -> tuple:
        """
        Get the full, prepared dataset. Since the dataset is so small, this works well.
        Then we cans end it directly to keras' fit-method
        """

        if training is True:
            images, classes = self.train_images, self.train_labels
        else:
            images, classes = self.test_images, self.test_labels
        return images, classes

    def __prepare_data_set(self, training: bool = True) -> tuple:
        """
        Do transformations of the data as needed: Make binary, stacking, rescaling
        """
        if training:
            images = self.train_images
            labels = self.train_labels
        else:
            images = self.test_images
            labels = self.test_labels

        # Recode, incl scale to 0--1
        images = images / 255.0
        labels = labels.astype(np.int)

        # Drop specific digit?  --- will only do this from training data
        if training is True:
            images = images[labels != self.remove_class]
            labels = labels[labels != self.remove_class]

        # Make binary?
        if self.make_binary is True:
            images[images >= .5] = 1.
            images[images < .5] = 0.
            images = images.astype(np.int)

        # Make colorful?
        if self.channels > 1:

            indexes = np.random.choice(a=images.shape[0],
                                       size=(images.shape[0], self.channels))

            # Choose the images to get a thing that is <default_batch_size, 28, 28, self.channels>
            # where the last dim is over the dims of the indexes
            generated_images = np.zeros(shape=(images.shape[0], 28, 28, self.channels),
                                        dtype=images.dtype)
            generated_labels = np.zeros(shape=(images.shape[0],), dtype=np.int)
            for channel in range(self.channels):
                generated_images[:, :, :, channel] = images[indexes[:, channel], :, :, 0]
                generated_labels += np.power(10, channel) * labels[indexes[:, channel]]

            images = generated_images
            labels = generated_labels

        return images, labels

    def get_random_batch(self, training: bool = True, batch_size: np.int = None) -> tuple:
        """
        Generate a batch of data. We can choose to use training or testing data.
        Also, we can ask for a specific batch-size (if we don't, we use the default
        defined through __init__.
        """

        batch_size = self.default_batch_size if batch_size is None else batch_size

        if training:
            images = self.train_images
            labels = self.train_labels
        else:
            images = self.test_images
            labels = self.test_labels

        indexes = np.random.choice(a=images.shape[0], size=batch_size)
        images, labels = images[indexes], labels[indexes]
        if len(images.shape) == 3:
            # Selected single image, which leads to collapse of first dim --> must add dim back
            images = np.expand_dims(images, axis=0)

        return images, labels

    def batch_generator(self, training: bool = True, batch_size: np.int = None) -> tuple:
        """
        Create a  batch generator. We can choose to use training or testing data.
        Also, we can ask for a specific batch-size (if we don't, we use the default
        defined through __init__.
        """

        batch_size = self.default_batch_size if batch_size is None else batch_size

        if training:
            images = self.train_images
            labels = self.train_labels
        else:
            images = self.test_images
            labels = self.test_labels

        start_position = 0
        no_elements = images.shape[0]

        while start_position < no_elements:
            end_position = np.min([start_position + batch_size, no_elements])

            yield images[start_position:end_position],  labels[start_position:end_position]
            start_position = end_position

    def plot_example(self, images: np.ndarray = None, labels: np.ndarray = None) -> None:
        """
        Plot data in RGB (3-channel data) or monochrome (one-channel data).
        If data is submitted, we need to generate an example.
        If there are many images, do a subplot-thing.
        """

        # Do we need to generate data?
        if images is None or labels is None:
            images, labels = self.get_random_batch(batch_size=16)

        no_images = images.shape[0]

        # Do the plotting
        plt.Figure()
        no_rows = np.ceil(np.sqrt(no_images))
        no_cols = np.ceil(no_images / no_rows)
        for img_idx in range(no_images):
            plt.subplot(no_rows, no_cols, img_idx + 1)
            if self.channels == 1:
                plt.imshow(images[img_idx, :, :, 0], cmap="binary")
            else:
                plt.imshow(images[img_idx, :, :, :].astype(np.float))
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Class is {str(labels[img_idx]).zfill(self.channels)}")

        # Show the thing ...
        plt.show()


if __name__ == "__main__":
    gen = StackedMNISTData(mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=9)
    img, cls = gen.get_random_batch(batch_size=9)
    gen.plot_example(images=img, labels=cls)

    for (img, cls) in gen.batch_generator(training=False, batch_size=2048):
        print(f"Batch has size: Images: {img.shape}; Labels {cls.shape}")

