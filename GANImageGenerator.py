import os
import tensorflow as tf
from keras.src.layers import Add
from keras.src.utils import load_img
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, \
    LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras import Model, Sequential
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
IMG_HEIGHT, IMG_WIDTH = 64, 64
CHANNELS = 1
LATENT_DIM = 250
CATEGORIES = ['Mild Dementia', 'Moderate Dementia', 'Non Demented']
NUM_CLASSES = len(CATEGORIES)


# Function to load images and labels
def load_images_and_labels(dataset_path, n = 67):
    images = []
    labels = []
    for label, category in enumerate(CATEGORIES):
        category_path = os.path.join(dataset_path, category)
        image_names = os.listdir(category_path)[:n]  # Limit to n images per category
        for image_name in image_names:
            image_path = os.path.join(category_path, image_name)
            image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
            from keras.src.utils import img_to_array
            image = img_to_array(image) / 255.0  # Normalize to [0, 1]
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)


# Load the dataset
dataset_path = '.\\dataset\\data'
images, labels = load_images_and_labels(dataset_path)


# Build the generator
# Build the generator
def build_generator():
    noise_input = Input(shape=(LATENT_DIM,))
    label_input = Input(shape=(1,))

    # Embed the label and concatenate with noise
    label_embedding = Embedding(NUM_CLASSES, LATENT_DIM)(label_input)
    label_embedding = Flatten()(label_embedding)
    combined_input = Concatenate()([noise_input, label_embedding])

    # Dense layer and reshape
    x = Dense(16 * 16 * 128, activation='relu')(combined_input)
    x = Reshape((16, 16, 128))(x)

    # Upsampling layers
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(CHANNELS, kernel_size=4, strides=2, padding='same', activation='tanh')(x)

    model = Model([noise_input, label_input], x)
    return model


# Wasserstein loss function
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)
# Build the discriminator
def build_discriminator():
    image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    label_input = Input(shape=(1,))

    # Embed the label and reshape to match image dimensions
    label_embedding = Embedding(NUM_CLASSES, IMG_HEIGHT * IMG_WIDTH * CHANNELS)(label_input)
    label_embedding = Flatten()(label_embedding)
    label_embedding = Reshape((IMG_HEIGHT, IMG_WIDTH, CHANNELS))(label_embedding)

    # Combine image input and label embedding
    combined_input = Concatenate()([image_input, label_embedding])

    # Convolutional layers
    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(combined_input)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    output = Dense(1)(x)  # No sigmoid activation for WGAN

    model = Model([image_input, label_input], output)
    return model

# Compile the models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5),
                      metrics=['accuracy'])

discriminator.trainable = False
noise_input = Input(shape=(LATENT_DIM,))
label_input = Input(shape=(1,))
generated_image = generator([noise_input, label_input])
validity = discriminator([generated_image, label_input])

cgan = Model([noise_input, label_input], validity)
cgan.compile(loss=wasserstein_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))


# Training function
def train_wgan(generator, discriminator, cgan, images, labels, epochs=10000, batch_size=64, n_critic=5, clip_value=0.03):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        for _ in range(n_critic):  # Train the discriminator n_critic times
            # Select a random batch of real images
            idx = np.random.randint(0, images.shape[0], half_batch)
            real_images, real_labels = images[idx], labels[idx]
            real_labels = real_labels.reshape(-1, 1)

            # Generate a batch of fake images
            noise = np.random.normal(0, 1, (half_batch, LATENT_DIM))
            fake_labels = np.random.randint(0, NUM_CLASSES, half_batch).reshape(-1, 1)
            fake_images = generator.predict([noise, fake_labels])

            # Create labels for real (1) and fake (-1)
            real_targets = np.ones((half_batch, 1))
            fake_targets = -np.ones((half_batch, 1))

            # Train the discriminator
            discriminator.trainable = True
            # Add small Gaussian noise to real and fake images
            # Set initial noise level
            initial_noise_std = 0.05  # Standard deviation of noise at the start
            final_noise_std = 0.01      # Standard deviation of noise at the end

            # Compute current noise level based on epoch
            decay_rate = 0.95  # Decay rate per epoch (must be < 1)
            current_noise_std = max(final_noise_std, initial_noise_std * (decay_rate ** epoch))

            # Add Gaussian noise to real and fake images
            real_images += np.random.normal(0, current_noise_std, real_images.shape)
            fake_images += np.random.normal(0, current_noise_std, fake_images.shape)

            d_loss_real = discriminator.train_on_batch([real_images, real_labels], real_targets)
            d_loss_fake = discriminator.train_on_batch([fake_images, fake_labels], fake_targets)
            discriminator.trainable = False
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

            # Clip discriminator weights
            for layer in discriminator.layers:
                if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                    weights = layer.get_weights()
                    weights = [tf.clip_by_value(w, -clip_value, clip_value) for w in weights]
                    layer.set_weights(weights)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        sampled_labels = np.random.randint(0, NUM_CLASSES, batch_size).reshape(-1, 1)
        valid_targets = np.ones((batch_size, 1))  # Real labels for generator

        g_loss = cgan.train_on_batch([noise, sampled_labels], valid_targets)

        # Print progress
        print(f"Epoch {epoch} / {epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

        if epoch == 10 or epoch == 30 or epoch == 60 or epoch == 90 or epoch % 25 == 0:
            sample_images(generator, epoch, 10)


# Function to generate and save sample images
def sample_images(generator, epoch, num_images=5):
    # Create a folder for the current epoch
    folder_name = f"epoch_{epoch}"
    os.makedirs(folder_name, exist_ok=True)

    # Generate images
    noise = np.random.normal(0, 1, (num_images, LATENT_DIM))
    sampled_labels = np.arange(num_images).reshape(-1, 1) % NUM_CLASSES
    generated_images = generator.predict([noise, sampled_labels])
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    # Save each image individually
    for i in range(num_images):
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.title(CATEGORIES[sampled_labels[i][0]])
        plt.axis('off')
        image_path = os.path.join(folder_name, f"image_{i}.png")
        plt.savefig(image_path)
        plt.close()
# Train the cGAN
train_wgan(generator, discriminator, cgan, images, labels)
