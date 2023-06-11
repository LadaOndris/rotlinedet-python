import tensorflow as tf

from nn.data_loader import DataLoader
from nn.model import get_model


class Trainer:

    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, num_epochs):
        self.model.compile(
            loss=self.loss_fn,
            optimizer=self.optimizer,
            metrics='accuracy'
        )
        history = self.model.fit(
            self.train_loader,
            steps_per_epoch=1000,
            validation_data=self.val_loader,
            validation_steps=100,
            epochs=num_epochs,
        )

        return history.history['loss'], history.history['val_loss']


def get_dataset(type: str, batch_size: int,
                stripe_width: int, image_height: int):
    data_loader = DataLoader(f'data/without_laser/{type}',
                             stripe_width=stripe_width,
                             line_width_range=(1, 12),
                             length_range=(0.33, 1),
                             intensity_range=(16, 128),
                             target_image_height=image_height)
    dataset = data_loader.build_pipeline(batch_size=batch_size)
    return dataset


if __name__ == "__main__":
    stripe_width = 45
    image_height = 480
    batch_size = 8

    train_dataset = get_dataset('train', batch_size, stripe_width, image_height)
    val_dataset = get_dataset('val', batch_size, stripe_width, image_height)
    model = get_model(image_size=(image_height, stripe_width))
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    trainer = Trainer(model, train_dataset, val_dataset, loss_fn, 'Adam')
    trainer.train(num_epochs=50)
