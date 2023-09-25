from __future__ import print_function

import argparse
import logging
import os

import tensorflow as tf
from model import MyModel
from utils import generate_triton_config_mnist

from pathlib import Path

import wandb
from dataloader import MnistDataloader


logging.basicConfig(level=logging.DEBUG)

if os.getenv("WANDB_API_KEY") is None:
    raise Exception("please set WANDB_API_KEY key.")

wandb.login()


def train(config):
    # create data loader from the train / test channels

    mnist_data_loader = MnistDataloader()
    train_loader = mnist_data_loader.get_train_ds(config.batch_size)
    test_loader = mnist_data_loader.get_test_ds(config.batch_size)

    model = MyModel()
    model.compile()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, beta_1=args.beta_1, beta_2=args.beta_2)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
        return

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_fn(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        return

    print("Training starts ...")
    for epoch in range(config.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for batch, (images, labels) in enumerate(train_loader):
            train_step(images, labels)

        for images, labels in test_loader:
            test_step(images, labels)

        print(
            f"epoch {epoch + 1}, "
            f"train loss: {train_loss.result()}, "
            f"train accuracy: {train_accuracy.result() * 100}, "
            f"test loss: {test_loss.result()}, "
            f"test accuracy: {test_accuracy.result() * 100}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train loss": train_loss.result(),
                "train accuracy": train_accuracy.result() * 100,
                "test loss": test_loss.result(),
                "test accuracy": test_accuracy.result() * 100,
            }
        )

    ckpt_dir = Path(args.model_dir) / args.model_name / args.version / "model.savedmodel"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(ckpt_dir)
    generate_triton_config_mnist(str(Path(ckpt_dir).parents[1]), args.model_name)
    return


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--model_name", type=str, default="mnist")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--version", type=str, default="1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config_defaults = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "version": args.version,
    }

    wandb.init(project="digit_classification", config=config_defaults)
    train(wandb.config)
