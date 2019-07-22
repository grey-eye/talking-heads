import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import torch
from PIL import Image
from torch.nn import DataParallel
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader

import config
import network
from dataset import preprocess_dataset, VoxCelebDataset

import matplotlib.pyplot as plt


class ParallelDiscriminator(DataParallel):
    @property
    def W(self):
        return self.module.W


def meta_train(gpu, dataset_path, continue_id):
    run_start = datetime.now()
    logging.info('===== META-TRAINING =====')
    # GPU / CPU --------------------------------------------------------------------------------------------------------
    if gpu:
        dtype = torch.cuda.FloatTensor
        torch.set_default_tensor_type(dtype)
        logging.info(f'Running on GPU: {torch.cuda.current_device()}.')
    else:
        dtype = torch.FloatTensor
        torch.set_default_tensor_type(dtype)
        logging.info(f'Running on CPU.')

    # DATASET-----------------------------------------------------------------------------------------------------------
    logging.info(f'Training using dataset located in {dataset_path}')
    dataset = VoxCelebDataset(
        root=dataset_path,
        extension='.vid',
        shuffle_frames=True,
        transform=transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    )
    dataset = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # NETWORK ----------------------------------------------------------------------------------------------------------

    E = network.Embedder().type(dtype)
    G = network.Generator().type(dtype)
    D = network.Discriminator(len(dataset)).type(dtype)

    optimizer_E_G = Adam(
        params=list(E.parameters()) + list(G.parameters()),
        lr=config.LEARNING_RATE_E_G
    )
    optimizer_D = Adam(
        params=D.parameters(),
        lr=config.LEARNING_RATE_D
    )

    criterion_E_G = network.LossEG(feed_forward=True)
    criterion_D = network.LossD()

    if gpu:
        E = DataParallel(E)
        G = DataParallel(G)
        D = ParallelDiscriminator(D)
        criterion_E_G = DataParallel(criterion_E_G)
        criterion_D = DataParallel(criterion_D)

    if continue_id is not None:
        E = load_model(E, 'Embedder', continue_id)
        G = load_model(G, 'Generator', continue_id)
        D = load_model(D, 'Discriminator', continue_id)

    # TRAINING LOOP ----------------------------------------------------------------------------------------------------
    logging.info(f'Starting training loop. '
                 f'Epochs: {config.EPOCHS} '
                 f'Batches: {len(dataset)} '
                 f'Batch Size: {config.BATCH_SIZE}')

    for epoch in range(config.EPOCHS):
        epoch_start = datetime.now()
        batch_durations = []

        E.train()
        G.train()
        D.train()

        for batch_num, (i, video) in enumerate(dataset):
            batch_start = datetime.now()
            video = video.type(dtype)  # [B, K+1, 2, C, W, H]

            # Put one frame aside (frame t)
            t = video[:, -1, ...]  # [B, 2, C, W, H]
            video = video[:, :-1, ...]  # [B, K, C, W, H]
            dims = video.shape

            # Calculate average encoding vector for video
            e_in = video.reshape(dims[0] * dims[1], dims[2], dims[3], dims[4], dims[5])  # [BxK, 2, C, W, H]
            x, y = e_in[:, 0, ...], e_in[:, 1, ...]
            e_vectors = E(x, y).reshape(dims[0], dims[1], -1, 1)  # B, K, len(e), 1
            e_hat = e_vectors.mean(dim=1)

            # Generate frame using landmarks from frame t
            x_t, y_t = t[:, 0, ...], t[:, 1, ...]
            x_hat = G(y_t, e_hat)

            # Optimize E_G and D
            r_x_hat, D_act_hat = D(x_hat, y_t, i)
            r_x, D_act = D(x_t, y_t, i)

            optimizer_E_G.zero_grad()
            optimizer_D.zero_grad()

            loss_E_G = criterion_E_G(x_t, x_hat, r_x_hat, e_hat, D.W[:, i], D_act, D_act_hat).mean()
            loss_D = criterion_D(r_x, r_x_hat).mean()
            loss = loss_E_G + loss_D
            loss.backward()

            optimizer_E_G.step()
            optimizer_D.step()

            # Optimize D again
            x_hat = G(y_t, e_hat).detach()
            r_x_hat, D_act_hat = D(x_hat, y_t, i)
            r_x, D_act = D(x_t, y_t, i)

            optimizer_D.zero_grad()
            loss_D = criterion_D(r_x, r_x_hat).mean()
            loss_D.backward()
            optimizer_D.step()

            batch_end = datetime.now()
            batch_durations.append(batch_end - batch_start)
            # SHOW PROGRESS --------------------------------------------------------------------------------------------
            if (batch_num + 1) % 1 == 0 or batch_num == 0:
                avg_time = sum(batch_durations, timedelta(0)) / len(batch_durations)
                logging.info(f'Epoch {epoch + 1}: [{batch_num + 1}/{len(dataset)}] | '
                             f'Avg Time: {avg_time} | '
                             f'Loss_E_G = {loss_E_G.item():.4} Loss_D {loss_D.item():.4}')
                logging.debug(f'D(x) = {r_x.mean().item():.4} D(x_hat) = {r_x_hat.mean().item():.4}')

            # SAVE IMAGES ----------------------------------------------------------------------------------------------
            if (batch_num + 1) % 1 == 0:
                if not os.path.isdir(config.GENERATED_DIR):
                    os.makedirs(config.GENERATED_DIR)

                save_image(os.path.join(config.GENERATED_DIR, f'{datetime.now():%Y%m%d_%H%M%S%f}_x.png'), x_t[0])
                save_image(os.path.join(config.GENERATED_DIR, f'{datetime.now():%Y%m%d_%H%M%S%f}_x_hat.png'), x_hat[0])

            if (batch_num + 1) % 100 == 0:
                save_model(E, 'Embedder', gpu)
                save_model(G, 'Generator', gpu)
                save_model(D, 'Discriminator', gpu)

        # SAVE MODELS --------------------------------------------------------------------------------------------------

        save_model(E, 'Embedder', gpu, run_start)
        save_model(G, 'Generator', gpu, run_start)
        save_model(D, 'Discriminator', gpu, run_start)
        epoch_end = datetime.now()
        logging.info(f'Epoch {epoch + 1} finished in {epoch_end - epoch_start}. '
                     f'Average batch time: {sum(batch_durations, timedelta(0)) / len(batch_durations)}')


def save_model(model, name, gpu, time_for_name=None):
    if time_for_name is None:
        time_for_name = datetime.now()

    model.eval()

    if gpu:
        model.cpu()

    if not os.path.exists(config.MODELS_DIR):
        os.makedirs(config.MODELS_DIR)

    filename = f'{name}_{time_for_name:%Y%m%d_%H%M}.pth'
    torch.save(
        model.state_dict(),
        os.path.join(config.MODELS_DIR, filename)
    )

    if gpu:
        model.cuda()

    logging.info(f'Model saved: {filename}')

    model.train()


def load_model(model, name, continue_id):
    filename = f'{name}_{continue_id}.pth'
    model.load_state_dict(
        torch.load(
            os.path.join(config.MODELS_DIR, filename)
        )
    )
    return model


def save_image(filename, data):
    data = data.clone().detach().cpu()

    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.numpy()
    img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def imshow(data):
    data = data.clone().detach().cpu()

    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.numpy()
    img = ((img * std + mean).transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    plt.imshow(img)
    plt.show()


def main():
    # ARGUMENTS --------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Talking Heads')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # ARGUMENTS: DATASET PRE-PROCESSING  -------------------------------------------------------------------------------
    dataset_parser = subparsers.add_parser("dataset", help="Pre-process the dataset for its use.")
    dataset_parser.add_argument("--source", type=str, required=True,
                                help="Path to the source folder where the raw VoxCeleb dataset is located.")
    dataset_parser.add_argument("--output", type=str, required=True,
                                help="Path to the folder where the pre-processed dataset will be stored.")
    dataset_parser.add_argument("--size", type=int, default=0,
                                help="Number of videos from the dataset to process.")
    dataset_parser.add_argument("--gpu", action="store_true",
                                help="Run the model on GPU.")
    dataset_parser.add_argument("--overwrite", action="store_true",
                                help="Add this flag to overwrite already pre-processed files. The default functionality"
                                     "is to ignore videos that have already been pre-processed.")

    # ARGUMENTS: META_TRAINING  ----------------------------------------------------------------------------------------
    train_parser = subparsers.add_parser("meta-train", help="Starts the meta-training process.")
    train_parser.add_argument("--dataset", type=str, required=True,
                              help="Path to the pre-processed dataset.")
    train_parser.add_argument("--gpu", action="store_true",
                              help="Run the model on GPU.")
    train_parser.add_argument("--continue_id", type=str, default=None,
                              help="Id of the models to continue training.")

    args = parser.parse_args()

    # LOGGING ----------------------------------------------------------------------------------------------------------

    if not os.path.isdir(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=os.path.join(config.LOG_DIR, f'{datetime.now():%Y%m%d}.log'),
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # EXECUTE ----------------------------------------------------------------------------------------------------------
    try:
        if args.subcommand == "meta-train":
            meta_train(
                dataset_path=args.dataset,
                gpu=(torch.cuda.is_available() and args.gpu),
                continue_id=args.continue_id,
            )
        elif args.subcommand == "dataset":
            preprocess_dataset(
                source=args.source,
                output=args.output,
                device='cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu',
                size=args.size,
                overwrite=args.overwrite,
            )
        else:
            print("invalid command")
    except Exception as e:
        logging.error(f'Something went wrong: {e}')
        raise e


if __name__ == '__main__':
    main()
