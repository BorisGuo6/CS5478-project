from aerial_gym.utils.vae.VAE2 import VAE
import torch
import os
import torch.nn as nn


def clean_state_dict(state_dict):
    clean_dict = {}
    for key, value in state_dict.items():
        if "module." in key:
            key = key.replace("module.", "")
        if "dronet." in key:
            key = key.replace("dronet.", "encoder.")
        clean_dict[key] = value
    return clean_dict


class VAEImageEncoder:
    """
    Class that wraps around the VAE class for efficient inference for the aerial_gym class
    """

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.vae_model = VAE(input_dim=1, latent_dim=self.config.latent_dims).to(device)
        # combine module path with model file name
        weight_file_path = os.path.join(self.config.model_folder, self.config.model_file)
        # load model weights
        print("Loading weights from file: ", weight_file_path)
        state_dict = clean_state_dict(torch.load(weight_file_path))
        self.vae_model.load_state_dict(state_dict)
        self.vae_model.eval()

    def encode(self, image_tensors):
        """
        Class to encode the set of images to a latent space. We can return both the means and sampled latent space variables.
        """
        with torch.no_grad():
            # need to squeeze 0th dimension and unsqueeze 1st dimension to make it work with the VAE
            image_tensors = image_tensors.squeeze(0).unsqueeze(1)
            x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
            if self.config.image_res != (x_res, y_res):
                interpolated_image = torch.nn.functional.interpolate(
                    image_tensors,
                    self.config.image_res,
                    mode=self.config.interpolation_mode,
                )
            else:
                interpolated_image = image_tensors
            z_sampled, means, *_ = self.vae_model.encode(interpolated_image)
        if self.config.return_sampled_latent:
            returned_val = z_sampled
        else:
            returned_val = means
        return returned_val

    def decode(self, latent_spaces):
        """
        Decode a latent space to reconstruct full images
        """
        with torch.no_grad():
            if latent_spaces.shape[-1] != self.config.latent_dims:
                print(
                    f"ERROR: Latent space size of {latent_spaces.shape[-1]} does not match network size {self.config.latent_dims}"
                )
            decoded_image = self.vae_model.decode(latent_spaces)
        return decoded_image

    def get_latent_dims_size(self):
        """
        Function to get latent space dims
        """
        return self.config.latent_dims


class VAE_train_test:
    """
    Class that wraps around the VAE class for efficient training and testing
    """

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.vae_model = VAE(input_dim=1, latent_dim=self.config.latent_dims).to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.vae_model.parameters(), lr=self.config.lr)
    
    def encode(self, image_tensors):
        """
        Class to encode the set of images to a latent space. We can return both the means and sampled latent space variables.
        """
        with torch.no_grad():
            # need to squeeze 0th dimension and unsqueeze 1st dimension to make it work with the VAE
            image_tensors = image_tensors.squeeze(0).unsqueeze(1)
            x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
            if self.config.image_res != (x_res, y_res):
                interpolated_image = torch.nn.functional.interpolate(
                    image_tensors,
                    self.config.image_res,
                    mode=self.config.interpolation_mode,
                )
            else:
                interpolated_image = image_tensors
            z_sampled, means, log_var = self.vae_model.encode(interpolated_image)
        if self.config.return_sampled_latent:
            returned_val = z_sampled
        else:
            returned_val = means
        return returned_val, means, log_var

    def decode(self, latent_spaces):
        """
        Decode a latent space to reconstruct full images
        """
        with torch.no_grad():
            if latent_spaces.shape[-1] != self.config.latent_dims:
                print(
                    f"ERROR: Latent space size of {latent_spaces.shape[-1]} does not match network size {self.config.latent_dims}"
                )
            decoded_image = self.vae_model.decode(latent_spaces)
        return decoded_image

    def get_latent_dims_size(self):
        """
        Function to get latent space dims
        """
        return self.config.latent_dims

    def VAELoss(self, x, x_recon, mu, log_var):
        """
        Function to calculate the loss for the VAE using reconstruction loss and KL divergence
        """
        recon_loss = nn.functional.mse_loss(input=x_recon, target=x, reduction='mean')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_divergence
    
    
    def train(self, image_tensors):
        """
        Train the VAE model
        """
        self.vae_model.train()
        self.optimizer.zero_grad()
        # image_tensors = image_tensors.squeeze(0).unsqueeze(1)
        x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
        if self.config.image_res != (x_res, y_res):
            interpolated_image = torch.nn.functional.interpolate(
                image_tensors, self.config.image_res, mode=self.config.interpolation_mode
            )
        else:
            interpolated_image = image_tensors
        z_sampled, means, log_var = self.vae_model.encode(interpolated_image)
        decoded_image = self.vae_model.decode(z_sampled)
        if decoded_image.shape != interpolated_image.shape:
            decoded_image = torch.nn.functional.interpolate(
                decoded_image, (x_res, y_res), mode=self.config.interpolation_mode
            )
        loss = self.VAELoss(interpolated_image, decoded_image, means, log_var)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, image_tensors):
        """
        Test the VAE model
        """
        self.vae_model.eval()
        # image_tensors = image_tensors.squeeze(0).unsqueeze(1)
        x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
        if self.config.image_res != (x_res, y_res):
            interpolated_image = torch.nn.functional.interpolate(
                image_tensors, self.config.image_res, mode=self.config.interpolation_mode
            )
        else:
            interpolated_image = image_tensors
        z_sampled, means, log_var = self.vae_model.encode(interpolated_image)
        decoded_image = self.vae_model.decode(z_sampled)
        if decoded_image.shape != interpolated_image.shape:
            decoded_image = torch.nn.functional.interpolate(
                decoded_image, (x_res, y_res), mode=self.config.interpolation_mode
            )
        loss = self.VAELoss(interpolated_image, decoded_image, means, log_var)
        return loss.item()
    
    def save_model(self, model_path):
        """
        Save the model
        """
        torch.save(self.vae_model.state_dict(), model_path)

    def load_model(self, model_path):
        """
        Load the model
        """
        self.vae_model.load_state_dict(torch.load(model_path))
        self.vae_model.eval()
        return self.vae_model



def train_test_vae():
    # train and test the VAE model

    # initialise config for training
    config = {
        "data_folder": "./data/Dataset_PULP_Dronet_v3_training/",
        "data_folder_test": "./data/Dataset_PULP_Dronet_v3_testing/",
        "log_folder": "./logs/",
        "model_weights_path": "./weights/pulp_dronet_v3_vae.pth",
        "gpu": '0',
        "workers": 4,
        "checkpoint_path": "./checkpoints/",
        "model_name": "pulp_dronet_v3_vae",
        "resume_training": False,
        "epochs": 100,
        "batch_size": 32,
        "lr": 0.001,
        "latent_dims": 64,
        "image_res": (200, 200),
        "interpolation_mode": "bilinear",
        "early_stopping": False,
        "early_stopping_patience": 15,
        "delta": 0.0,
    }
    config = type("Config", (object,), config)()
    
    # select the device
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    print("pytorch version: ", torch.__version__)

    # initialise the VAE model
    from torchinfo import summary
    vae_wrapper = VAE_train_test(config, device)
    vae_model = vae_wrapper.vae_model
    summary(vae_model, (1, 1, 200, 200))
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    from aerial_gym.utils.vae.utility import EarlyStopping
    if config.early_stopping:
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience, verbose=True, delta=config.delta
        )
    # print dataset paths
    from os import listdir
    print("Training data path:\t", listdir(config.data_folder))
    print("Testing data path:\t", listdir(config.data_folder_test))
    # load the data
    from aerial_gym.utils.vae.classes import Dataset
    # training dataset
    dataset = Dataset(config.data_folder)
    dataset.initialize_from_filesystem()
    # testing dataset
    dataset_test = Dataset(config.data_folder_test)
    dataset_test.initialize_from_filesystem()
    # transformations
    from torchvision import transforms
    transformations = transforms.Compose(
        [
            transforms.CenterCrop(200),
            transforms.ToTensor(),
        ]
    )
    # load the data
    from aerial_gym.utils.vae.utility import DronetDatasetV3
    from torch.utils.data import DataLoader
    train_dataset = DronetDatasetV3(
        transform=transformations, dataset=dataset, selected_partition="train"
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.workers
    )
    # validation dataset
    val_dataset = DronetDatasetV3(
        transform=transformations, dataset=dataset, selected_partition="valid"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers
    )
    # testing dataset
    test_dataset = DronetDatasetV3(
        transform=transformations, dataset=dataset_test, selected_partition="test"
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers
    )

    # create training directory for logs
    from datetime import datetime
    training_dir = os.path.join(config.log_folder, 'training')
    training_model_dir = os.path.join(training_dir, config.model_name)
    log_dir = os.path.join(training_model_dir, 'logs')
    tensorboard_dir = os.path.join(training_model_dir, 'tensorboard_'+ datetime.now().strftime('%b%d_%H:%M:%S'))
    checkpoint_dir = os.path.join(training_model_dir, 'checkpoints')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # write the training/validation/testing paths to logs. By doing so we keep track of which dataset we use (augemented VS not augmented)
    from aerial_gym.utils.vae.utility import write_log
    write_log(log_dir, 'Training data path:\t'   + config.data_folder, prefix='train', should_print=False, mode='a', end='\n')
    write_log(log_dir, 'Validation data path:\t' + config.data_folder, prefix='valid', should_print=False, mode='a', end='\n')
    write_log(log_dir, 'Testing data path:\t'    + config.data_folder_test, prefix='test', should_print=False, mode='a', end='\n')

    # logging utils
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_writer = SummaryWriter(tensorboard_dir)
    # training stats
    from aerial_gym.utils.vae.utility import AverageMeter
    loss_train = AverageMeter('Loss', ':.4f')
    loss_val = AverageMeter('Loss', ':.4f')
    loss_test = AverageMeter('Loss', ':.4f')
    # dataframes for csv files
    import pandas as pd
    df_train = pd.DataFrame( columns=['Epoch','Loss'])
    df_valid = pd.DataFrame( columns=['Epoch','Loss'])
    df_test= pd.DataFrame( columns=['Epoch','Loss'])

    # training loop
    from tqdm import tqdm
    from aerial_gym.utils.vae.utility import custom_mse, custom_bce, custom_accuracy
    for epoch in range(config.epochs+1):
        # reset stats
        for obj in [loss_train]:
            obj.reset()
        # training
        print("Epoch: %d/%d" %  (epoch, config.epochs))
        vae_model.train()
        with tqdm(total=len(train_loader), desc='Train', disable=False) as pbar:
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                loss = vae_wrapper.train(inputs)
                # update stats
                loss_train.update(loss)
                # update progress bar
                pbar.set_postfix({'Loss': loss_train.avg})
                pbar.update(1)
        
        # validation
        vae_model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc='Valid', disable=False) as pbar:
                for i, images in enumerate(val_loader):
                    images = images[0].to(device)
                    loss = vae_wrapper.test(images)
                    loss_val.update(loss)
                    # update progress bar
                    pbar.set_postfix({'Loss': loss_val.avg})
                    pbar.update(1)
        # testing
        vae_model.eval()
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc='Test', disable=False) as pbar:
                for i, images in enumerate(test_loader):
                    images = images[0].to(device)
                    loss = vae_wrapper.test(images)
                    loss_test.update(loss)
                    # update progress bar
                    pbar.set_postfix({'Loss': loss_test.avg})
                    pbar.update(1)
        # write to tensorboard
        tensorboard_writer.add_scalar('Loss/train', loss_train.avg, epoch)
        tensorboard_writer.add_scalar('Loss/valid', loss_val.avg, epoch)
        tensorboard_writer.add_scalar('Loss/test', loss_test.avg, epoch)
        # write to csv
        to_append = pd.DataFrame([[epoch, loss_train.avg]], columns=['Epoch','Loss'])
        df_train = pd.concat([df_train, to_append], ignore_index=True)
        df_train.to_csv(os.path.join(log_dir, 'train.csv'), index=False)
        # save model
        if epoch % 10 == 0:
            vae_wrapper.save_model(os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))
            print("Model saved at epoch: ", epoch)
        # early stopping
        if config.early_stopping:
            early_stopping(loss_train.avg, vae_wrapper.vae_model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    # save the model
    vae_wrapper.save_model(config.model_weights_path)
    print("Model saved at: ", config.model_weights_path)
    # close tensorboard writer
    tensorboard_writer.close()

def main():
    from torchinfo import summary
    #print model weights
    # file_path = "./weights/pulp_dronet_v3_vae.pth"
    # print("Model weights path: ", file_path)
    # load the model
    model = VAE(input_dim=1, latent_dim=64)
    summary(model, (1, 1, 200, 200))
    # model.load_state_dict(torch.load(file_path))
    # print each layer and its weights shape
    for name, param in model.named_parameters():
        print(name, param.shape)



if __name__ == "__main__":
    main()