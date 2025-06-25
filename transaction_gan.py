import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gan_training.log'),
        logging.StreamHandler()
    ]
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TransactionDataset(Dataset):
    def __init__(self, csv_path, categorical_columns=None, label_column='isFraud'):
        logging.info(f"Loading dataset from {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column
        # Store original column names
        self.column_names = self.df.columns.tolist()
        # Handle categorical columns
        if categorical_columns is None:
            self.categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            self.categorical_columns = categorical_columns
        logging.info(f"Detected categorical columns: {self.categorical_columns}")
        # Create one-hot encoders for categorical columns
        self.categorical_encoders = {}
        for col in self.categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('MISSING')
                self.df[col] = self.df[col].astype('category')
                self.categorical_encoders[col] = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, self.categorical_encoders[col]], axis=1)
                self.df = self.df.drop(col, axis=1)
        # Handle numerical columns (excluding label)
        self.numerical_columns = [col for col in self.df.select_dtypes(include=['float64', 'int64']).columns.tolist() if col != self.label_column]
        # Fill NaN values in numerical columns with 0
        self.df[self.numerical_columns] = self.df[self.numerical_columns].fillna(0)
        # Convert all numerical columns to float32
        self.df[self.numerical_columns] = self.df[self.numerical_columns].astype('float32')
        # Scale numerical columns
        self.scaler = StandardScaler()
        self.df[self.numerical_columns] = self.scaler.fit_transform(self.df[self.numerical_columns])
        # Ensure all data is float32
        self.df = self.df.astype('float32')
        # Store features and labels separately
        self.features = self.df.drop(self.label_column, axis=1).values
        self.labels = self.df[self.label_column].values.astype('int64')
        logging.info(f"Dataset shape after preprocessing: features {self.features.shape}, labels {self.labels.shape}")
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    def inverse_transform(self, generated_data, labels=None):
        df = pd.DataFrame(generated_data, columns=self.df.drop(self.label_column, axis=1).columns)
        df[self.numerical_columns] = self.scaler.inverse_transform(df[self.numerical_columns])
        for col in self.numerical_columns:
            if 'amount' in col.lower():
                df[col] = df[col].round(2)
            else:
                df[col] = df[col].round(0)
        if labels is not None:
            df[self.label_column] = labels
        return df

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, label_dim=1):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.label_dim = label_dim
        input_dim = latent_dim + label_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, z, labels):
        # z: (batch, latent_dim), labels: (batch, label_dim)
        x = torch.cat([z, labels], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim=1):
        super(Discriminator, self).__init__()
        self.label_dim = label_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim + label_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x, labels):
        # x: (batch, input_dim), labels: (batch, label_dim)
        x = torch.cat([x, labels], dim=1)
        return self.model(x)

class TransactionGAN:
    def __init__(self, latent_dim, data_dim, label_dim=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.label_dim = label_dim
        logging.info(f"Initializing GAN on device: {device}")
        logging.info(f"Latent dimension: {latent_dim}")
        logging.info(f"Data dimension: {data_dim}")
        # Initialize networks
        self.generator = Generator(latent_dim, data_dim, label_dim).to(device)
        self.discriminator = Discriminator(data_dim, label_dim).to(device)
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        # Loss function
        self.criterion = nn.BCELoss()
        
    def train(self, dataloader, num_epochs, save_interval=10):
        logging.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            for i, real_data in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')):
                batch_size = real_data[0].size(0)
                real_data = (real_data[0].to(self.device, non_blocking=True), real_data[1].to(self.device, non_blocking=True))
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real data
                real_labels = torch.ones(batch_size, 1, device=self.device)
                real_outputs = self.discriminator(real_data[0], real_data[1])
                d_loss_real = self.criterion(real_outputs, real_labels)
                
                # Fake data
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(z, real_data[1])
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                fake_outputs = self.discriminator(fake_data.detach(), real_data[1])
                d_loss_fake = self.criterion(fake_outputs, fake_labels)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                # Generate fake data
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(z, real_data[1])
                fake_outputs = self.discriminator(fake_data, real_data[1])
                
                # Generator loss
                g_loss = self.criterion(fake_outputs, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                
                if i % 100 == 0:
                    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], '
                               f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
            
            # Save model checkpoints
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
    
    def save_checkpoint(self, epoch):
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict()
        }
        
        torch.save(checkpoint, f'{checkpoint_dir}/gan_checkpoint_epoch_{epoch}.pt')
        logging.info(f"Saved checkpoint for epoch {epoch}")
    
    def generate_samples(self, num_samples, label_value=0):
        """Generate samples with a given label (0: non-fraud, 1: fraud)"""
        logging.info(f"Generating {num_samples} samples with label {label_value}")
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            labels = torch.full((num_samples, 1), label_value, device=self.device, dtype=torch.float32)
            fake_data = self.generator(z, labels)
        return fake_data.cpu().numpy(), labels.cpu().numpy().astype(int)

def main():
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        logging.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    # Hyperparameters
    LATENT_DIM = 100
    BATCH_SIZE = 256
    NUM_EPOCHS = 200
    LABEL_DIM = 1
    # Create output directory
    os.makedirs('output', exist_ok=True)
    # Load dataset
    dataset = TransactionDataset('datasets/train_transaction.csv', label_column='isFraud')
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    # Initialize GAN
    gan = TransactionGAN(LATENT_DIM, dataset.features.shape[1], label_dim=LABEL_DIM, device=device)
    # Train the model
    gan.train(dataloader, NUM_EPOCHS)
    # Generate new samples with 10% fraud
    num_samples = 10000
    num_fraud = int(num_samples * 0.10)
    num_nonfraud = num_samples - num_fraud
    # Generate non-fraud samples
    gen_nonfraud, labels_nonfraud = gan.generate_samples(num_nonfraud, label_value=0)
    # Generate fraud samples
    gen_fraud, labels_fraud = gan.generate_samples(num_fraud, label_value=1)
    # Concatenate
    all_samples = np.vstack([gen_nonfraud, gen_fraud])
    all_labels = np.vstack([labels_nonfraud, labels_fraud]).flatten()
    # Shuffle
    idx = np.random.permutation(num_samples)
    all_samples = all_samples[idx]
    all_labels = all_labels[idx]
    # Convert generated data back to original format
    generated_df = dataset.inverse_transform(all_samples, labels=all_labels)
    # Save generated data
    output_path = 'output/generated_transactions.csv'
    generated_df.to_csv(output_path, index=False)
    logging.info(f"Generated {num_samples} samples (10% fraud) and saved to '{output_path}'")

if __name__ == "__main__":
    main() 