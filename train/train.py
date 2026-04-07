import numpy as np
import torch
from datasets import create_dataloader
from model import DeterministicResNetRegression, BasicBlock, BayesianResNetRegression, ResNetRegression
import matplotlib.pyplot as plt
import os
import json
import sys
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torchsummary import summary
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar, minimize
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

def setup_distributed():
    """Initialize distributed training."""
    # Initialize process group
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Fallback for single-node multi-GPU (if not using torchrun)
        rank = 0
        world_size = 1
        local_rank = 0
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

class Tee:
    """Class to duplicate output to both stdout and a file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Force write to file immediately
    
    def flush(self):
        for f in self.files:
            f.flush()

def gaussian_nll_loss(mean, logvar, target):
    """
    Gaussian negative log-likelihood loss for heteroscedastic regression.
    This captures aleatoric uncertainty.
    """
    precision = torch.exp(-logvar)
    loss = 0.5 * (logvar + precision * (target - mean) ** 2)
    return loss.mean()

def sigma_loss(z_pred, z_spec, use_median=True):
    """
    PyTorch loss function version of sigma (NMAD - Normalized Median Absolute Deviation).
    
    Args:
        z_pred: Predicted values (torch.Tensor), can be batched
        z_spec: True/spectroscopic values (torch.Tensor), same shape as z_pred
        use_median: If True, uses median (matches original but limited gradients).
                   If False, uses mean (better gradients for training).
    
    Returns:
        loss: Scalar loss value (torch.Tensor) that can be used for backpropagation
    """
    # Flatten to 1D if needed (handles batches)
    z_pred = z_pred.flatten()
    z_spec = z_spec.flatten()
    
    del_z = z_pred - z_spec
    
    if use_median:
        # Original version using median (matches numpy version)
        # Note: median has limited gradient flow, but PyTorch supports it
        median_del_z = torch.median(del_z)
        normalized_errors = torch.abs((del_z - median_del_z) / (1 + z_spec))
        sigma_nmad = 1.48 * torch.median(normalized_errors)
    else:
        # Mean-based version (better gradients for training)
        mean_del_z = torch.mean(del_z)
        normalized_errors = torch.abs((del_z - mean_del_z) / (1 + z_spec))
        sigma_nmad = 1.48 * torch.mean(normalized_errors)
    
    return sigma_nmad

def train_model(model, train_dataloader, test_dataloader, num_epochs=10, lr=0.001, 
                device=None, use_bayesian=False, save_path=None, 
                weight_decay=1e-4, train_sampler=None, n_mc_val_samples=10):
    
    local_rank = int(os.environ["LOCAL_RANK"])
    
    is_bayesian_model = isinstance(model, BayesianResNetRegression)
    
    if use_bayesian or is_bayesian_model:
        print(f'Using Bayesian model (MC dropout validation with {n_mc_val_samples} samples)')
        use_bayesian = True
    else:
        print('Using deterministic model...')
    
    model = model.to(local_rank)
    # MNF-based BNNs have variational parameters that may not always receive gradients
    # Setting find_unused_parameters=True is necessary for MNF layers
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f'Using AdamW optimizer with weight_decay={weight_decay}')
    
    # Use appropriate loss function
    if use_bayesian:
        criterion = gaussian_nll_loss
    else:
        criterion = torch.nn.MSELoss()
        # criterion = sigma_loss
    
    # Use cosine annealing scheduler with PyTorch's built-in schedulers
    eta_min = lr * 0.01  # Minimum LR is 1% of initial LR
    
    # Cosine annealing scheduler: decrease from base LR to eta_min
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=eta_min
    )
    
    print(f'Using CosineAnnealingLR with {num_epochs} epochs')
    
    training_loss_recorder = []
    testing_loss_recorder = []
    training_sigma_recorder = []
    testing_sigma_recorder = []
    learning_rate_recorder = []
    best_sigma = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        training_loss = 0.0
        training_preds = []
        training_trues = []
        
        for batch_idx, (image, z, ID) in enumerate(tqdm(train_dataloader)):
            image = image.to(local_rank)
            z = z.to(local_rank)
            
            optimizer.zero_grad()
            
            if use_bayesian:
                mean, logvar = model(image)
                loss = criterion(mean.squeeze(), logvar.squeeze(), z.squeeze())
                # For recording, use mean prediction
                output = mean
            else:
                output = model(image)
                loss = criterion(output.squeeze(), z.squeeze())
            
            loss.backward()
            optimizer.step() 
            
            training_loss += loss.item()
            
            training_preds.append(output.detach().cpu().numpy())
            training_trues.append(z.detach().cpu().numpy())
            
        training_preds = np.concatenate(training_preds, axis=0).reshape(-1)
        training_trues = np.concatenate(training_trues, axis=0).reshape(-1)
        
        training_sigma = sigma(training_preds, training_trues)
        training_sigma = training_sigma.item()
        
        training_loss /= len(train_dataloader)
        
        if use_bayesian:
            model.train()  # Keep dropout active for MC sampling during validation
        else:
            model.eval()
        testing_loss = 0.0
        testing_preds = []
        testing_trues = []
        
        for batch_idx, (image, z, ID) in enumerate(tqdm(test_dataloader)):
            image = image.to(local_rank)
            z = z.to(local_rank)
            
            with torch.no_grad():
                if use_bayesian:
                    # MC dropout: average over n_mc_val_samples runs for unbiased sigma
                    batch_mean_samples = []
                    batch_losses = []
                    for _ in range(n_mc_val_samples):
                        mean, logvar = model(image)
                        batch_mean_samples.append(mean.cpu().numpy())
                        batch_losses.append(criterion(mean.squeeze(), logvar.squeeze(), z.squeeze()).item())
                    output = np.mean(np.stack(batch_mean_samples, axis=0), axis=0)
                    testing_loss += np.mean(batch_losses)
                else:
                    output = model(image)
                    loss = criterion(output.squeeze(), z.squeeze())
                    testing_loss += loss.item()
                
            testing_preds.append(output if isinstance(output, np.ndarray) else output.cpu().numpy())
            testing_trues.append(z.cpu().numpy())
            
        testing_preds = np.concatenate(testing_preds, axis=0).reshape(-1)
        testing_trues = np.concatenate(testing_trues, axis=0).reshape(-1)
        
        testing_sigma = sigma(testing_preds, testing_trues)
        testing_sigma = testing_sigma.item()
        
        testing_loss /= len(test_dataloader)
        
        training_loss_recorder.append(training_loss)
        testing_loss_recorder.append(testing_loss)
        training_sigma_recorder.append(training_sigma)
        testing_sigma_recorder.append(testing_sigma)
        
        if testing_sigma < best_sigma:
            print(f'New best sigma: {testing_sigma:.4f} at epoch {epoch+1}')
            best_sigma = testing_sigma
            # Save model state dict (unwrap DDP model if needed)
            model_to_save = model.module if isinstance(model, DDP) else model
            torch.save(model_to_save.state_dict(), 
                       os.path.join(save_path, 'best_model.pth'))
        
        # Step scheduler (cosine annealing)
        cosine_scheduler.step()
        current_lr = cosine_scheduler.get_last_lr()[0]
        learning_rate_recorder.append(current_lr)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {training_loss:.4f}, Test Loss: {testing_loss:.4f}, Train Sigma: {training_sigma:.4f}, Test Sigma: {testing_sigma:.4f}, LR: {current_lr:.6f}')
        
    history = {}
    history['training_loss'] = training_loss_recorder
    history['testing_loss'] = testing_loss_recorder
    history['training_sigma'] = training_sigma_recorder
    history['testing_sigma'] = testing_sigma_recorder
    history['learning_rate'] = learning_rate_recorder
    
    return history

def mc_dropout_predict(model, dataloader, device=None, n_samples=100):
    """
    Monte Carlo Dropout inference for epistemic uncertainty estimation.
    
    Args:
        model: Bayesian model (BayesianResNetRegression or BayesianVisionTransformerRegression)
        dataloader: DataLoader for inference
        device: Device to run inference on
        n_samples: Number of MC samples to draw
    
    Returns:
        mean_preds: Mean predictions across MC samples
        epistemic_uncertainty: Epistemic uncertainty (std across MC samples)
        aleatoric_uncertainty: Aleatoric uncertainty (mean of predicted variances)
        z_true: True values
        source_ids: Source IDs
    """
    model.to(device)
    model.train()  # Keep dropout active for MC sampling
    
    all_mean_samples = []
    all_logvar_samples = []
    z_true_list = []
    source_ids_list = []
    
    print(f'Running MC Dropout inference with {n_samples} samples...')
    
    for batch_idx, (image, z, ID) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        z = z.to(device)
        
        batch_mean_samples = []
        batch_logvar_samples = []
        
        # Collect MC samples
        with torch.no_grad():
            for _ in range(n_samples):
                mean, logvar = model(image)
                batch_mean_samples.append(mean.cpu().numpy())
                batch_logvar_samples.append(logvar.cpu().numpy())
        
        # Stack samples: [n_samples, batch_size, num_params]
        batch_mean_samples = np.stack(batch_mean_samples, axis=0)
        batch_logvar_samples = np.stack(batch_logvar_samples, axis=0)
        
        all_mean_samples.append(batch_mean_samples)
        all_logvar_samples.append(batch_logvar_samples)
        z_true_list.append(z.cpu().numpy())
        source_ids_list.append(ID)
    
    # Concatenate all batches
    all_mean_samples = np.concatenate(all_mean_samples, axis=1)  # [n_samples, total_samples, num_params]
    all_logvar_samples = np.concatenate(all_logvar_samples, axis=1)
    z_true = np.concatenate(z_true_list, axis=0)
    source_ids = np.concatenate(source_ids_list, axis=0)
    
    # Compute statistics across MC samples
    # Epistemic uncertainty: variance across MC samples
    mean_preds = np.mean(all_mean_samples, axis=0)  # [total_samples, num_params]
    epistemic_uncertainty = np.std(all_mean_samples, axis=0)  # [total_samples, num_params]
    
    # Aleatoric uncertainty: mean of predicted variances
    aleatoric_variance = np.mean(np.exp(all_logvar_samples), axis=0)  # [total_samples, num_params]
    aleatoric_uncertainty = np.sqrt(aleatoric_variance)
    
    # Total uncertainty: sqrt(epistemic^2 + aleatoric^2)
    total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
    
    return mean_preds, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, z_true, source_ids


def evaluate_model(model, test_dataloader, device=None, use_bayesian=False, n_mc_samples=100):
    """
    Evaluate model with optional MC Dropout for Bayesian models.
    
    Returns:
        For Bayesian models: z_pred, z_true, source_ids, epistemic_unc, aleatoric_unc, total_unc
        For deterministic models: z_pred, z_true, source_ids, None, None, None
    """
    is_bayesian_model = isinstance(model, BayesianResNetRegression)
    
    if use_bayesian or is_bayesian_model:
        print('Evaluating Bayesian model with MC Dropout...')
        mean_preds, epistemic_unc, aleatoric_unc, total_unc, z_true, source_ids = mc_dropout_predict(
            model, test_dataloader, device, n_samples=n_mc_samples
        )
        return mean_preds, z_true, source_ids, epistemic_unc, aleatoric_unc, total_unc
    else:
        print('Evaluating deterministic model...')
        
        model.to(device)
        model.eval()
        
        z_pred = []
        z_true = []
        source_ids = []
        
        for batch_idx, (image, z, ID) in enumerate(tqdm(test_dataloader)):
            image = image.to(device)
            z = z.to(device)
            
            with torch.no_grad():
                output = model(image)
        
            z_pred.append(output.cpu().numpy())
            z_true.append(z.cpu().numpy())
            source_ids.append(ID)
            
        z_pred = np.concatenate(z_pred, axis=0)
        z_true = np.concatenate(z_true, axis=0)
        source_ids = np.concatenate(source_ids, axis=0)
        
        return z_pred, z_true, source_ids, None, None, None
    
def plot_history(history, save_path=None):
    
    plt.figure(figsize=(12, 4))
    plt.plot(history['training_loss'], label='Training Loss')
    plt.plot(history['testing_loss'], label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(save_path, 'losses.png'))
    plt.close()
    
    plt.figure(figsize=(12, 4))
    plt.plot(history['training_sigma'], label='Training Sigma')
    plt.plot(history['testing_sigma'], label='Testing Sigma')
    plt.xlabel('Epoch')
    plt.ylabel('Sigma')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(save_path, 'sigmas.png'))
    plt.close()
    
    # Plot learning rate if available
    if 'learning_rate' in history:
        plt.figure(figsize=(12, 4))
        plt.plot(history['learning_rate'], label='Learning Rate', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(os.path.join(save_path, 'learning_rate.png'))
        plt.close()
    
def custom_serializer(obj):
    """
    Custom serializer for JSON encoding.
    Supports types such as numpy arrays and numpy floats/ints.
    """
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    # Add further custom handling as needed
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    
def save_history(history, save_path):
    loss = {
        'training_loss': history['training_loss'],
        'testing_loss': history['testing_loss'],
        'training_sigma': history['training_sigma'],
        'testing_sigma': history['testing_sigma'],
    }
    if 'learning_rate' in history:
        loss['learning_rate'] = history['learning_rate']
    with open(os.path.join(save_path, 'history.json'), 'w') as f:
        json.dump(loss, f, indent=4, default=custom_serializer)
    
def sigma(z_pred, z_spec):
    del_z = z_pred - z_spec
    sigma_nmad = 1.48 * \
        np.median(np.abs((del_z - np.median(del_z))/(1 + z_spec)))
    return np.around(sigma_nmad, 5)
    
def plot_results(z_pred, z_true, save_path=None, epistemic_unc=None, aleatoric_unc=None, total_unc=None):
    
    min_value = np.min(z_true)
    max_value = np.max(z_true)
    
    z_pred = z_pred.reshape(-1)
    z_true = z_true.reshape(-1)
    
    sigma_nmad = sigma(z_pred, z_true)
    print(f'Sigma NMAD: {sigma_nmad:.4f}')
    
    plt.figure(figsize=(8, 8))
    plt.scatter(z_true, z_pred, s=1, c='red', alpha=0.5)
    plt.plot([min_value, max_value], [min_value, max_value], 'k--')
    plt.xlabel('True Redshift')
    plt.ylabel('Predicted Redshift')
    plt.grid(True)
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.title(f'Sigma NMAD: {sigma_nmad:.4f}')
    plt.savefig(os.path.join(save_path, 'results.png'))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    _ = plt.hist((z_pred - z_true) / (1 + z_true), bins=100, color='blue', alpha=0.7)
    plt.xlabel('(z_pred - z_true) / (1 + z_true)')
    plt.ylabel('Number of Sources')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'residuals_histogram.png'))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    _ = plt.hist(z_true, bins=100, color='green', alpha=0.7, label='True Redshift', histtype='step')
    _ = plt.hist(z_pred, bins=100, color='red', alpha=0.5, label='Predicted Redshift', histtype='step')
    plt.xlabel('Redshift')
    plt.ylabel('Number of Sources')
    plt.xlim(min_value, max_value)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'compared_distributions.png'))
    plt.close()
    
    # Plot uncertainty if available
    if total_unc is not None:
        total_unc = total_unc.reshape(-1)
        epistemic_unc = epistemic_unc.reshape(-1) if epistemic_unc is not None else None
        aleatoric_unc = aleatoric_unc.reshape(-1) if aleatoric_unc is not None else None
        
        # Plot uncertainty vs error
        errors = np.abs(z_pred - z_true) / (1 + z_true)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(total_unc, errors, s=1, alpha=0.5, c='blue', label='Total Uncertainty')
        plt.xlabel('Total Uncertainty')
        plt.ylabel('Normalized Absolute Error')
        plt.grid(True)
        plt.legend()
        plt.title('Uncertainty vs Prediction Error')
        plt.savefig(os.path.join(save_path, 'uncertainty_vs_error.png'))
        plt.close()
        
        # Plot uncertainty distribution
        plt.figure(figsize=(10, 6))
        if epistemic_unc is not None:
            plt.hist(epistemic_unc, bins=100, alpha=0.5, label='Epistemic Uncertainty', histtype='step')
        if aleatoric_unc is not None:
            plt.hist(aleatoric_unc, bins=100, alpha=0.5, label='Aleatoric Uncertainty', histtype='step')
        plt.hist(total_unc, bins=100, alpha=0.5, label='Total Uncertainty', histtype='step')
        plt.xlabel('Uncertainty')
        plt.ylabel('Number of Sources')
        plt.legend()
        plt.grid(True)
        plt.title('Uncertainty Distribution')
        plt.savefig(os.path.join(save_path, 'uncertainty_distribution.png'))
        plt.close()
        
        # Plot predictions with uncertainty bars
        # Sample a subset for clarity
        n_plot = min(500, len(z_pred))
        indices = np.random.choice(len(z_pred), n_plot, replace=False)
        
        plt.figure(figsize=(10, 8))
        plt.errorbar(z_true[indices], z_pred[indices], 
                    yerr=total_unc[indices], fmt='o', 
                    markersize=2, alpha=0.5, capsize=1, capthick=0.5)
        plt.plot([min_value, max_value], [min_value, max_value], 'k--', linewidth=2)
        plt.xlabel('True Redshift')
        plt.ylabel('Predicted Redshift')
        plt.grid(True)
        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)
        plt.title(f'Predictions with Uncertainty (n={n_plot})')
        plt.savefig(os.path.join(save_path, 'predictions_with_uncertainty.png'))
        plt.close()
        
def plot_results_with_error(results, save_path):
    
    z_pred = results['z_pred'].values
    z_true = results['z_true'].values
    total_uncertainty_calibrated = results['total_uncertainty_calibrated'].values
    
    sigma_nmad = sigma(z_pred, z_true)
    
    min_value = np.min(z_true)
    max_value = np.max(z_true)
    
    plt.figure(figsize=(10, 10))
    plt.errorbar(z_true, z_pred, yerr=total_uncertainty_calibrated, 
                 fmt='.', markersize=2, color='red',
                 ecolor='gray', elinewidth=1, capsize=1)
    plt.xlabel('True Redshift')
    plt.ylabel('Predicted Redshift')
    plt.title(f'Results with Error (Sigma NMAD: {sigma_nmad:.4f})')
    plt.grid(True)
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.savefig(os.path.join(save_path, 'results_with_error_calibrated.png'))
    plt.close()
    
def compute_coverage(mean_pred, uncertainty, z_true, confidence_level):
    """
    Compute the coverage (fraction of true values within prediction interval).
    
    Args:
        mean_pred: Predicted mean values (n_samples,)
        uncertainty: Uncertainty values (n_samples,) - can be std or total uncertainty
        z_true: True values (n_samples,)
        confidence_level: Confidence level (0-1), e.g., 0.9 for 90% interval
    
    Returns:
        coverage: Fraction of samples within the interval
        in_interval: Boolean array indicating which samples are in interval
    """
    # Compute z-score for the confidence level (two-tailed)
    z_score = norm.ppf(0.5 + confidence_level / 2.0)
    
    # Compute prediction interval
    lower_bound = mean_pred - z_score * uncertainty
    upper_bound = mean_pred + z_score * uncertainty
    
    # Check which samples fall within the interval
    in_interval = (z_true >= lower_bound) & (z_true <= upper_bound)
    coverage = np.mean(in_interval)
    
    return coverage, in_interval

def calibrate_uncertainty(mean_pred, uncertainty, z_true, method='isotonic'):
    """
    Calibrate uncertainty estimates using isotonic regression, temperature scaling, or Platt scaling.
    
    Args:
        mean_pred: Predicted mean values (n_samples,)
        uncertainty: Uncalibrated uncertainty values (n_samples,)
        z_true: True values (n_samples,)
        method: Calibration method ('isotonic', 'temperature', or 'platt')
            - isotonic: Non-parametric isotonic regression mapping
            - temperature: Single-parameter scaling (sigma_cal = T * sigma)
            - platt: 2-parameter linear transformation (sigma_cal = A * sigma + B)
    
    Returns:
        calibrated_uncertainty: Calibrated uncertainty values (n_samples,)
        calibration_model: The fitted calibration model (isotonic regressor, temperature, or dict with A,B)
    """
    # Compute normalized errors
    errors = np.abs(mean_pred - z_true)
    
    if method == 'isotonic':
        # Use isotonic regression to map predicted uncertainty to actual error
        # We'll calibrate by learning a mapping: uncertainty -> actual_error
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(uncertainty, errors)
        calibrated_uncertainty = ir.predict(uncertainty)
        return calibrated_uncertainty, ir
    
    elif method == 'temperature':
        # Temperature scaling: scale uncertainty by a learned temperature parameter
        # Find optimal temperature that minimizes calibration error
        
        def calibration_loss(temp):
            scaled_unc = uncertainty * temp
            # Compute coverage for different confidence levels and measure calibration error
            conf_levels = np.linspace(0.1, 0.99, 20)
            errors = []
            for conf in conf_levels:
                coverage, _ = compute_coverage(mean_pred, scaled_unc, z_true, conf)
                errors.append((coverage - conf) ** 2)
            return np.mean(errors)
        
        result = minimize_scalar(calibration_loss, bounds=(0.1, 10.0), method='bounded')
        temperature = result.x
        calibrated_uncertainty = uncertainty * temperature
        return calibrated_uncertainty, temperature
    
    elif method == 'platt':
        # Platt scaling: 2-parameter linear transformation sigma_cal = A * sigma + B
        # Inspired by Platt (1999) and Kuleshov et al. (2018) for regression uncertainty
        # Parameters are learned to minimize calibration error (coverage mismatch)
        
        def calibration_loss(params):
            A, B = params
            # Ensure A > 0 via softplus-like transform: A = exp(log_A) so A > 0
            A_pos = np.exp(params[0])
            scaled_unc = np.maximum(A_pos * uncertainty + B, 1e-6)
            conf_levels = np.linspace(0.1, 0.99, 20)
            errors = []
            for conf in conf_levels:
                coverage, _ = compute_coverage(mean_pred, scaled_unc, z_true, conf)
                errors.append((coverage - conf) ** 2)
            return np.mean(errors)
        
        # Optimize: params = [log_A, B] so A = exp(log_A) > 0
        # log_A in [-2, 2] => A in [0.14, 7.4]; B allows additive correction
        mean_unc = np.mean(uncertainty)
        result = minimize(
            calibration_loss,
            x0=[0.0, 0.0],  # log_A=0 => A=1, B=0 (identity)
            method='L-BFGS-B',
            bounds=[(-2.0, 2.0), (-2.0 * mean_unc, 2.0 * mean_unc)]
        )
        log_A, B = result.x
        A = np.exp(log_A)
        calibrated_uncertainty = np.maximum(A * uncertainty + B, 1e-6)
        platt_params = {'A': float(A), 'B': float(B)}
        return calibrated_uncertainty, platt_params
    
    else:
        raise ValueError(f"Unknown calibration method: {method}. Choose from 'isotonic', 'temperature', 'platt'.")

def plot_confidence_vs_coverage(mean_pred, uncertainty, z_true, 
                                calibrated_uncertainty=None,
                                save_path=None, title_suffix=''):
    """
    Plot confidence vs coverage curve to assess uncertainty calibration.
    
    Args:
        mean_pred: Predicted mean values (n_samples,)
        uncertainty: Uncalibrated uncertainty values (n_samples,)
        z_true: True values (n_samples,)
        calibrated_uncertainty: Optional calibrated uncertainty values (n_samples,)
        save_path: Path to save the plot
        title_suffix: Additional text for plot title
    """
    # Define confidence levels to evaluate
    confidence_levels = np.linspace(0.05, 0.95, 20)
    
    # Compute coverage for uncalibrated uncertainty
    coverages_uncal = []
    for conf in confidence_levels:
        coverage, _ = compute_coverage(mean_pred, uncertainty, z_true, conf)
        coverages_uncal.append(coverage)
    
    # Compute coverage for calibrated uncertainty if provided
    coverages_cal = None
    if calibrated_uncertainty is not None:
        coverages_cal = []
        for conf in confidence_levels:
            coverage, _ = compute_coverage(mean_pred, calibrated_uncertainty, z_true, conf)
            coverages_cal.append(coverage)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot diagonal line (perfect calibration)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    
    # Plot uncalibrated curve
    plt.plot(confidence_levels, coverages_uncal, 'b-o', 
             markersize=6, linewidth=2, label='Uncalibrated', alpha=0.8)
    
    # Plot calibrated curve if available
    if coverages_cal is not None:
        plt.plot(confidence_levels, coverages_cal, 'r-s', 
                 markersize=6, linewidth=2, label='Calibrated', alpha=0.8)
    
    plt.xlabel('Predicted Confidence Level', fontsize=12)
    plt.ylabel('Actual Coverage', fontsize=12)
    plt.title(f'Confidence vs Coverage{title_suffix}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add text with calibration metrics
    # Compute Expected Calibration Error (ECE) for uncalibrated
    ece_uncal = np.mean(np.abs(np.array(coverages_uncal) - confidence_levels))
    textstr = f'ECE (Uncalibrated): {ece_uncal:.4f}'
    
    if coverages_cal is not None:
        ece_cal = np.mean(np.abs(np.array(coverages_cal) - confidence_levels))
        textstr += f'\nECE (Calibrated): {ece_cal:.4f}'
    
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'confidence_vs_coverage.png'), 
                   dpi=300, bbox_inches='tight')
    else:
        plt.savefig('confidence_vs_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return confidence_levels, coverages_uncal, coverages_cal

def perform_uncertainty_calibration(mean_pred, uncertainty, z_true, 
                                    method='isotonic', save_path=None):
    """
    Perform uncertainty calibration and generate calibration plots.
    
    Args:
        mean_pred: Predicted mean values (n_samples,)
        uncertainty: Uncalibrated uncertainty values (n_samples,)
        z_true: True values (n_samples,)
        method: Calibration method ('isotonic', 'temperature', or 'platt')
        save_path: Path to save results and plots
    
    Returns:
        calibrated_uncertainty: Calibrated uncertainty values
        calibration_model: The fitted calibration model
        calibration_metrics: Dictionary with calibration metrics
    """
    print(f"\nPerforming uncertainty calibration using {method} method...")
    
    # Flatten arrays if needed
    mean_pred = mean_pred.reshape(-1)
    uncertainty = uncertainty.reshape(-1)
    z_true = z_true.reshape(-1)
    
    # Calibrate uncertainty
    calibrated_uncertainty, calibration_model = calibrate_uncertainty(
        mean_pred, uncertainty, z_true, method=method
    )
    
    print(f"Calibration complete. Mean uncertainty before: {np.mean(uncertainty):.6f}")
    print(f"Mean uncertainty after: {np.mean(calibrated_uncertainty):.6f}")
    
    # Compute calibration metrics
    confidence_levels = np.linspace(0.05, 0.95, 20)
    
    # Uncalibrated metrics
    coverages_uncal = []
    for conf in confidence_levels:
        coverage, _ = compute_coverage(mean_pred, uncertainty, z_true, conf)
        coverages_uncal.append(coverage)
    ece_uncal = np.mean(np.abs(np.array(coverages_uncal) - confidence_levels))
    
    # Calibrated metrics
    coverages_cal = []
    for conf in confidence_levels:
        coverage, _ = compute_coverage(mean_pred, calibrated_uncertainty, z_true, conf)
        coverages_cal.append(coverage)
    ece_cal = np.mean(np.abs(np.array(coverages_cal) - confidence_levels))
    
    calibration_metrics = {
        'method': method,
        'ece_uncalibrated': float(ece_uncal),
        'ece_calibrated': float(ece_cal),
        'improvement': float(ece_uncal - ece_cal),
        'mean_uncertainty_uncalibrated': float(np.mean(uncertainty)),
        'mean_uncertainty_calibrated': float(np.mean(calibrated_uncertainty))
    }
    if method == 'temperature':
        calibration_metrics['temperature'] = float(calibration_model)
    elif method == 'platt':
        calibration_metrics['platt_A'] = calibration_model['A']
        calibration_metrics['platt_B'] = calibration_model['B']
    
    print(f"Expected Calibration Error (ECE) - Uncalibrated: {ece_uncal:.4f}")
    print(f"Expected Calibration Error (ECE) - Calibrated: {ece_cal:.4f}")
    print(f"Improvement: {ece_uncal - ece_cal:.4f}")
    
    # Generate plots
    if save_path:
        plot_confidence_vs_coverage(
            mean_pred, uncertainty, z_true, 
            calibrated_uncertainty=calibrated_uncertainty,
            save_path=save_path,
            title_suffix=' (Uncertainty Calibration)'
        )
        
        # Save calibration metrics
        with open(os.path.join(save_path, 'calibration_metrics.json'), 'w') as f:
            json.dump(calibration_metrics, f, indent=4)
    
    return calibrated_uncertainty, calibration_model, calibration_metrics

def result_snr_threshold(df, snr_threshold=3.0):
    df_high_snr = df[df['GI_SNR'] >= snr_threshold]
    high_sigma = sigma(df_high_snr['z_pred'].values, df_high_snr['z_true'].values)
    print(f'SNR >= {snr_threshold}, Number of sources: {len(df_high_snr)}, Sigma NMAD: {high_sigma}')
    
    return len(df_high_snr), high_sigma
    
if __name__ == '__main__':
    
    setup_distributed()
    local_rank = int(os.environ['LOCAL_RANK'])
    
    
    lmdb_dir = '../sls/output/lmdb_data'
    dataframe_path = os.path.join(lmdb_dir, 'valid_sources.csv')
    batch_size = 4096
    num_workers = 12
    num_epochs = 100
    lr = 0.001
    weight_decay = 1e-4  # Weight decay for AdamW
    model_type = 'ResNet34'
    train_augment = True
    test_augment = False
    save_path = f'./results_{model_type}'
    n_runs = 100
    n_mc_val_samples = 10
    dropout_rate = 0.1
    use_bayesian = True
    bayesian_type = 'mc_dropout'
    # bayesian_type = 'mnf'
    num_fc_layers = 2
    fix_feature_extractor = False
    if use_bayesian:
        save_path += '_bayesian_' + bayesian_type
    if fix_feature_extractor:
        save_path += '_fix_feature_extractor'
    if train_augment:
        save_path += '_train_augment'
    if test_augment:
        save_path += '_test_augment'
        
    augmentation_types = ['original', 'flip_vertical', 
                               'shift_x', 'shift_y', 
                               'shift_x', 'shift_y', 
                               'shift_x', 'shift_y']
        
    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump({
            'lmdb_dir': lmdb_dir,
            'dataframe_path': dataframe_path,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'num_epochs': num_epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'model_type': model_type, 
            'train_augment': train_augment,
            'test_augment': test_augment, 
            'n_runs': n_runs,
            'n_mc_samples': n_runs,
            'n_mc_val_samples': n_mc_val_samples,
            'use_bayesian': use_bayesian,
            'bayesian_type': bayesian_type,
            'dropout_rate': dropout_rate,
            'num_fc_layers': num_fc_layers,
            'fix_feature_extractor': fix_feature_extractor,
            'augmentation_types': augmentation_types
        }, f, indent=4)
        
    log_file = open(os.path.join(save_path, 'logging.txt'), 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    # train_dataloader, test_dataloader = create_dataloader(
    train_dataset, test_dataset = create_dataloader(
        lmdb_dir=lmdb_dir,
        dataframe_path=dataframe_path,
        batch_size=batch_size,
        num_workers=num_workers,
        train_augment=train_augment,
        test_augment=test_augment, 
        augmentation_types=augmentation_types,
        save_path=save_path
    )
    
    device = torch.device(f"cuda:{local_rank}")
    
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                  sampler=train_sampler, num_workers=num_workers)
    
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 sampler=test_sampler, num_workers=num_workers)
    
    
    if model_type == 'ResNet18':
        if use_bayesian:
            
            base_model = DeterministicResNetRegression(BasicBlock, [2, 2, 2, 2], num_params=1)
            base_model.load_state_dict(
                torch.load('opt/results_ResNet_train_augment/best_model.pth', 
                           weights_only=True, map_location=local_rank),
            )
            
            # Extract layers from input to avgpool (not including flatten) from base_model
            # We will pass a new extractor to BayesianResNetRegression
            feature_extractor = torch.nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                base_model.avgpool
            )
            
            model = BayesianResNetRegression(BasicBlock, [2, 2, 2, 2], num_params=1,
                                             dropout_rate=dropout_rate,
                                             num_fc_layers=num_fc_layers,
                                             fix_feature_extractor=fix_feature_extractor, 
                                             feature_extractor=feature_extractor,
                                             bayesian_type=bayesian_type)
        else:
            model = DeterministicResNetRegression(BasicBlock, [2, 2, 2, 2], num_params=1)
            
    elif model_type == 'ResNet34':
        if use_bayesian:
            base_model = ResNetRegression(BasicBlock, [3, 4, 6, 3], num_params=1)
            base_model.load_state_dict(
                torch.load('results_ResNet34_train_augment/best_model.pth', 
                           weights_only=True, map_location=device),
            )
            feature_extractor = base_model.feature_extractor
            
            model = BayesianResNetRegression(BasicBlock, [3, 4, 6, 3], num_params=1,
                                             dropout_rate=dropout_rate,
                                             num_fc_layers=num_fc_layers,
                                             fix_feature_extractor=fix_feature_extractor, 
                                             feature_extractor=feature_extractor,
                                             bayesian_type=bayesian_type)
        else:
            model = ResNetRegression(BasicBlock, [3, 4, 6, 3], num_params=1, 
                                     dropout_rate=dropout_rate)
    
    orig_stdout = sys.stdout
    
    with open(os.path.join(save_path, 'model_summary.txt'), 'w') as f:
        sys.stdout = f
        summary(model, (2, 40, 480), device='cpu')
        
    sys.stdout = orig_stdout
    
    history = train_model(
        model, 
        train_dataloader, 
        test_dataloader, 
        num_epochs, 
        lr, 
        device, 
        use_bayesian,
        save_path,
        weight_decay=weight_decay, 
        train_sampler=train_sampler,
        n_mc_val_samples=n_mc_val_samples)
    
    plot_history(history, save_path)
    try:
        save_history(history, save_path)
    except Exception as e:
        print(f'Error saving history: {e}')
    
    # Unwrap DDP and load best model for evaluation
    model_to_eval = model.module if isinstance(model, DDP) else model
    model_to_eval.load_state_dict(
        torch.load(os.path.join(save_path, 'best_model.pth'), 
                   weights_only=True, map_location=device))
    
    # Evaluate with MC Dropout if Bayesian
    z_pred, z_true, source_ids, epistemic_unc, aleatoric_unc, total_unc = evaluate_model(
        model_to_eval, test_dataloader, device, use_bayesian, n_mc_samples=n_runs)
    
    z_pred = z_pred.reshape(-1)
    z_true = z_true.reshape(-1)
    source_ids = source_ids.reshape(-1)
    
    # Create results dataframe
    df_dict = {
        'source_id': source_ids,
        'z_pred': z_pred,
        'z_true': z_true
    }
    
    if total_unc is not None:
        epistemic_unc = epistemic_unc.reshape(-1)
        aleatoric_unc = aleatoric_unc.reshape(-1)
        total_unc = total_unc.reshape(-1)
        df_dict['epistemic_uncertainty'] = epistemic_unc
        df_dict['aleatoric_uncertainty'] = aleatoric_unc
        df_dict['total_uncertainty'] = total_unc
        print(f'Mean Epistemic Uncertainty: {np.mean(epistemic_unc):.6f}')
        print(f'Mean Aleatoric Uncertainty: {np.mean(aleatoric_unc):.6f}')
        print(f'Mean Total Uncertainty: {np.mean(total_unc):.6f}')
    
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(save_path, 'results.csv'), index=False)
    
    plot_results(z_pred, z_true, save_path, epistemic_unc, aleatoric_unc, total_unc)
    
    # Perform uncertainty calibration if Bayesian model
    if total_unc is not None:
        print("\n" + "="*60)
        print("UNCERTAINTY CALIBRATION")
        print("="*60)
        
        # Calibrate total uncertainty
        calibrated_total_unc, cal_model_total, cal_metrics_total = perform_uncertainty_calibration(
            z_pred, total_unc, z_true, method='temperature', save_path=save_path
        )
        
        # Add calibrated uncertainty to dataframe
        df['total_uncertainty_calibrated'] = calibrated_total_unc
        df.to_csv(os.path.join(save_path, 'results.csv'), index=False)
        
        # Optionally calibrate epistemic and aleatoric separately (without plots)
        if epistemic_unc is not None:
            calibrated_epistemic_unc, _ = calibrate_uncertainty(
                z_pred, epistemic_unc, z_true, method='temperature'
            )
            df['epistemic_uncertainty_calibrated'] = calibrated_epistemic_unc
        
        if aleatoric_unc is not None:
            calibrated_aleatoric_unc, _ = calibrate_uncertainty(
                z_pred, aleatoric_unc, z_true, method='temperature'
            )
            df['aleatoric_uncertainty_calibrated'] = calibrated_aleatoric_unc
        
        df.to_csv(os.path.join(save_path, 'results.csv'), index=False)
        print("="*60 + "\n")
        
        
        try:
            plot_results_with_error(df, save_path)
        except Exception as e:
            print(f'Error plotting results with error: {e}')
    
    dataframe_ = pd.read_csv(dataframe_path)
    
    df_all = pd.merge(
        df, 
        dataframe_,
        left_on='source_id',
        right_on='ID',
        how='left',
    )
    
    for snr_thresh in [3.0, 5.0, 10.0]:
        result_snr_threshold(df_all, snr_threshold=snr_thresh)
    
    dist.destroy_process_group()
    # log_file.close()
