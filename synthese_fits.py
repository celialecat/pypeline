import numpy as np
import matplotlib.pyplot as plt
import torch
import time as time
from astropy.io import fits
import ST  # Make sure ST.py is in the same directory

def image_synthesis(image, J, L, num_pixel,
                    learnable_param_list = [(100, 1e-3)],
                    savedir = '',
                    device='cpu',
                    coef = 'ST',
                    random_seed = 987,
                    low_bound = -0.010):
    """
    Image synthesis function based on scattering transform coefficients.
    Adapted from ST_image_synthesis.ipynb.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    image = image[:num_pixel, :num_pixel]
    image_torch = torch.from_numpy(image).type(torch.FloatTensor) + 5
    if device=='cuda':
        image_torch = image_torch.cuda()
    
    # Initialization of calculators (must be defined outside the function)
    # We assume `bispectrum_calculator` and `ST_calculator` are available
    # in the global scope or passed as arguments if necessary.
    
    target_bi = bispectrum_calculator.forward(image_torch)
    target_ST = ST_calculator.forward(image_torch[None,:,:], J, L)[0][0]
    target_ST = target_ST[target_ST!=0].log()
    target_PS, _ = ST.get_power_spectrum(image_torch, 20, device)
    target_PS = target_PS.log()

#-------------------------------------------------------------------------------
    # define mock image
    class model_image(torch.nn.Module):
        def __init__(self):
            super(model_image, self).__init__()

            # initialize with GRF of same PS as target image
            image_to_train = torch.from_numpy(
                ST.get_random_data(image, num_pixel, num_pixel, "image").reshape(1,-1)*1
            ).type(torch.FloatTensor) + 5 + 0.000
            if device=='cuda':
                image_to_train = image_to_train.cuda()
            self.param = torch.nn.Parameter( image_to_train )
#-------------------------------------------------------------------------------
    model_fit = model_image()

    # define learnable
    for learnable_group in range(len(learnable_param_list)):
        num_step = learnable_param_list[learnable_group][0]
        learning_rate = learnable_param_list[learnable_group][1]
        
        optimizer = torch.optim.Adamax(model_fit.parameters(), lr=learning_rate)

        # optimize
        for i in range(int(num_step)):
            # loss: power spectrum
            PS, _ = ST.get_power_spectrum(model_fit.param.reshape(num_pixel,num_pixel), 20, device)
            PS = PS.log()
            loss_PS = ((target_PS - PS)**2).sum()
            # loss: L1
            target_L1 = (image_torch-5).abs().mean() 
            loss_L1 = (
                ((model_fit.param-5).abs().mean() - target_L1) / target_L1 
            )**2*1e6
            # loss: mean
            loss_mean = (
                model_fit.param.reshape(1,num_pixel,num_pixel).mean() - 
                image_torch.mean()
            )**2*1e7
            # loss: bound
            loss_bound = torch.exp(
                (5 + low_bound - model_fit.param.reshape(1,num_pixel,num_pixel))/0.003
            ).mean()

            if coef=='bi':
                bi = bispectrum_calculator.forward(
                    model_fit.param.reshape(num_pixel,num_pixel)
                )
                loss_bi = ((target_bi - bi)**2).sum()
                loss = loss_bi + loss_bound + loss_mean + loss_L1

            if coef=='bi+P':
                bi = bispectrum_calculator.forward(
                    model_fit.param.reshape(num_pixel,num_pixel)
                )
                loss_bi = ((target_bi - bi)**2).sum()
                loss = loss_bi + loss_bound + loss_mean + loss_L1 + loss_PS
                
            elif coef=='ST':
                ST_coef = ST_calculator.forward(
                    model_fit.param.reshape(1,num_pixel,num_pixel), J, L,
                )[0][0]
                ST_coef = ST_coef[ST_coef!=0].log()
                loss_ST = ((target_ST - ST_coef)**2).sum()*1000
                loss = loss_ST + loss_bound + loss_mean + loss_L1
            else:
                loss = loss_bound + loss_mean + loss_L1
                
            if i%100== 0:
                print(f"Step {i}")
                print(f"  loss: {loss.item():.4f}")
                print(f"  loss_mean: {loss_mean.item():.4f}")
                print(f"  loss_bound: {loss_bound.item():.4f}")
                print(f"  loss_L1: {loss_L1.item():.4f}")
                if coef=='bi' or coef=='bi+P':
                    print(f"  loss_bi: {loss_bi.item():.4f}")
                if coef=='bi+P':
                    print(f"  loss_PS: {loss_PS.item():.4f}")
                if coef=='ST':
                    print(f"  loss_ST: {loss_ST.item():.4f}")

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

    return model_fit.param.reshape(1,num_pixel,num_pixel).cpu().detach().numpy()-5

def load_fits_image(path):
    """Loads an image from a FITS file."""
    with fits.open(path) as hdul:
        # Assumes the image is in the first HDU (Header Data Unit)
        image_data = hdul[0].data
    # Convert to float32 for PyTorch compatibility
    return image_data.astype(np.float32)

# --- Main entry point ---
if __name__ == '__main__':
    # --- Parameters ---
    FITS_FILE_PATH = 'your_image.fits'  
    SAVE_DIR = '.'  # Directory to save results
    
    # Image size (must be a power of 2, e.g., 256, 512)
    M = 256 
    N = M

    # Scattering transform parameters
    J = 7
    L = 8
    
    # Choose CPU or GPU
    # 'cuda' if you have an NVIDIA GPU with CUDA configured, otherwise 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")
    
    # Coefficient type for the loss function
    # 'ST' (recommended), 'bi' (bispectrum), 'bi+P' (bispectrum + power spectrum)
    coef = 'ST'
    
    # Random seed for reproducibility
    random_seed = 987

    # --- 1. Load the target image ---
    print(f"Loading image from: {FITS_FILE_PATH}")
    try:
        target_image = load_fits_image(FITS_FILE_PATH)
    except FileNotFoundError:
        print(f"ERROR: File '{FITS_FILE_PATH}' not found.")
        print("Please create a test FITS image named 'your_image.fits' or modify the FITS_FILE_PATH variable.")
        # Creating a test image if the file does not exist
        print("Creating a test image...")
        target_image = np.random.rand(M, N).astype(np.float32)
        fits.writeto(FITS_FILE_PATH, target_image, overwrite=True)
        print(f"Test image '{FITS_FILE_PATH}' created.")

    # Resize if necessary
    target_image = target_image[:M, :N]
    
    # --- 2. Generate filters and calculators ---
    print("Generating scattering transform filters...")
    filters_set = ST.FiltersSet(M=M, N=N, J=J, L=L).generate_morlet(
        if_save=False, save_dir=None, precision='single'
    )
    
    print("Creating calculators (ST and Bispectrum)...")
    ST_calculator = ST.ST_2D(filters_set, J, L, device=device)
    
    bin_edges = np.linspace(150/(360/3.5), M/2*1.4, 7)
    bispectrum_calculator = ST.Bispectrum_Calculator(bin_edges, M, N, device=device)

    # --- 3. Start synthesis ---
    print("Starting image synthesis...")
    start_time = time.time()
    
    synthesised_image = image_synthesis(
        target_image, J=J, L=L, num_pixel=M,
        learnable_param_list=[
            (400*2, 5e-3), (400*2, 1e-3), (400*1, 1e-3),
            (200*1, 5e-4), (200*1, 2e-4), (200*1, 1e-4),
        ],
        savedir=SAVE_DIR,
        device=device,
        coef=coef,
        random_seed=random_seed,
        low_bound=-0.015,
    )
    
    print(f"Synthesis finished in {time.time() - start_time:.2f} seconds.")
    
    # --- 4. Save and display results ---
    synthesised_image = synthesised_image[0]
    
    # Save the synthesized image in FITS and Numpy formats
    final_fits_path = f"{SAVE_DIR}/synthesised_image.fits"
    final_npy_path = f"{SAVE_DIR}/synthesised_image.npy"
    print(f"Saving synthesized image to '{final_fits_path}'")
    fits.writeto(final_fits_path, synthesised_image, overwrite=True)
    np.save(final_npy_path, synthesised_image)
    
    # Display
    plt.figure(figsize=(18, 6), dpi=100)
    
    vmin = np.percentile(target_image, 1)
    vmax = np.percentile(target_image, 99)
    
    # Target image
    plt.subplot(1, 3, 1)
    plt.imshow(target_image, vmin=vmin, vmax=vmax, cmap='viridis')
    plt.title('Target Image')
    plt.xticks([])
    plt.yticks([])

    # Synthesized image
    plt.subplot(1, 3, 2)
    plt.imshow(synthesised_image, vmin=vmin, vmax=vmax, cmap='viridis')
    plt.title('Synthesized Image')
    plt.xticks([])
    plt.yticks([])

    # Histogram
    plt.subplot(1, 3, 3)
    y_target, x_target = np.histogram(target_image.ravel(), bins=100, density=True)
    y_synth, x_synth = np.histogram(synthesised_image.ravel(), bins=100, density=True)
    plt.plot((x_target[1:] + x_target[:-1]) / 2, y_target, label='Target')
    plt.plot((x_synth[1:] + x_synth[:-1]) / 2, y_synth, label='Synthesized')
    plt.legend()
    plt.title('Pixel Distribution')
    plt.yscale('log')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.show()

    print("\nImage Statistics:")
    print(f"  - Target:       mean={target_image.mean():.4f}, std_dev={target_image.std():.4f}")
    print(f"  - Synthesized:  mean={synthesised_image.mean():.4f}, std_dev={synthesised_image.std():.4f}")