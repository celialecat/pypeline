import numpy as np
from sbi.utils import BoxUniform
from torch.distributions import MultivariateNormal
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.analysis import pairplot
from sbi.inference import NPE, FMPE
from sbi.inference import NLE, NRE_A
# from torch import Tensor
from sbi.neural_nets.estimators.shape_handling import reshape_to_batch_event
from sbi.utils.torchutils import assert_all_finite
from sbi.neural_nets import (
    posterior_nn,
    likelihood_nn,
    classifier_nn,
)
from .tsz_sims import simulator
from .sbi_utils import MixedPrior
import torch


def draw_samples_from_uniform_prior(low = None, high = None, num_sims = 10000):

    prior = BoxUniform(low=low, high=high)
    theta = prior.sample((num_sims,))

    return theta

def draw_samples_from_uniform_prior_LHC(low=None, high=None, num_sims=1000):
    from scipy.stats import qmc
    if low is None or high is None:
        raise ValueError("You must specify both low and high bounds.")

    sampler = qmc.LatinHypercube(d=len(low))
    sample = sampler.random(n=num_sims)
    theta = qmc.scale(sample, low, high)

    return torch.tensor(theta)

def draw_samples_from_constrained_prior(low_cosmo = None, high_cosmo = None, num_sims = 10000):
    
    foreground_data_directory = "/home/lx256/tsz_project/tszpower/data/data_fg-ell-cib_rs_ir_cn-total-planck-collab-15_RC.txt"
    rc_data_directory = "/home/lx256/tsz_project/tszpower/data/data_rc-ell-rc-errrc_backup.txt"
    obs_data_directory = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15_RC.txt"

    D_fg = np.loadtxt(foreground_data_directory)
    D_rc = np.loadtxt(rc_data_directory)
    D_obs = np.loadtxt(obs_data_directory)
    fg_ell = D_fg[:, 0]
    A_CIB_MODEL = D_fg[:, 1][12:]
    A_RS_MODEL  = D_fg[:, 2][12:]
    A_IR_MODEL  = D_fg[:, 3][12:]
    A_CN_MODEL  = D_fg[:, 4][12:]

    Dl_obs = D_obs[:, 1][12:]
    Dl_rc = D_rc[:, 1][12:]

    A_cn_fixed = 0.9033
    samples = []
    prior_cosmo = BoxUniform(low=low_cosmo, high=high_cosmo)
    

    while len(samples)< num_sims:
        theta_rest = prior_cosmo.sample((1,)).squeeze(0).numpy()  # (6,)

        # Foreground amplitudes
        A_cib = np.random.uniform(0, 5)
        A_rs  = np.random.uniform(0, 5)
        A_ir  = np.random.uniform(0, 5)
        A_cn  = A_cn_fixed

        fg_sum = (
            A_cib * A_CIB_MODEL +
            A_rs  * A_RS_MODEL +
            A_ir  * A_IR_MODEL +
            A_cn  * A_CN_MODEL
        )

        if np.all(fg_sum < (Dl_obs - Dl_rc)):
            theta_full = np.concatenate([theta_rest, [A_cib, A_rs, A_ir]])
            samples.append(theta_full)
    samples = np.stack(samples)

    return torch.tensor(samples, dtype=torch.float32)

def draw_samples_from_mixed_prior(prior_specs = None, num_sims = 10000):

    
    mixed_prior = MixedPrior(prior_specs, device='cpu')
    theta = mixed_prior.sample((num_sims,))

    return theta


def draw_simulated_data_from_samples(theta, simulator_func):
    
    x = simulator_func(theta)

    return x


def draw_simulated_data_from_samples_and_noise(key, theta, simulator_func):
    
    x = simulator_func(key, theta)

    return x

def check_prior_consistency(prior, simulator_func):

    processed_prior, num_parameters, prior_returns_numpy = process_prior(prior)
    processed_simulator = process_simulator(simulator_func, processed_prior, prior_returns_numpy)
    check_sbi_inputs(processed_simulator, processed_prior)

    return "Check prior is successful."

# Usage:
# prior = draw_uniform_prior(low=torch.tensor([...]), high=torch.tensor([...]))
# print(check_prior_consistency(prior, tsz_simulator))



def infer_npe(theta, x, prior = None, show_train_summary=True, num_sims = 2000,
            training_batch_size = 100,
            stop_after_epochs = 200, 
            learning_rate = 5e-5,
            num_posterior_samples = 100000, 
            true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):
    density_estimator_build_fun = posterior_nn(model="maf",
                                               hidden_features=50,
                                               num_transforms= 5,
                                               num_components = 10) #TODO: Find this

    inference = NPE(prior=prior, density_estimator = density_estimator_build_fun)
    inference.append_simulations(theta, x).train(show_train_summary=show_train_summary,
                                                 training_batch_size=training_batch_size,
                                                 stop_after_epochs=stop_after_epochs, 
                                                 learning_rate=learning_rate,
                                                 clip_max_norm = 0.5,)
    posterior = inference.build_posterior()


    # TODO: This only works for my own paper. Need to make it more general.
    data_file = np.loadtxt(true_data_path)
    x_o = data_file[:, 1]
    samples = posterior.sample((num_posterior_samples,), x=x_o)

    return samples.numpy()

def infer_fmpe(theta, x, prior = None, show_train_summary=True, num_sims = 2000,
            training_batch_size = 100,
            stop_after_epochs = 200, 
            learning_rate = 5e-5,
            num_posterior_samples = 100000, 
            true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):
    density_estimator_build_fun = posterior_nn(model="nsf",
                                               hidden_features=50,
                                               num_transforms= 5,
                                               num_components = 10) #TODO: Find this

    inference = FMPE(prior=prior, density_estimator = density_estimator_build_fun)
    inference.append_simulations(theta, x).train(show_train_summary=show_train_summary,
                                                 training_batch_size=training_batch_size,
                                                 stop_after_epochs=stop_after_epochs, 
                                                 learning_rate=learning_rate,
                                                 clip_max_norm = 0.5,)
    posterior = inference.build_posterior()


    # TODO: This only works for my own paper. Need to make it more general.
    data_file = np.loadtxt(true_data_path)
    x_o = data_file[:, 1]
    posterior.set_default_x(x_o)
    samples = posterior.sample((num_posterior_samples,), )

    return samples.numpy()



def npe_nn(theta, x, prior=None, show_train_summary=True, num_sims=2000,
           training_batch_size=100, stop_after_epochs=200, 
           learning_rate=5e-5, num_posterior_samples=100000,
           true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):
    # Define the density estimator builder function with the specified architecture.
    density_estimator_build_fun = posterior_nn(model="maf", 
                                               hidden_features=50, 
                                               num_transforms=5)
    
    # Create the inference object with the given prior and density estimator builder.
    inference = NPE(prior=prior, density_estimator=density_estimator_build_fun)
    
    # Append the simulations (parameter-observation pairs) to the inference object.
    inference.append_simulations(theta, x)
    
    # Train the density estimator. Save the estimator separately so that the inference object remains intact.
    density_estimator = inference.train(show_train_summary=show_train_summary,
                                        training_batch_size=training_batch_size,
                                        stop_after_epochs=stop_after_epochs,
                                        learning_rate=learning_rate)
    
    # Return the inference object. You can later call inference.build_posterior(density_estimator)
    # outside of this function if needed.
    return inference

def nle_nn(theta, x, prior=None, show_train_summary=True, num_sims=2000,
           training_batch_size=100, stop_after_epochs=200, 
           learning_rate=5e-5, num_posterior_samples=100000,
           true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):
    # Define the density estimator builder function with the specified architecture.
    density_estimator_build_fun = likelihood_nn(model="maf", hidden_features=50, num_transforms=5)
    
    # Create the inference object with the given prior and density estimator builder.
    inference = NLE(prior=prior, density_estimator=density_estimator_build_fun)
    
    # Append the simulations (parameter-observation pairs) to the inference object.
    inference.append_simulations(theta, x)
    
    # Train the density estimator. Save the estimator separately so that the inference object remains intact.
    density_estimator = inference.train(show_train_summary=show_train_summary,
                                        training_batch_size=training_batch_size,
                                        stop_after_epochs=stop_after_epochs,
                                        learning_rate=learning_rate)
    
    # Return the inference object. You can later call inference.build_posterior(density_estimator)
    # outside of this function if needed.
    return inference




def infer_nle_single_round(theta, x, prior = None, 
                           training_batch_size = 200,
                           show_train_summary=True, 
                           num_sims = 2000, 
                           num_rounds = 1, 
                           stop_after_epochs = 50, 
                           learning_rate = 5e-4,
                           num_posterior_samples = 100000, 
                           true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):
    
    density_estimator_build_fun = likelihood_nn(model="maf",
                                                hidden_features=50,
                                                num_transforms=5, #10
                                                num_components = 5) #TODO: Find this
    inference = NLE(prior=prior, density_estimator = density_estimator_build_fun)
    proposal = prior
    for round_idx in range(1, num_rounds + 1):
        print(f"Processing round {round_idx}")
        # print("Training simulation x shape:", x.shape)

        _ = inference.append_simulations(theta, x).train(show_train_summary=show_train_summary, 
                                                        training_batch_size=training_batch_size,
                                                        stop_after_epochs=stop_after_epochs,
                                                        learning_rate=learning_rate)
                            
        posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                              mcmc_parameters={"num_chains": 20,
                                                               "thin": 5})
        # TODO: This only works for my own paper. Need to make it more general.
        data_file = np.loadtxt(true_data_path)
        x_o = data_file[:, 1]

        # x_o = torch.tensor(x_o).unsqueeze(0)  # Now x_o.shape is [1, 18]
        proposal = posterior.set_default_x(x_o)
    
    # print(proposal)
    samples = proposal.sample((num_posterior_samples,))

    return samples.numpy()

def infer_nle_single_round_bench(theta, x, prior = None, 
                           training_batch_size = 200,
                           show_train_summary=True, 
                           num_sims = 2000, 
                           num_rounds = 1, 
                           stop_after_epochs = 200, 
                           learning_rate = 5e-5,
                           num_posterior_samples = 100000, 
                           true_data_path = "/home/lx256/tsz_project/tszpower/benchmark/data_ell_yell_yerr_benchmark.txt"):
    
    density_estimator_build_fun = likelihood_nn(model="maf",
                                                hidden_features=50,
                                                num_transforms=5, #10
                                                num_components = 5) #TODO: Find this
    inference = NLE(prior=prior, density_estimator = density_estimator_build_fun)
    proposal = prior
    for round_idx in range(1, num_rounds + 1):
        print(f"Processing round {round_idx}")
        # print("Training simulation x shape:", x.shape)

        _ = inference.append_simulations(theta, x).train(show_train_summary=show_train_summary, 
                                                        training_batch_size=training_batch_size,
                                                        stop_after_epochs=stop_after_epochs,
                                                        learning_rate=learning_rate)
                            
        posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                              mcmc_parameters={"num_chains": 20,
                                                               "thin": 5})
        # TODO: This only works for my own paper. Need to make it more general.
        data_file = np.loadtxt(true_data_path)
        x_o = data_file[:, 1]

        # x_o = torch.tensor(x_o).unsqueeze(0)  # Now x_o.shape is [1, 18]
        proposal = posterior.set_default_x(x_o)
    
    # print(proposal)
    samples = proposal.sample((num_posterior_samples,))

    return samples.numpy()


def infer_nle_single_round_ellg100(theta, x, prior = None, 
                           training_batch_size = 200,
                           show_train_summary=True, 
                           num_sims = 2000, 
                           num_rounds = 1, 
                           stop_after_epochs = 200, 
                           learning_rate = 5e-5,
                           num_posterior_samples = 100000, 
                           true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):
    
    density_estimator_build_fun = likelihood_nn(model="maf",
                                                hidden_features=50,
                                                num_transforms=5, #10
                                                num_components = 5) #TODO: Find this
    inference = NLE(prior=prior, density_estimator = density_estimator_build_fun)
    proposal = prior
    for round_idx in range(1, num_rounds + 1):
        print(f"Processing round {round_idx}")
        # print("Training simulation x shape:", x.shape)

        _ = inference.append_simulations(theta, x).train(show_train_summary=show_train_summary, 
                                                        training_batch_size=training_batch_size,
                                                        stop_after_epochs=stop_after_epochs,
                                                        learning_rate=learning_rate)
                            
        posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                              mcmc_parameters={"num_chains": 20,
                                                               "thin": 5})
        # TODO: This only works for my own paper. Need to make it more general.
        data_file = np.loadtxt(true_data_path)
        x_o = data_file[:, 1][-9:]

        # x_o = torch.tensor(x_o).unsqueeze(0)  # Now x_o.shape is [1, 18]
        proposal = posterior.set_default_x(x_o)
    
    # print(proposal)
    samples = proposal.sample((num_posterior_samples,))

    return samples.numpy()

def infer_nle_single_round_bench_ellg100(theta, x, prior = None, 
                           training_batch_size = 200,
                           show_train_summary=True, 
                           num_sims = 2000, 
                           num_rounds = 1, 
                           stop_after_epochs = 200, 
                           learning_rate = 5e-5,
                           num_posterior_samples = 100000, 
                           true_data_path = "/home/lx256/tsz_project/tszpower/benchmark/data_ell_yell_yerr_benchmark.txt"):
    
    density_estimator_build_fun = likelihood_nn(model="maf",
                                                hidden_features=50,
                                                num_transforms=5, #10
                                                num_components = 5) #TODO: Find this
    inference = NLE(prior=prior, density_estimator = density_estimator_build_fun)
    proposal = prior
    for round_idx in range(1, num_rounds + 1):
        print(f"Processing round {round_idx}")
        # print("Training simulation x shape:", x.shape)

        _ = inference.append_simulations(theta, x).train(show_train_summary=show_train_summary, 
                                                        training_batch_size=training_batch_size,
                                                        stop_after_epochs=stop_after_epochs,
                                                        learning_rate=learning_rate)
                            
        posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                              mcmc_parameters={"num_chains": 20,
                                                               "thin": 5})
        # TODO: This only works for my own paper. Need to make it more general.
        data_file = np.loadtxt(true_data_path)
        x_o = data_file[:, 1][-9:]

        # x_o = torch.tensor(x_o).unsqueeze(0)  # Now x_o.shape is [1, 18]
        proposal = posterior.set_default_x(x_o)
    
    # print(proposal)
    samples = proposal.sample((num_posterior_samples,))

    return samples.numpy()

def infer_nle_single_round_bench_ellg400(theta, x, prior = None, 
                           training_batch_size = 200,
                           show_train_summary=True, 
                           num_sims = 2000, 
                           num_rounds = 1, 
                           stop_after_epochs = 200, 
                           learning_rate = 5e-5,
                           num_posterior_samples = 100000, 
                           true_data_path = "/home/lx256/tsz_project/tszpower/benchmark/data_ell_yell_yerr_benchmark.txt"):
    
    density_estimator_build_fun = likelihood_nn(model="maf",
                                                hidden_features=50,
                                                num_transforms=5, #10
                                                num_components = 5) #TODO: Find this
    inference = NLE(prior=prior, density_estimator = density_estimator_build_fun)
    proposal = prior
    for round_idx in range(1, num_rounds + 1):
        print(f"Processing round {round_idx}")
        # print("Training simulation x shape:", x.shape)

        _ = inference.append_simulations(theta, x).train(show_train_summary=show_train_summary, 
                                                        training_batch_size=training_batch_size,
                                                        stop_after_epochs=stop_after_epochs,
                                                        learning_rate=learning_rate)
                            
        posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                              mcmc_parameters={"num_chains": 20,
                                                               "thin": 5})
        # TODO: This only works for my own paper. Need to make it more general.
        data_file = np.loadtxt(true_data_path)
        x_o = data_file[:, 1][-4:]

        # x_o = torch.tensor(x_o).unsqueeze(0)  # Now x_o.shape is [1, 18]
        proposal = posterior.set_default_x(x_o)
    
    # print(proposal)
    samples = proposal.sample((num_posterior_samples,))

    return samples.numpy()


def infer_nle_single_round_ellg400(theta, x, prior = None, 
                           training_batch_size = 200,
                           show_train_summary=True, 
                           num_sims = 2000, 
                           num_rounds = 1, 
                           stop_after_epochs = 200, 
                           learning_rate = 5e-5,
                           num_posterior_samples = 100000, 
                           true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):
    
    density_estimator_build_fun = likelihood_nn(model="maf",
                                                hidden_features=50,
                                                num_transforms=5, #10
                                                num_components = 5) #TODO: Find this
    inference = NLE(prior=prior, density_estimator = density_estimator_build_fun)
    proposal = prior
    for round_idx in range(1, num_rounds + 1):
        print(f"Processing round {round_idx}")
        # print("Training simulation x shape:", x.shape)

        _ = inference.append_simulations(theta, x).train(show_train_summary=show_train_summary, 
                                                        training_batch_size=training_batch_size,
                                                        stop_after_epochs=stop_after_epochs,
                                                        learning_rate=learning_rate)
                            
        posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                              mcmc_parameters={"num_chains": 20,
                                                               "thin": 5})
        # TODO: This only works for my own paper. Need to make it more general.
        data_file = np.loadtxt(true_data_path)
        x_o = data_file[:, 1][-4:]

        # x_o = torch.tensor(x_o).unsqueeze(0)  # Now x_o.shape is [1, 18]
        proposal = posterior.set_default_x(x_o)
    
    # print(proposal)
    samples = proposal.sample((num_posterior_samples,))

    return samples.numpy()

# In NLE including n_rounds, we need to update the proposal after each round.
def infer_nle_likelihood(low = None, high = None, prior = None, show_train_summary=True, 
            simulator_func = None,
            num_sims_round = 200, 
            training_batch_size = 20,
            num_rounds = 5, 
            stop_after_epochs = 200, 
            learning_rate = 5e-5,
            num_posterior_samples = 100000, 
            true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):
    # TODO: This only works for my own paper. Need to make it more general.
    data_file = np.loadtxt(true_data_path)
    x_o = data_file[:, 1]

    density_estimator_build_fun = likelihood_nn(model="maf",hidden_features=50,num_transforms=5) #TODO: Find this
    inference = NLE(prior=prior, density_estimator = density_estimator_build_fun)
    proposal = prior
    for round_idx in range(1, num_rounds + 1):
        print(f"Processing round {round_idx}")
        theta = proposal.sample((num_sims_round,))
        x = simulator_func(theta)

        _ = inference.append_simulations(theta, x).train(show_train_summary=show_train_summary, 
                                                        training_batch_size=training_batch_size,
                                                        stop_after_epochs=stop_after_epochs, 
                                                        learning_rate=learning_rate) 
        
        posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                                mcmc_parameters={"num_chains": 20,
                                                                 "thin": 5})
        proposal = posterior.set_default_x(x_o)

    samples = proposal.sample((num_posterior_samples,))

    return samples.numpy()


# def infer_nle_single_round_validation(theta, x, prior = None, 
#                            training_batch_size = 200,
#                            show_train_summary=True, 
#                            num_sims = 2000, 
#                            num_rounds = 1, 
#                            stop_after_epochs = 200, 
#                            learning_rate = 5e-5,
#                            num_posterior_samples = 100000, 
#                            obs_file_path = "/rds-d6/user/lx256/hpc-work/tsz-project/sbi_lkl_trisp/x_obs_paint_benchmark_paint.pt"):
#     density_estimator_build_fun = likelihood_nn(model="maf",
#                                                 hidden_features=50,
#                                                 num_transforms=5, #10
#                                                 num_components = 5) #TODO: Find this
#     inference = NLE(prior=prior, density_estimator = density_estimator_build_fun)
#     proposal = prior
#     for round_idx in range(1, num_rounds + 1):
#         print(f"Processing round {round_idx}")
#         # print("Training simulation x shape:", x.shape)

#         _ = inference.append_simulations(theta, x).train(show_train_summary=show_train_summary, 
#                                                         training_batch_size=training_batch_size,
#                                                         stop_after_epochs=stop_after_epochs,
#                                                         learning_rate=learning_rate)
                            
#         posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
#                                               mcmc_parameters={"num_chains": 20,
#                                                                "thin": 5})
#         # TODO: This only works for my own paper. Need to make it more general.
#         data_file = np.loadtxt(obs_file_path)
#         x_o = data_file[0]

#         # x_o = torch.tensor(x_o).unsqueeze(0)  # Now x_o.shape is [1, 18]
#         proposal = posterior.set_default_x(x_o)
    
#     # print(proposal)
#     samples = proposal.sample((num_posterior_samples,))

#     return samples.numpy()




# Loss function for NLE
# def nle_loss(neural_net, theta: Tensor, x: Tensor) -> Tensor:
#     """
#     Computes the negative log-likelihood loss for Neural Likelihood Estimation (NLE).

#     Args:
#         neural_net: The density estimator (neural network) used for NLE. It must
#                     have attributes `condition_shape` and `input_shape`, and a method `loss`.
#         theta: Tensor of parameter values.
#         x: Tensor of corresponding simulated data.

#     Returns:
#         A tensor representing the negative log-likelihood loss.
#     """
#     # Reshape theta and x according to the neural network's expected input shapes.
#     theta = reshape_to_batch_event(theta, event_shape=neural_net.condition_shape)
#     x = reshape_to_batch_event(x, event_shape=neural_net.input_shape)

#     # Compute the loss as the negative log probability of x given theta.
#     loss = neural_net.loss(x, condition=theta)

#     # Ensure that the computed loss values are finite.
#     assert_all_finite(loss, "NLE loss")
#     return loss

def infer_nle_loss(theta, x, prior=None, 
                   show_train_summary=True, 
                   num_rounds=1, 
                   training_batch_size=100,
                   num_sims=2000,
                   stop_after_epochs=200, 
                   learning_rate=5e-5):
    """
    Trains an NLE model on the given simulations and returns the final loss value.
    
    Args:
        theta (Tensor): Tensor containing parameter samples.
        x (Tensor): Tensor containing simulation outputs corresponding to theta.
        prior: A prior distribution object with `.log_prob()` and `.sample()` methods.
        show_train_summary (bool): Whether to print the training summary after each round.
        num_rounds (int): Number of rounds to train the NLE model.
        stop_after_epochs (int): Number of epochs after which training stops if no improvement.
        learning_rate (float): Learning rate used for training the neural network.
    
    Returns:
        float: The final average loss (negative log-likelihood) computed on the training data.
    """
    # Initialize the NLE inference object with the given prior.
    density_estimator_build_fun = likelihood_nn(model="maf",hidden_features=100,num_transforms=5) #TODO: Find this
    inference = NLE(prior=prior, density_estimator = density_estimator_build_fun)
    
    # Run training rounds.
    for round_idx in range(1, num_rounds + 1):
        print(f"Processing round {round_idx}")
        inference.append_simulations(theta, x).train(
            show_train_summary=show_train_summary,
            training_batch_size=training_batch_size,
            
            stop_after_epochs=stop_after_epochs,
            learning_rate=learning_rate
        )
    
    # After training, compute the loss on the training data.
    # The internal _loss function reshapes theta and x appropriately and returns the per-sample loss.
    loss_value = inference._loss(theta, x)
    return loss_value

def infer_nle_validation_performance(theta, 
                                     x, 
                                     prior=None, 
                                     show_train_summary=True,
                                    training_batch_size=100,
                                    num_sims=2000,
                                    num_rounds=1,
                                    stop_after_epochs=200, 
                                    learning_rate=5e-5):
    """
    Trains an NLE model on the provided simulations and returns the best validation performance,
    which is computed on the validation set and stored in the training summary.
    
    Args:
        theta (Tensor): Tensor of parameter samples.
        x (Tensor): Tensor of simulation outputs.
        prior: Prior distribution with `.sample()` and `.log_prob()` methods.
        show_train_summary (bool): Whether to print training summaries.
        num_rounds (int): Number of training rounds.
        stop_after_epochs (int): Number of epochs to run before early stopping.
        learning_rate (float): Learning rate for training.
    
    Returns:
        float: The best validation performance (negative log-likelihood) achieved during training.
    """
    density_estimator_build_fun = likelihood_nn(model="maf",hidden_features=100,num_transforms=5) #TODO: Find this
    inference = NLE(prior=prior, density_estimator = density_estimator_build_fun)
    
    # Perform training for the specified number of rounds.
    for round_idx in range(1, num_rounds + 1):
        print(f"Processing round {round_idx}")
        inference.append_simulations(theta, x).train(
            show_train_summary=show_train_summary,
            training_batch_size=training_batch_size,
            stop_after_epochs=stop_after_epochs,
            learning_rate=learning_rate
        )
    
    # Retrieve the best validation performance.
    # This value is stored in the summary dictionary, or alternatively as _best_val_loss.
    best_val_performance = inference._summary["best_validation_loss"][-1]
    # Alternatively, if accessible, you might use:
    # best_val_performance = inference._best_val_loss

    return best_val_performance

def infer_nre_a(theta, x, prior = None, show_train_summary=True, num_sims = 2000,
            training_batch_size = 100,
            stop_after_epochs = 200, 
            learning_rate = 5e-5,
            num_posterior_samples = 100000, 
            true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):
    classifier_build_fun = classifier_nn(
                                model="mlp",
                                z_score_theta = "independent",
                                z_score_x = "independent",
                                hidden_features = 50,)
                                # embedding_net_theta: nn.Module = nn.Identity(),
                                # embedding_net_x: nn.Module = nn.Identity(),) #TODO: Find this

    inference = NRE_A(prior=prior, classifier = classifier_build_fun)
    inference.append_simulations(theta, x).train(show_train_summary=show_train_summary,
                                                 training_batch_size=training_batch_size,
                                                 stop_after_epochs=stop_after_epochs, 
                                                 learning_rate=learning_rate,
                                                 clip_max_norm = 0.5,)
    # posterior = inference.build_posterior()


    # TODO: This only works for my own paper. Need to make it more general.
    data_file = np.loadtxt(true_data_path)
    x_o = data_file[:, 1]
    posterior = inference.build_posterior().set_default_x(x_o)
    samples = posterior.sample((num_posterior_samples,), x=x_o)

    return samples.numpy()




# true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):