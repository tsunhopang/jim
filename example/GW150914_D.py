import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.prior import CombinePrior, UniformPrior, CosinePrior, SinePrior, PowerLawPrior
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD
from jimgw.transforms import BoundToUnbound
from jimgw.single_event.transforms import ComponentMassesToChirpMassSymmetricMassRatioTransform, SkyFrameToDetectorFrameSkyPositionTransform, ComponentMassesToChirpMassMassRatioTransform, DistanceToSNRWeightedDistanceTransform, GeocentricArrivalTimeToDetectorArrivalTimeTransform, GeocentricArrivalPhaseToDetectorArrivalPhaseTransform
from jimgw.single_event.utils import Mc_q_to_m1_m2
from flowMC.strategy.optimization import optimization_Adam

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
duration = 4
post_trigger_duration = 2
start_pad = duration - post_trigger_duration
end_pad = post_trigger_duration
fmin = 20.0
fmax = 1024.0

ifos = [H1, L1]

for ifo in ifos:
    ifo.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
m_1_prior = UniformPrior(Mc_q_to_m1_m2(M_c_min, q_max)[0], Mc_q_to_m1_m2(M_c_max, q_min)[0], parameter_names=["m_1"])
m_2_prior = UniformPrior(Mc_q_to_m1_m2(M_c_min, q_min)[1], Mc_q_to_m1_m2(M_c_max, q_max)[1], parameter_names=["m_2"])
s1z_prior = UniformPrior(-1.0, 1.0, parameter_names=["s1_z"])
s2z_prior = UniformPrior(-1.0, 1.0, parameter_names=["s2_z"])
dL_prior = PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
iota_prior = SinePrior(parameter_names=["iota"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = CombinePrior(
    [
        m_1_prior,
        m_2_prior,
        s1z_prior,
        s2z_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        iota_prior,
        psi_prior,
        ra_prior,
        dec_prior,
    ]
)

sample_transforms = [
    # all the user reparametrization transform
    ComponentMassesToChirpMassMassRatioTransform(name_mapping=[["m_1", "m_2"], ["M_c", "q"]]),
    DistanceToSNRWeightedDistanceTransform(name_mapping=[["d_L"], ["d_hat_unbounded"]], conditional_names=["M_c","ra", "dec", "psi", "iota"], gps_time=gps, ifos=ifos, dL_min=dL_prior.xmin, dL_max=dL_prior.xmax),
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(name_mapping = [["phase_c"], ["phase_det"]], conditional_names=["ra", "dec", "psi", "iota"], gps_time=gps, ifo=ifos[0]),
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(name_mapping = [["t_c"], ["t_det_unbounded"]], tc_min=t_c_prior.xmin, tc_max=t_c_prior.xmax, conditional_names=["ra", "dec"], gps_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(name_mapping = [["ra", "dec"], ["zenith", "azimuth"]], gps_time=gps, ifos=ifos),
    # all the bound to unbound transform
    BoundToUnbound(name_mapping = [["M_c"], ["M_c_unbounded"]], original_lower_bound=M_c_min, original_upper_bound=M_c_max),
    BoundToUnbound(name_mapping = [["q"], ["q_unbounded"]], original_lower_bound=q_min, original_upper_bound=q_max),
    BoundToUnbound(name_mapping = [["s1_z"], ["s1_z_unbounded"]] , original_lower_bound=-1.0, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = [["s2_z"], ["s2_z_unbounded"]] , original_lower_bound=-1.0, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = [["iota"], ["iota_unbounded"]], original_lower_bound=0., original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["psi"], ["psi_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["zenith"], ["zenith_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["azimuth"], ["azimuth_unbounded"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["phase_det"], ["phase_det_unbounded"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
]

likelihood_transforms = [
    ComponentMassesToChirpMassSymmetricMassRatioTransform(name_mapping=[["m_1", "m_2"], ["M_c", "eta"]]),
]

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=RippleIMRPhenomD(),
    trigger_time=gps,
    duration=4,
    post_trigger_duration=2,
)


mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 3e-3}

Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1)

n_epochs = 30
n_loop_training = 100
learning_rate = 1e-4


jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_loop_training=n_loop_training,
    n_loop_production=20,
    n_local_steps=10,
    n_global_steps=1000,
    n_chains=500,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    n_max_examples=30000,
    n_flow_samples=100000,
    momentum=0.9,
    batch_size=30000,
    use_global=True,
    train_thinning=1,
    output_thinning=10,
    local_sampler_arg=local_sampler_arg,
    strategies=[Adam_optimizer, "default"],
)

jim.sample(jax.random.PRNGKey(42))
jim.get_samples()
jim.print_summary()


###########################################
########## Visualize the Data #############
###########################################
import corner
import matplotlib.pyplot as plt
import numpy as np

production_summary = jim.sampler.get_sampler_state(training=False)
production_chain = production_summary["chains"].reshape(-1, len(jim.parameter_names)).T
if jim.sample_transforms:
    transformed_chain = jim.add_name(production_chain)
    for transform in reversed(jim.sample_transforms):
        transformed_chain = transform.backward(transformed_chain)
result = transformed_chain
labels = list(transformed_chain.keys())

samples = np.array(list(result.values())).reshape(int(len(labels)), -1) # flatten the array
transposed_array = samples.T # transpose the array
figure = corner.corner(transposed_array, labels=labels, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True)
plt.savefig("GW1500914_D_reparam.jpeg")

###########################################
############# Save the Run ################
###########################################
import pickle
pickle.dump(result, open("GW150914_D_reparam.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)