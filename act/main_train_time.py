import jax.numpy as jnp
import jax
import os
import os.path as osp
from datetime import datetime
from flax import linen as nn
from ppo import PPO, PPO_Network
from cartpole import JaxCartPole
from reacher import Reacher
from pendulum import Pendulum
from cartpole_goal import JaxCartPoleGoal
from reacher_goal import ReacherGoal
from pendulum_goal import PendulumGoal
from evo_utils import MfosWrapper, VecNormalizer
from gymnax_utils import GymnaxWrapper
from evosax import OpenES, SimpleGA, CMA_ES, ParameterReshaper, FitnessShaper, NetworkMapper
from evosax.utils import ESLog
import gymnax
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--rew_type", type=str, required=True)
parser.add_argument("--n_dims", type=int, default=2)
parser.add_argument("--scale", type=float, default=2 * jnp.pi)
parser.add_argument("--num_rollouts", type=int, default=1)
parser.add_argument("--outer_algo", type=str, default="OPEN_ES")
parser.add_argument("--pmap", action="store_true")
parser.add_argument("--log", action="store_true")
parser.add_argument("--gpu_mult", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--centered_rank", action="store_true")
parser.add_argument("--static", action="store_true")
parser.add_argument("--group-name", type=str, default="")
parser.add_argument("--project", type=str, default="")
parser.add_argument("--normalize-obs", action="store_true")
parser.add_argument("--end-only", action="store_true")
parser.add_argument("--include-actions", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    """
    INIT INNER AGENT THINGS
    """
    if args.env == "CARTPOLE":
        config = {
            "DISCRETE": True,
            "MAX_GRAD_NORM": 0.5,
            "NUM_ENVS": 4,
            "NUM_ACTIONS": 2,
            "NUM_UPDATES": 32,
            "UPDATE_PERIOD": 256,
            "GAMMA": 0.99,
            "NUM_EPOCHS_PER_UPDATE": 16,
            "CLIP_EPS": 0.2,
            "GAE_LAMBDA": 0.95,
            "CRITIC_COEFF": 0.5,
            "ENTROPY_COEFF": 0.01,
            "LR": 0.005,
            "OBS_SIZE": 4 + args.n_dims,
            "DEBUG": False,
            "POP_SIZE": 1024,
        }
        if args.rew_type == "GEN":
            inner_env = JaxCartPoleGoal()
            config["ENV_OBS_SPACE"] = 5  # TODO: PUT THIS IN THE ENV
        else:
            inner_env = JaxCartPole()
            config["ENV_OBS_SPACE"] = 4
        network = PPO_Network(2, hsize=32, discrete=config["DISCRETE"], activation_fn=nn.tanh)

    elif args.env == "REACHER":
        config = {
            "OBS_SIZE": 10 + args.n_dims,
            "MAX_GRAD_NORM": 0.5,
            "NUM_ENVS": 32,
            "NUM_ACTIONS": 2,
            "NUM_UPDATES": 256,
            "UPDATE_PERIOD": 128,
            "GAMMA": 0.99,
            "NUM_EPOCHS_PER_UPDATE": 10,
            "CLIP_EPS": 0.2,
            "GAE_LAMBDA": 0.95,
            "CRITIC_COEFF": 0.5,
            "ENTROPY_COEFF": 0.0005,
            "LR": 0.004,
            "DEBUG": False,
            "DISCRETE": False,
            "POP_SIZE": 128,
        }
        if args.rew_type == "GEN":
            inner_env = ReacherGoal()
            config["ENV_OBS_SPACE"] = 12
        else:
            inner_env = Reacher()
            config["ENV_OBS_SPACE"] = 10
        network = PPO_Network(2, discrete=config["DISCRETE"], hsize=128, activation_fn=nn.relu)

    elif args.env == "PENDULUM":
        config = {
            "OBS_SIZE": 3 + args.n_dims,
            "ACTION_SIZE": 1,
            "MAX_GRAD_NORM": 0.5,
            "NUM_ENVS": 16,
            "NUM_ACTIONS": 1,
            "NUM_UPDATES": 128,
            "UPDATE_PERIOD": 256,
            "GAMMA": 0.95,
            "NUM_EPOCHS_PER_UPDATE": 16,
            "CLIP_EPS": 0.2,
            "GAE_LAMBDA": 0.95,
            "CRITIC_COEFF": 0.5,
            "ENTROPY_COEFF": 0.005,
            "LR": 0.02,
            "DEBUG": False,
            "DISCRETE": False,
            "POP_SIZE": 512 + 256,
        }
        if args.rew_type == "GEN":
            inner_env = PendulumGoal()
            config["ENV_OBS_SPACE"] = 5
        else:
            inner_env = Pendulum()
            config["ENV_OBS_SPACE"] = 3

        network = PPO_Network(1, discrete=config["DISCRETE"], hsize=32, activation_fn=nn.tanh)

    elif args.env == "BREAKOUT":
        config = {
            "OBS_SIZE": 400 + args.n_dims,
            "MAX_GRAD_NORM": 0.5,
            "NUM_ENVS": 64,
            "NUM_ACTIONS": 3,
            "NUM_UPDATES": 1024,
            "UPDATE_PERIOD": 128,
            "GAMMA": 0.99,
            "NUM_EPOCHS_PER_UPDATE": 32,
            "CLIP_EPS": 0.2,
            "GAE_LAMBDA": 0.95,
            "CRITIC_COEFF": 0.5,
            "ENTROPY_COEFF": 0.01,
            "LR": 3e-4,
            "DEBUG": False,
            "DISCRETE": True,
            "POP_SIZE": 128,
        }

        if args.rew_type == "GEN":
            raise NotImplementedError
        else:
            env, env_params = gymnax.make("Breakout-MinAtar")
            inner_env = GymnaxWrapper(env, env_params)
            config["ENV_OBS_SPACE"] = 400
        network = PPO_Network(3, discrete=config["DISCRETE"], hsize=256, activation_fn=nn.relu)
    else:
        raise NotImplementedError

    config["PUPPET"] = False
    config["REW_TYPE"] = args.rew_type
    config["ENV"] = args.env
    config["OUTER_ALGO"] = args.outer_algo
    config["STATIC"] = args.static
    config["CT_SCALE"] = args.scale
    config["GEN_EXPLOIT"] = args.rew_type == "GEN"
    config["N_DEVICES"] = jax.local_device_count()
    config["CENTERED_RANK"] = args.centered_rank
    config["NORMALIZE"] = args.normalize_obs
    config["END_ONLY"] = args.end_only
    config["INCLUDE_ACTIONS"] = args.include_actions

    """
    INIT MFOS AGENT THIGNS
    """
    rng = jax.random.PRNGKey(args.seed)
    if args.include_actions:
        pholder = jnp.zeros((config["ENV_OBS_SPACE"] + config["NUM_ACTIONS"],))
    else:
        pholder = jnp.zeros((config["ENV_OBS_SPACE"],))
    if args.static:
        mfos_network = NetworkMapper["MLP"](
            num_hidden_units=64,
            num_hidden_layers=2,
            num_output_units=args.n_dims,
            hidden_activation="relu",
            output_activation="tanh",
        )
        params = mfos_network.init(
            rng,
            x=pholder,
            rng=rng,
        )
    else:
        mfos_network = NetworkMapper["LSTM"](
            num_hidden_units=32,
            num_output_units=args.n_dims,
            output_activation="tanh",
        )
        carry_init = mfos_network.initialize_carry()
        params = mfos_network.init(
            rng,
            x=pholder,
            carry=carry_init,
            rng=rng,
        )

    param_reshaper = ParameterReshaper(params["params"])

    def single_rollout(rng_input, mfos_params):
        if config["STATIC"]:
            env = MfosWrapper(inner_env, mfos_network.apply, mfos_params, config)
        else:
            mfos_init = mfos_network.initialize_carry()
            env = MfosWrapper(inner_env, mfos_network.apply, mfos_params, config, mfos_init)

        if config["NORMALIZE"]:
            env = VecNormalizer(env, config)  # TODO: MAKE IT PART OF TRAIN STATE INSTEAD OF AN ENV WRAPPER SO CAN SAVE?
            agent = PPO(
                network,
                # env,
                env.reset,
                env.step,
                config,
            )
        else:
            agent = PPO(
                network,
                # env,
                jax.vmap(env.reset),
                jax.vmap(env.step),
                config,
            )
        val, metric = agent.train(rng_input)
        print(metric.shape)
        if args.rew_type == "PRO":
            if args.end_only:
                return metric[-1]
            else:
                return metric.mean()
        elif args.rew_type == "ANTI":
            if args.end_only:
                return -metric[-1]
            else:
                return -metric.mean()
        elif args.rew_type == "GEN":
            if args.end_only:
                return -metric[-1]
            else:
                return metric[1].mean()
        else:
            raise NotImplementedError

    vmap_rollout = jax.vmap(single_rollout, in_axes=(0, None))
    if args.pmap:
        rollout = jax.pmap(jax.jit(jax.vmap(vmap_rollout, in_axes=(None, param_reshaper.vmap_dict))))
    else:
        rollout = jax.jit(jax.vmap(vmap_rollout, in_axes=(None, param_reshaper.vmap_dict)))

    popsize = int(config["POP_SIZE"] * args.gpu_mult)
    if args.outer_algo == "OPEN_ES":
        strategy = OpenES(popsize=popsize, num_dims=param_reshaper.total_params, opt_name="adam")
    elif args.outer_algo == "SIMPLE_GA":
        strategy = SimpleGA(popsize=popsize, num_dims=param_reshaper.total_params, elite_ratio=0.5)
    elif args.outer_algo == "CMA_ES":
        strategy = CMA_ES(popsize=popsize, num_dims=param_reshaper.total_params, elite_ratio=0.5)
    else:
        raise NotImplementedError

    es_params = strategy.default_params

    num_generations = 2049
    num_rollouts = args.num_rollouts
    save_every_k_gens = 4

    es_logging = ESLog(param_reshaper.total_params, num_generations, top_k=5, maximize=True)
    log = es_logging.initialize()

    fit_shaper = FitnessShaper(centered_rank=args.centered_rank, z_score=True, w_decay=0.0, maximize=True)

    state = strategy.initialize(rng, es_params)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_dir = f"{dir_path}/data/{str(datetime.now()).replace(' ', '_')}_{args.env}_{args.outer_algo}_{args.n_dims}_{args.rew_type}"
    os.mkdir(save_dir)
    config["SAVE_DIR"] = save_dir
    if args.log:
        wandb.init(project=args.project, config=config, group=args.group_name)

    for gen in range(num_generations):
        rng, rng_init, rng_ask, rng_eval = jax.random.split(rng, 4)
        x, state = strategy.ask(rng_ask, state, es_params)
        reshaped_params = param_reshaper.reshape(x)
        batch_rng = jax.random.split(rng_eval, num_rollouts)
        if args.pmap:
            batch_rng_pmap = jnp.tile(batch_rng, (jax.local_device_count(), 1, 1))
            fitness = rollout(batch_rng_pmap, reshaped_params).reshape(-1, num_rollouts).mean(axis=1)
        else:
            fitness = rollout(batch_rng, reshaped_params).mean(axis=1)
        fit_re = fit_shaper.apply(x, fitness)
        state = strategy.tell(x, fit_re, state, es_params)
        log = es_logging.update(log, x, fitness)

        print(f"Generation: {gen}, Best: {log['log_top_1'][gen]}, Fitness: {fitness.mean()}")
        if args.log:
            wandb.log(
                {
                    "Best Training Score": log["log_top_1"][gen],
                    "Fitness": fitness.mean(),
                }
            )
        if gen % save_every_k_gens == 0:
            if not args.pmap:
                top_params = param_reshaper.reshape_single(log["top_params"][0])
                jnp.save(osp.join(save_dir, f"top_param_{gen}.npy"), top_params)
            if args.outer_algo == "OPEN_ES":
                jnp.save(osp.join(save_dir, f"curr_param_{gen}.npy"), state['mean'])
