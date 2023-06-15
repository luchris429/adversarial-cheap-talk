import jax.numpy as jnp
import jax
import os
import os.path as osp
from datetime import datetime
from flax import linen as nn
from ppo import PPO, PPO_Network
from cartpole import JaxCartPole
from cartpole_goal import JaxCartPoleGoal
from reacher import Reacher
from reacher_goal import ReacherGoal
from pendulum import Pendulum
from pendulum_goal import PendulumGoal
from evo_utils import MfosWrapper, VecNormalizer, StaticVecNormalizer
from flax.core.frozen_dict import FrozenDict, unfreeze
from evosax import OpenES, SimpleGA, CMA_ES, ParameterReshaper, FitnessShaper, NetworkMapper
from evosax.utils import ESLog
from gymnax_utils import GymnaxWrapper, GymnaxGoalWrapper
import gymnax
import functools
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True)
parser.add_argument("--n_dims", type=int, default=2)
parser.add_argument("--scale", type=int, default=jnp.pi)
parser.add_argument("--outer_algo", type=str, default="OPEN_ES")
parser.add_argument("--gpu_mult", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log", action="store_true")
parser.add_argument("--pmap", action="store_true")
parser.add_argument("--centered_rank", action="store_true")
parser.add_argument("--group-name", type=str, default="")
parser.add_argument("--project", type=str, default="")
parser.add_argument("--static", action="store_true")
parser.add_argument("--zero-shot", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--normalize-obs", action="store_true")
parser.add_argument("--include-actions", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    if args.debug:
        from jax.config import config as jax_config

        jax_config.update("jax_disable_jit", True)

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
        inner_env = JaxCartPole()
        network = PPO_Network(2, discrete=config["DISCRETE"], hsize=32, activation_fn=nn.tanh)

        exploiter_config = {
            "DISCRETE": False,
            "MAX_GRAD_NORM": 0.5,
            "NUM_ENVS": 4,
            "NUM_ACTIONS": args.n_dims,
            "NUM_UPDATES": 64,
            "UPDATE_PERIOD": 256,
            "GAMMA": 0.99,
            "NUM_EPOCHS_PER_UPDATE": 16,
            "CLIP_EPS": 0.2,
            "GAE_LAMBDA": 0.95,
            "CRITIC_COEFF": 0.5,
            "ENTROPY_COEFF": 0.005,
            "LR": 0.02,
            "OBS_SIZE": 5,
            "DEBUG": False,
        }
        exploiter_env_fn = JaxCartPoleGoal
        exploiter_net = PPO_Network(args.n_dims, discrete=exploiter_config["DISCRETE"], hsize=32, activation_fn=nn.tanh)

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
        inner_env = Pendulum()
        network = PPO_Network(1, discrete=config["DISCRETE"], hsize=32, activation_fn=nn.tanh)

        exploiter_config = {
            "OBS_SIZE": 3 + 2,
            "ACTION_SIZE": 1,
            "MAX_GRAD_NORM": 0.5,
            "NUM_ENVS": 16,
            "NUM_ACTIONS": args.n_dims,
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
        exploiter_env_fn = PendulumGoal
        exploiter_net = PPO_Network(args.n_dims, discrete=config["DISCRETE"], hsize=32, activation_fn=nn.tanh)

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
        inner_env = Reacher()
        network = PPO_Network(2, discrete=config["DISCRETE"], hsize=128, activation_fn=nn.relu)

        exploiter_config = {
            "OBS_SIZE": 10 + 2,
            "MAX_GRAD_NORM": 0.5,
            "NUM_ENVS": 32,
            "NUM_ACTIONS": args.n_dims,
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
        exploiter_env_fn = ReacherGoal
        exploiter_net = PPO_Network(args.n_dims, discrete=config["DISCRETE"], hsize=128, activation_fn=nn.relu)

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
        env, env_params = gymnax.make("Breakout-MinAtar")
        inner_env = GymnaxWrapper(env, env_params)
        config["ENV_OBS_SPACE"] = 400
        network = PPO_Network(3, discrete=config["DISCRETE"], hsize=256, activation_fn=nn.relu)

        exploiter_config = {
            "OBS_SIZE": 400 + 2,
            "MAX_GRAD_NORM": 0.5,
            "NUM_ENVS": 64,
            "NUM_ACTIONS": args.n_dims,
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
            "DISCRETE": False,
            "POP_SIZE": 128,
        }
        exploiter_env_fn = functools.partial(GymnaxGoalWrapper, env=inner_env)
        exploiter_net = PPO_Network(args.n_dims, discrete=exploiter_config["DISCRETE"], hsize=256, activation_fn=nn.relu)

    else:
        raise NotImplementedError

    config["PUPPET"] = True
    config["ENV"] = args.env
    config["OUTER_ALGO"] = args.outer_algo
    config["STATIC"] = args.static
    config["CT_SCALE"] = args.scale
    config["GEN_EXPLOIT"] = False
    config["N_DEVICES"] = jax.local_device_count()
    config["CENTER_RANK"] = args.centered_rank
    config["ZERO_SHOT"] = args.zero_shot
    config["NUM_EVAL_ENVS"] = 64
    config["NORMALIZE"] = args.normalize_obs
    config["INCLUDE_ACTIONS"] = args.include_actions
    config["ADD_WRAPPER"] = False
    exploiter_config["NUM_EVAL_ENVS"] = 64
    exploiter_config["INCLUDE_ACTIONS"] = args.include_actions
    exploiter_config["ADD_WRAPPER"] = False
    exploiter_config["GEN_EXPLOIT"] = False
    """
    INIT MFOS AGENT THIGNS
    """
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_network, rng_init = jax.random.split(rng, 3)
    if args.include_actions:
        pholder = jnp.zeros((inner_env.observation_space.shape[0] + config["NUM_ACTIONS"],))
    else:
        pholder = jnp.zeros((inner_env.observation_space.shape[0],))
    if args.static:
        mfos_network = NetworkMapper["MLP"](
            num_hidden_units=64,
            num_hidden_layers=2,
            num_output_units=args.n_dims,
            hidden_activation="relu",
            output_activation="tanh",
        )
        params = mfos_network.init(
            rng_network,
            x=pholder,
            rng=rng_init,
        )
    else:
        mfos_network = NetworkMapper["LSTM"](
            num_hidden_units=32,
            num_output_units=args.n_dims,
            output_activation="tanh",
        )
        carry_init = mfos_network.initialize_carry()
        params = mfos_network.init(
            rng_network,
            x=pholder,
            carry=carry_init,
            rng=rng_init,
        )

    if args.zero_shot:
        rng, rng_init = jax.random.split(rng)
        init_x = jnp.zeros(exploiter_config["OBS_SIZE"])
        exploiter_params = exploiter_net.init(rng_init, x=init_x)
        both_params = {
            "mfos": unfreeze(params["params"]),
            "exploiter": unfreeze(exploiter_params["params"]),
        }
        param_reshaper = ParameterReshaper(both_params)

    else:
        param_reshaper = ParameterReshaper(params["params"])

    def single_rollout(rng_input, mfos_params):
        if args.zero_shot:
            exploiter_params = {"params": mfos_params["exploiter"]}
            mfos_params = mfos_params["mfos"]

        rng_train, rng_exploit = jax.random.split(rng_input)
        # env = MfosWrapper(inner_env, mfos_network.apply, mfos_params, config)
        if config["STATIC"]:
            env = MfosWrapper(inner_env, mfos_network.apply, mfos_params, config)
        else:
            mfos_init = mfos_network.initialize_carry()
            env = MfosWrapper(inner_env, mfos_network.apply, mfos_params, config, mfos_init)

        if config["NORMALIZE"]:
            env = VecNormalizer(env, config)
            agent = PPO(
                network,
                env.reset,
                env.step,
                config,
            )
        else:
            agent = PPO(
                network,
                jax.vmap(env.reset),
                jax.vmap(env.step),
                config,
            )
        val, metric = agent.train(rng_train)

        if config["NORMALIZE"]:
            env = exploiter_env_fn(config=config, network_apply=val[0].apply_fn, network_params=val[0].params, ob_mean=val[1].ob_mean, ob_var=val[1].ob_var)
        else:
            env = exploiter_env_fn(config=config, network_apply=val[0].apply_fn, network_params=val[0].params)

        env_reset = jax.vmap(env.reset)
        env_step = jax.vmap(env.step)
        env_reset_zero = jax.vmap(env.reset_zero)

        if not args.zero_shot:
            agent = PPO(
                exploiter_net,
                env_reset,
                env_step,
                exploiter_config,
            )
            val, metric = agent.train(rng_exploit)
            return metric.mean()
            # return metric[-4:].mean()

        else:
            agent = PPO(
                exploiter_net,
                env_reset_zero,
                env_step,
                exploiter_config,
            )
            metric = agent.eval_network(exploiter_params, rng_exploit)
            return metric.mean()

    vmap_rollout = jax.vmap(single_rollout, in_axes=(0, None))
    if args.pmap:
        rollout = jax.pmap(jax.jit(jax.vmap(vmap_rollout, in_axes=(None, param_reshaper.vmap_dict))))
    else:
        rollout = jax.jit(jax.vmap(vmap_rollout, in_axes=(None, param_reshaper.vmap_dict)))
    # rollout = jax.jit(jax.vmap(vmap_rollout, in_axes=(None, param_reshaper.vmap_dict)))

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
    num_rollouts = 4
    save_every_k_gens = 32

    es_logging = ESLog(param_reshaper.total_params, num_generations, top_k=5, maximize=True)
    log = es_logging.initialize()

    fit_shaper = FitnessShaper(centered_rank=args.centered_rank, z_score=True, w_decay=0.0, maximize=True)

    state = strategy.initialize(rng, es_params)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_dir = f"{dir_path}/data/{str(datetime.now()).replace(' ', '_')}_{args.env}_{args.outer_algo}_{args.n_dims}_test_time"
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
        # fitness = rollout(batch_rng, reshaped_params).mean(axis=1)
        fit_re = fit_shaper.apply(x, fitness)
        state = strategy.tell(x, fit_re, state, es_params)
        log = es_logging.update(log, x, fitness)

        # print("Generation: ", gen, "Generation: ", log["log_top_1"][gen])
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
