import logging
import jax
from octo.model.octo_model import OctoModel
from octo.data import dataset as data_lib
from octo.utils import train_utils
from octo.utils import spec
from octo.model.components import action_heads
from octo.model.components import tokenizers
from absl import flags
import optax
import sys
import tqdm
from tensorflow_datasets import as_numpy as t_n
from robosuite.recorder.dataset_builder import RobosuiteDatasetBuilder

RECORDING_PATH = flags.DEFINE_string(
    "recording_path", "/data/episodes/train", "path to recordings for training"
)

BASE_MODEL_PATH = flags.DEFINE_string(
    "base_model_path", "/data/model/octo-base-1.5", "path to the pre finetuning path"
)
BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 1, "batch size for finetuning"
)
ACTION_SIZE = flags.DEFINE_integer(
    "action_size", 50, "action horizon length"
)
CHECKPOINT_DIR = flags.DEFINE_string(
    "checkpoint_dir", "/data/checkpoints/octo-finetune", "path to store checkpoints"
)
FREEZE_TRANSFORMER =flags.DEFINE_bool(
    "freeze_trans", False, "whether to freeze transformer"
)
obs_tokenizers = "observation_tokenizers"
proprio = "proprio"

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    pretrained = OctoModel.load_pretrained(BASE_MODEL_PATH.value)
    dataset = data_lib.make_single_dataset(
        dataset_kwargs=dict(
            name="RobosuiteDatasetBuilder",
            data_dir='/home/vivekbagade/tensorflow_datasets',
            image_obs_keys={"primary": "image", "wrist": "wrist_image"},
            proprio_obs_key="state",
            language_key = "language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            action_horizon=ACTION_SIZE.value,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256), "wrist": (256, 256)}
        ),
        train=True,
    )
    iter = (
        t_n(dataset.repeat().unbatch().shuffle(10000).batch(BATCH_SIZE.value))
    )
    text_processor = pretrained.text_processor

    def process_batch(batch):
        batch = train_utils.process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch
    iter = map(process_batch, iter)
    cur = next(iter)

    config = pretrained.config
    config["model"][obs_tokenizers][proprio] = spec.ModuleSpec.create(
        tokenizers.LowdimObsTokenizer,
        n_bins = 256,
        bin_type = "normal",
        low = -2.0,
        high = 2.0,
        obs_keys = [proprio],
    )

    config["model"]["heads"]["action"] = spec.ModuleSpec.create(
        action_heads.L1ActionHead,
        action_horizon=ACTION_SIZE.value,
        action_dim=7,
        readout_key = "readout_action",
    )

    model = OctoModel.from_config(
        config,
        example_batch=cur,
        text_processor=text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics)
    
    merged_params = train_utils.merge_params(
        model.params, pretrained.params
    )

    model = model.replace(params=merged_params)
    del pretrained

    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)],
        [100],
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FREEZE_TRANSFORMER.value:
        frozen_keys.append("BlockTransformer_0")
    tx = train_utils.freeze_weights(tx, model.params, frozen_keys)
    train_state = train_utils.TrainState.create(
        rng=jax.random.PRNGKey(0),
        model=model,
        tx=tx,
    )

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # action head knows to pull out the "action" readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics
    
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info
    
    logging.info("Starting finetuning loop..")
    for i in tqdm.tqdm(range(5000), total=5000, dynamic_ncols=True):
        batch = next(iter)
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 1000 == 0:
            update_info = jax.device_get(update_info)
            logging.info(update_info)
        if (i+1) % 1000 == 0:
            train_state.model.save_pretrained(step=i,
                                            checkpoint_path=CHECKPOINT_DIR.value)


