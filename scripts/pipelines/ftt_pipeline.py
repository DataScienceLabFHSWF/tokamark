import os
import sys
import torch
import torch.multiprocessing as mp
from multiprocessing import cpu_count

# === Repo root ===
REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(),
    "..", ".."
))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# === Import config ===
from scripts.pipelines.configs.ftt_config import (
    SUBSET_OF_SHOTS, OUTPUT_SUB_FOLDER, BATCH_SIZE, NUM_WORKERS, EPOCHS,
    REF_FREQ, SOURCE_SIGNAL_LIST, LOCAL_FLAG,
    WINDOW_SEGMENTER_PARAMS, MODEL_DTYPE, DEVICE, VERBOSE,
    DEFAULT_INPUT_ENCODER_BY_DIM, DEFAULT_TARGET_ENCODER
)

# === Common pipeline helpers ===
from scripts.pipelines.cnn_pipeline import (
    get_train_test_val_shots,
    fit_mean_and_std_for_signal_transform,
    initialize_datasets,
    initialize_dataloaders
)

# === Transforms ===
from scripts.pipelines.utils.utils import ComposeTransforms, collate_fttransform
from scripts.pipelines.transforms.signal_level_transforms.fill_profile_with_zeros_imputer_transform import FillProfileWithZerosTransform
from scripts.pipelines.transforms.signal_level_transforms.pretrained_stdscale_normalize_transform import StdScalingTransform
from scripts.pipelines.transforms.signal_level_transforms.sampling_reference_time_transform import SamplingToReferenceTimeTransform
from scripts.pipelines.transforms.shot_level_transforms.truncation_transform import TruncationTransform
from scripts.pipelines.transforms.shot_level_transforms.window_segmenter_transform import WindowSegmenterTransform
from scripts.pipelines.transforms.shot_level_transforms.drop_sample_with_nans import DropSampleWithNans
from scripts.pipelines.transforms.shot_level_transforms.ftt_transform import FTTransformPrep

# === Model & registries ===
from scripts.pipelines.models.ftt_model import MultiModalFTTransformer, InputRegistry, TargetRegistry
from scripts.pipelines.utils.modality_codecs import get_encoder, DECODER_REGISTRY
from scripts.pipelines.utils.ftt_utils import infer_modality_from_shape

def get_decoder(name: str, **kwargs):
    """Factory for decoder registry"""
    if name is None:
        return None
    return DECODER_REGISTRY[name](**kwargs)

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    print(f"\nNumber of available cores: {cpu_count()}\n")
    mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # DATA SPLITS & SCALING
    # ------------------------------------------------------------------------------------------------------------------
    train_shots, test_shots, val_shots = get_train_test_val_shots(max_index=SUBSET_OF_SHOTS)

    # dict_mean, dict_std = fit_mean_and_std_for_signal_transform(
    #     output_sub_dir=OUTPUT_SUB_FOLDER,
    #     verbose=True,
    #     local=LOCAL_FLAG
    # )

    # Per-variable signal transforms
    all_vars = [f"{src}-{sig}" for src, sig in SOURCE_SIGNAL_LIST]
    signal_transform_map = {
        var: ComposeTransforms([
            FillProfileWithZerosTransform(),
            # StdScalingTransform(dict_mean[var], dict_std[var]),
            # SamplingToReferenceTimeTransform(REF_FREQ),
        ])
        for var in all_vars
    }

    # ------------------------------------------------------------------------------------------------------------------
    # SHOT-LEVEL TRANSFORM FOR FTT
    # ------------------------------------------------------------------------------------------------------------------
    y_key_to_target = {yk: yk for yk in WINDOW_SEGMENTER_PARAMS['y_keys']}

    shot_transform = ComposeTransforms([
        TruncationTransform(),
        WindowSegmenterTransform(**WINDOW_SEGMENTER_PARAMS),
        DropSampleWithNans(),
        FTTransformPrep(
            x_keys=WINDOW_SEGMENTER_PARAMS['x_keys'],
            y_key_to_target=y_key_to_target,
            ensure_3d=True
        )
    ])

    # ------------------------------------------------------------------------------------------------------------------
    # DATASETS & LOADERS
    # ------------------------------------------------------------------------------------------------------------------
    datasets = initialize_datasets(
        sources_and_signals=SOURCE_SIGNAL_LIST,
        shots={"train": train_shots, "val": val_shots, "test": test_shots},
        sig_tran_map=signal_transform_map,
        shot_tran=shot_transform,
        local_flag=False,
        verbose=True
    )

    dataloaders = initialize_dataloaders(
        datasets=datasets,
        collate_function=collate_fttransform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        verbose=True
    )

    train_loader = dataloaders["train"]

    # ------------------------------------------------------------------------------------------------------------------
    # INFER SHAPES & BUILD REGISTRIES
    # ------------------------------------------------------------------------------------------------------------------
    xb, active_targets, y_native = next(iter(train_loader))
    input_shapes = [arr.shape for arr in xb[0]]
    target_shapes = [tuple(v.shape[1:]) for v in y_native.values()]  # drop batch dim

    # === Input Registry ===
    input_specs = {name: dict(shape=shp) for name, shp in zip(WINDOW_SEGMENTER_PARAMS['x_keys'], input_shapes)}
    input_registry = InputRegistry(
        specs=input_specs,
        get_encoder=get_encoder,
        infer_modality_from_shape=infer_modality_from_shape,
    )
    input_registry.bind_shapes({name: shp for name, shp in zip(WINDOW_SEGMENTER_PARAMS['x_keys'], input_shapes)})
    input_registry.auto_fill_modalities()
    for name, spec in input_registry.specs.items():
        if spec.get("encoder_name") is None:
            dim = spec["modality"]
            if dim in DEFAULT_INPUT_ENCODER_BY_DIM:
                spec.update(DEFAULT_INPUT_ENCODER_BY_DIM[dim])
    input_registry.build_encoders()

    # === Target Registry ===
    target_specs = {name: dict(shape=shp) for name, shp in zip(active_targets, target_shapes)}
    target_registry = TargetRegistry(
        specs=target_specs,
        get_encoder=get_encoder,
        get_decoder=get_decoder
    )
    target_registry.bind_shapes({name: shp for name, shp in zip(active_targets, target_shapes)})
    for name, spec in target_registry.specs.items():
        if spec.get("encoder_name") is None:
            spec.update(DEFAULT_TARGET_ENCODER)
    target_registry.auto_fill_decoders()

    # ------------------------------------------------------------------------------------------------------------------
    # MODEL INIT
    # ------------------------------------------------------------------------------------------------------------------
    model = MultiModalFTTransformer(
        input_registry=input_registry,
        target_registry=target_registry,
        dtype=MODEL_DTYPE,
        device=DEVICE,
        verbose=VERBOSE
    )

    print("\nFTTransformer initialized.")
    print("  Input shapes:", input_shapes)
    print("  Active targets:", active_targets)
    print("  Target shapes:", target_shapes)
