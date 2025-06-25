import os
import sys

# Adds the repo root (e.g.,/fairmast-data-preprocessing) to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
print(REPO_ROOT)

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from scripts.MAST_tools.MAST_dataset import MastDataset
from scripts.pipeline_test.utils.utils import read_data_split_csv, flatten_then_collate

from scripts.pipeline_test.preprocessing.sampled_shot_list import yamane_sampled_shot_list
from scripts.pipeline_test.preprocessing.standardscaling_preprocessing import get_mean_shot, get_std_shot

from scripts.pipeline_test.utils.utils import ComposeTransforms
from scripts.pipeline_test.transformers.signal_level_transformers.fill_with_zeros_imputer_transform import FillWithZerosImputerTransform
from scripts.pipeline_test.transformers.signal_level_transformers.forward_fill_imputer_transform import ForwardFillImputerTransform
from scripts.pipeline_test.transformers.signal_level_transformers.sample_wise_normalize_transform import SamplewiseNormalizeTransform
from scripts.pipeline_test.transformers.signal_level_transformers.pretrained_stdscale_normalize_transform import StdScalingTransform
from scripts.pipeline_test.transformers.signal_level_transformers.sampling_reference_time_transform import SamplingtoReferenceTimeTransform

from scripts.pipeline_test.transformers.shot_level_transformers.truncation_transform import TruncationTransform
from scripts.pipeline_test.transformers.shot_level_transformers.window_segmenter_transform import WindowSegmenterTransform
from scripts.pipeline_test.transformers.shot_level_transformers.cnn_transform import CNNTransform

from scripts.pipeline_test.models.cnn_model import MultiBranchCNNModel

from multiprocess import cpu_count
print(f"\nNumber of Cores: {cpu_count()}\n")

# ------------------------------------------------------------------------------------------------------------------- #

# Determine device to train on
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ------------------------------------------------------------------------------------------------------------------- #
if __name__== "__main__":

    mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------------------------ #
    # COMMON PIPELINE

    # Create sets of shot IDs for training, validation and testing
    train_shots, test_shots, val_shots = read_data_split_csv()

    # Create MAST datasets
    ref_freq = 0.005
    source_signal_list = [
        ('magnetics', 'flux_loop_flux' ),
        ('magnetics', 'b_field_pol_probe_ccbv_field' ),
        ('magnetics', 'b_field_pol_probe_obr_field' ),
        ('magnetics', 'b_field_pol_probe_obv_field' ),
        ('pf_active', 'solenoid_current'),
        ('pf_active', 'coil_voltage'),
        ('pf_active', 'coil_current'),
        ('pulse_schedule', 'i_plasma'),
        ('summary', 'power_nbi'),
        ('equilibrium', 'elongation'),
        ('equilibrium', 'elongation_axis'),
        ('equilibrium', 'triangularity_upper'),
        ('equilibrium', 'triangularity_lower'),
        ('equilibrium', 'minor_radius'),
        ('equilibrium', 'magnetic_axis_r'),
        ('equilibrium', 'magnetic_axis_z')
    ]

    # ------------------------------------------------------------------------------------ #
    # Fitting of mean and std for signal transform
    preprocessing_train_dataset = MastDataset(
        local=True,
        shots_list=yamane_sampled_shot_list(train_shots, error=0.05),
        source_signal_list=source_signal_list,
        signal_level_transform_map=None,
        shot_level_transform_map=None
    )
    print("len(preprocessing_train_dataset)", len(preprocessing_train_dataset))
    dict_mean = get_mean_shot(preprocessing_train_dataset)
    dict_std = get_std_shot(preprocessing_train_dataset)

    # Map Signal transform
    signal_transform_map = {var: ComposeTransforms([
        ForwardFillImputerTransform(),
        # SamplewiseNormalizeTransform(),
        StdScalingTransform(dict_mean[var], dict_std[var]),
        FillWithZerosImputerTransform(),
        SamplingtoReferenceTimeTransform(ref_freq),
        # SamplewiseNormalizeTransform()
    ])
        for var in [ f'{source}-{signal}' for source, signal in source_signal_list]
    }
    
    # ------------------------------------------------------------------------------------ #
    # CNN PIPELINE

    # Create CNN datasets
    # parameters_cnn = {  'x': { 'magnetics-flux_loop_flux' : 't',
    #                         'magnetics-b_field_pol_probe_ccbv_field' : 't',
    #                         'magnetics-b_field_pol_probe_obr_field' : 't',
    #                         'magnetics-b_field_pol_probe_obv_field' : 't',
    #                         'pf_active-solenoid_current' : 't',
    #                         'pf_active-coil_voltage' : 't',
    #                         'pf_active-coil_current' : 't',
    #                         'pulse_schedule-i_plasma' : 't',
    #                         'summary-power_nbi' : 't',
    #                         },
    #                     'y': { 'equilibrium-elongation' : 't+dt',
    #                         'equilibrium-elongation_axis' : 't+dt',
    #                         'equilibrium-triangularity_upper' : 't+dt',
    #                         'equilibrium-triangularity_lower' : 't+dt',
    #                         'equilibrium-minor_radius' : 't+dt',
    #                         'equilibrium-magnetic_axis_r' : 't+dt',
    #                         'equilibrium-magnetic_axis_z' : 't+dt', 
    #                         },
    #                     'dt': int(0.025/ref_freq)
    #                 }

    parameters_windows_segmenter = {
        'x_keys': [
            'equilibrium-elongation',
            'magnetics-flux_loop_flux',
            'magnetics-b_field_pol_probe_ccbv_field',
            'magnetics-b_field_pol_probe_obr_field',
            'magnetics-b_field_pol_probe_obv_field',
            'pf_active-solenoid_current',
            'pf_active-coil_voltage',
            'pf_active-coil_current',
            'pulse_schedule-i_plasma',
            'summary-power_nbi',
        ],
        'y_keys': [
            'equilibrium-elongation',
            'equilibrium-elongation_axis',
            'equilibrium-triangularity_upper',
            'equilibrium-triangularity_lower',
            'equilibrium-minor_radius',
            'equilibrium-magnetic_axis_r',
            'equilibrium-magnetic_axis_z',
        ],
        'x_window_sec': 0,
        'y_window_sec': 0,
        'dt_sec': 0.025,
        'stride_sec': None,
        'stride_unitary': True,
        'min_samples_per_window': 1,
        'verbose': False,
    }

    shot_transform_map = ComposeTransforms([  # shape-consistent transform
        TruncationTransform(),
        WindowSegmenterTransform(**parameters_windows_segmenter),  # shape modifying transform
        CNNTransform() # shape modifying transform
        ])
    
    # --------------------------------------------------------------------------------------------------- #
    # Prepare dataset and dataloader

    train_dataset = MastDataset(
        local=True,
        shots_list=train_shots,
        source_signal_list=source_signal_list,
        signal_level_transform_map=signal_transform_map,
        shot_level_transform_map=shot_transform_map
    )
    print("len(mast_train_dataset)", len(train_dataset))

    val_dataset = MastDataset(
        local=True,
        shots_list=val_shots,
        source_signal_list=source_signal_list,
        signal_level_transform_map=signal_transform_map,
        shot_level_transform_map=shot_transform_map
    )
    print("len(val_dataset)", len(val_dataset))

    # test_dataset = MastDataset(
    #     local=True,
    #     shots_list=test_shots[0:15],
    #     source_signal_list=source_signal_list,
    #     signal_level_transform_map=signal_transform_map,
    #     shot_level_transform_map=shot_transform_map
    # )
    # print("len(test_dataset)", len(test_dataset))


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=500, #500
        # batch_size=len(train_dataset),
        num_workers=64, #64
        # num_workers=5,
        shuffle=True,
        #    drop_last=True,
        collate_fn = flatten_then_collate
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=500, #500
        # batch_size=len(val_dataset),
        # num_workers=cpu_count(),
        num_workers=64, #64
        shuffle=True,
        #    drop_last=True,
        collate_fn = flatten_then_collate
    )

    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=5, #500
    #     # batch_size=len(test_dataset),
    #     # num_workers=cpu_count(),
    #     num_workers=0, #64
    #     shuffle=True,
    #     #    drop_last=True,
    #     collate_fn = flatten_then_collate
    # )
    
    # --------------------------------------------------------------------------------------------------- #
    # Create CNN architecture
    input_shapes = [arr.shape for arr in train_dataloader.dataset[0][0][0] ]
    print('input_shapes', input_shapes)
    output_shape = [arr.shape for arr in train_dataloader.dataset[0][0][1] ]
    print('output_shape', output_shape)
    model = MultiBranchCNNModel(input_shapes, output_shape).to(device)
    
    # --------------------------------------------------------------------------------------------------- #
    # Train CNN model
    num_epochs = 500
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    patience = 5  # You can change this
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (x_train, y_train) in enumerate(train_dataloader):

            x_train = [arr.to(torch.float32).to(device) for arr in x_train]
            # print([arr.shape for arr in x_train])
            # print(y_train.min().item(), y_train.max().item())
            y_train = y_train[0].to(torch.float32).to(device)
            # print(y_train.shape)
            print(f'\nBatch {batch_idx} size is {len(y_train)}')

            outputs = model(*x_train).squeeze()
            # print('outputs', outputs.shape)
            loss = criterion(outputs, y_train)
            print('Batch loss', loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += len(y_train)
            print(f'Actual num_batches is {num_batches}')

        avg_loss = running_loss / num_batches
        # print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Validation phase & Early stopping check

        model.eval()
        val_running_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x_val, y_val in val_dataloader:
                x_val = [arr.to(torch.float32).to(device) for arr in x_val]
                y_val = y_val[0].to(torch.float32).to(device)

                val_outputs = model(*x_val).squeeze()
                val_loss = criterion(val_outputs, y_val)

                val_running_loss += val_loss.item()
                val_batches += len(y_val)

        avg_val_loss = val_running_loss / val_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Save best model state
            os.makedirs("cnn_model_test_25_06/", exist_ok=True)
            torch.save(best_model_state, "cnn_model_test_25_06/best_model.pt")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                early_stop = True
                break
        
        

    # Optionally restore best model weights
    if early_stop:
        model.load_state_dict(best_model_state)


    # model.eval()  # Set the model to evaluation mode
    # test_loss = 0.0
    # test_batches = 0
    # criterion = torch.nn.MSELoss()  # or whatever you used during training

    # with torch.no_grad():  # Disable gradient calculation for efficiency
    #     for x_test, y_test in test_dataloader:
    #         x_test = [arr.to(torch.float32).to(device) for arr in x_test]
    #         y_test = y_test[0].to(torch.float32).to(device)

    #         outputs = model(*x_test).squeeze()
    #         loss = criterion(outputs, y_test)
    #         test_loss += loss.item()

    #         test_batches += len(y_test)

    # avg_test_loss = test_loss / test_batches
    # print(f"Test Loss: {avg_test_loss:.4f}")