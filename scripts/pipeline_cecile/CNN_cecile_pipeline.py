import os
import sys

cwd = os.path.dirname(os.getcwd())
mother_dir = os.path.dirname(cwd) + os.sep
# sys.path.append(os.path.abspath(os.path.join(mother_dir , "MAST_tools")))
sys.path.append(mother_dir)
sys.path.append(cwd)
sys.path.append(os.path.join( os.path.dirname(cwd) ) )

import torch
from MAST_transformer import (ComposeTransform, 
                              ForwardFillImputerTransform, 
                               SamplewiseNormalizeTransform,
                              FillWithZerosImputerTransform,
                              SamplingtoReferenceTimeTransform)
from utils import read_data_split_csv
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader, Dataset

from MAST_dataset import MAST_Dataset

from CNN_transform import CNNSpecificTransform
from CNN_model import MultiBranchCNNModel
import torch.multiprocessing as mp


# Determine device to train on
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def flatten_then_collate(batch):

    print(batch)

    try:
        print(f"Collating batch of size {len(batch)}")
        
        # Flatten the batch of lists into a single list
        flattened_batch = [item for sublist in batch for item in sublist]
        print(f'Number of samples from batch = {len(batch)} shots is N = {len(flattened_batch)}')
        # Use the default collate function
        return default_collate(flattened_batch)
    
    except Exception as e:
        print("Exception in collate_fn:", e)
        raise


if __name__== "__main__":

    mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------------------------ #
    # COMMON PIPELINE

    # Create sets of shot IDs for training, validation and testing
    train_shots, test_shots, val_shots = read_data_split_csv()

    # Create MAST datasets
    ref_freq = 0.005
    source_signal_list = [  ( 'magnetics', 'flux_loop_flux' ),
                            ( 'magnetics', 'b_field_pol_probe_ccbv_field' ), 
                            ( 'magnetics', 'b_field_pol_probe_obr_field' ), 
                            ( 'magnetics', 'b_field_pol_probe_obv_field' ),                            
                            ( 'pf_active', 'solenoid_current'),
                            ( 'pf_active', 'coil_voltage'),
                            ( 'pf_active', 'coil_current'),
                            ( 'pulse_schedule', 'i_plasma'),
                            ( 'summary', 'power_nbi'),
                            ( 'equilibrium', 'elongation'), 
                            ( 'equilibrium', 'elongation_axis'),
                            ( 'equilibrium', 'triangularity_upper'), 
                            ( 'equilibrium', 'triangularity_lower'), 
                            ( 'equilibrium', 'minor_radius'), 
                            ( 'equilibrium', 'magnetic_axis_r'), 
                            ( 'equilibrium', 'magnetic_axis_z') ]
    transform_map = {k: ComposeTransform([ForwardFillImputerTransform(),
                                            # SamplewiseNormalizeTransform(),
                                            FillWithZerosImputerTransform(),
                                            SamplingtoReferenceTimeTransform(ref_freq),
                                            # SamplewiseNormalizeTransform()
                                            ]) for k in [ f'{source}-{signal}' for source, signal in source_signal_list ] }
    
    # ------------------------------------------------------------------------------------ #
    # CNN PIPELINE

    # Create CNN datasets
    parameters_cnn = {  'x': { 'magnetics-flux_loop_flux' : 't',
                            'magnetics-b_field_pol_probe_ccbv_field' : 't',
                            'magnetics-b_field_pol_probe_obr_field' : 't',
                            'magnetics-b_field_pol_probe_obv_field' : 't',
                            'pf_active-solenoid_current' : 't',
                            'pf_active-coil_voltage' : 't',
                            'pf_active-coil_current' : 't',
                            'pulse_schedule-i_plasma' : 't',
                            'summary-power_nbi' : 't',
                            },
                        'y': { 'equilibrium-elongation' : 't+dt',
                            'equilibrium-elongation_axis' : 't+dt',
                            'equilibrium-triangularity_upper' : 't+dt',
                            'equilibrium-triangularity_lower' : 't+dt',
                            'equilibrium-minor_radius' : 't+dt',
                            'equilibrium-magnetic_axis_r' : 't+dt',
                            'equilibrium-magnetic_axis_z' : 't+dt', 
                            },
                        'dt': int(0.025/ref_freq)
                    }

    from torch.utils.data._utils.collate import default_collate

    
    # --------------------------------------------------------------------------------------------------- #
    # Prepare dataset and dataloader

    train_dataset = MAST_Dataset(local = True, 
                                shots_list = train_shots[0:50], 
                                source_signal_list = source_signal_list, 
                                transform_map=transform_map,
                                model_specific_transform=CNNSpecificTransform(parameters_cnn))
    print("len(mast_train_dataset)", len(train_dataset))

    val_dataset = MAST_Dataset(local = True, 
                                shots_list =val_shots[0:30], 
                                source_signal_list = source_signal_list, 
                                transform_map=transform_map,
                                model_specific_transform=CNNSpecificTransform(parameters_cnn))
    print("len(val_dataset)", len(val_dataset))

    test_dataset = MAST_Dataset(local = True, 
                                shots_list = test_shots[0:30], 
                                source_signal_list = source_signal_list, 
                                transform_map=transform_map,
                                model_specific_transform=CNNSpecificTransform(parameters_cnn))
    print("len(test_dataset)", len(test_dataset))


    train_dataloader = DataLoader( train_dataset,
            batch_size=32,
            num_workers=4,
            shuffle=True,
            #    drop_last=True, 
            collate_fn = flatten_then_collate)

    val_dataloader = DataLoader( val_dataset,
            batch_size=32,
            num_workers=4,
            shuffle=True,
            #    drop_last=True, 
            collate_fn = flatten_then_collate)

    test_dataloader = DataLoader( train_dataset,
            batch_size=32,
            num_workers=4,
            shuffle=True,
            #    drop_last=True, 
            collate_fn = flatten_then_collate)
    
    # --------------------------------------------------------------------------------------------------- #
    # Create CNN architecture
    input_shapes = [arr.shape for arr in train_dataloader.dataset[0][0][0] ]
    print('input_shapes', input_shapes)
    output_shape = train_dataloader.dataset[0][0][1].shape
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
            print(f'Batch size is {len(y_train)}')

            x_train = [arr.to(torch.float32).to(device) for arr in x_train]
            # print([arr.shape for arr in x_train])
            # print(y_train.min().item(), y_train.max().item())
            y_train = y_train.to(torch.float32).to(device)
            # print(y_train.shape)

            outputs = model(*x_train).squeeze()
            # print('outputs', outputs.shape)
            loss = criterion(outputs, y_train)
            # print('loss', loss)

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
                y_val = y_val.to(torch.float32).to(device)

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
            os.makedirs("cnn_model/", exist_ok=True)
            torch.save(best_model_state, "cnn_model/best_model.pt")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                early_stop = True
                break
        
        break

    # Optionally restore best model weights
    if early_stop:
        model.load_state_dict(best_model_state)



    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    test_batches = 0
    criterion = torch.nn.MSELoss()  # or whatever you used during training

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for x_test, y_test in test_dataloader:
            x_test = [arr.to(torch.float32).to(device) for arr in x_test]
            y_test = y_test.to(torch.float32).to(device)

            outputs = model(*x_test).squeeze()
            loss = criterion(outputs, y_test)
            test_loss += loss.item()

            test_batches += len(y_test)

    avg_test_loss = test_loss / test_batches
    print(f"Test Loss: {avg_test_loss:.4f}")