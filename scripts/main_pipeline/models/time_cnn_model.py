import torch
import torch.nn as nn



# ======================================================================================================================
class Conv1DBranch(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_channels, D):
        super().__init__()
        self.layer1 = nn.BatchNorm1d(input_channels)
        
        self.layer1_1 = nn.Conv1d(input_channels, D, kernel_size=3, padding=1)
        self.layer1_2 = nn.ReLU() # output shape [8, 16, 27, 33, 60]
        self.layer1_3 = nn.MaxPool1d(2, padding=1)
        self.layer1_4 = nn.BatchNorm1d(D)

        self.layer2_1 = nn.Conv1d(D, 2 * D, kernel_size=3, padding=1)
        self.layer2_2 = nn.ReLU() # output shape [8, 16, 27, 33, 60]
        self.layer2_3 = nn.MaxPool1d(2, padding=1)
        self.layer2_4 = nn.BatchNorm1d(2 * D)


    def forward(self, x):
        print('\nConv1D')
        print(x.shape)
        x = self.layer1(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer1_3(x)
        x = self.layer1_4(x)

        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer2_4(x)
        print(x.shape)

        return x



# ======================================================================================================================
class Conv2DBranch(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_channels, D):
        super().__init__()
        self.layer1 = nn.BatchNorm2d(input_channels)
        
        self.layer1_1 = nn.Conv2d(input_channels, D, kernel_size=3, padding=1)
        self.layer1_2 = nn.ReLU() # output shape [8, 16, 27, 33, 60]
        self.layer1_3 = nn.MaxPool2d(2, padding=1)
        self.layer1_4 = nn.BatchNorm2d(D)

        self.layer2_1 = nn.Conv2d(D, 2 * D, kernel_size=3, padding=1)
        self.layer2_2 = nn.ReLU() # output shape [8, 16, 27, 33, 60]
        self.layer2_3 = nn.MaxPool2d(2, padding=1)
        self.layer2_4 = nn.BatchNorm2d(2 * D)


    def forward(self, x):
        print('\nConv2D')
        print(x.shape)
        x = self.layer1(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer1_3(x)
        x = self.layer1_4(x)

        x = self.layer2_1(x)

        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer2_4(x)

        t_comp = x.shape[2]
        x = nn.AdaptiveMaxPool2d((t_comp, 1))(x).squeeze(-1)
        print(x.shape)
        
        return x



# ======================================================================================================================
class Conv3DBranch(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_channels, D):
        super().__init__()
        self.layer1 = nn.BatchNorm3d(input_channels)
        
        self.layer1_1 = nn.Conv3d(input_channels, D, kernel_size=3, padding=1)
        self.layer1_2 = nn.ReLU() # output shape [8, 16, 27, 33, 60]
        self.layer1_3 = nn.MaxPool3d(2, padding=1)
        self.layer1_4 = nn.BatchNorm3d(D)

        self.layer2_1 = nn.Conv3d(D, 2 * D, kernel_size=3, padding=1)
        self.layer2_2 = nn.ReLU() # output shape [8, 16, 27, 33, 60]
        self.layer2_3 = nn.MaxPool3d(2, padding=1)
        self.layer2_4 = nn.BatchNorm3d(2 * D)


    def forward(self, x):
        print('\nConv3D')
        print(x.shape)
        x = self.layer1(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer1_3(x)
        x = self.layer1_4(x)

        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer2_4(x)
       
        t_comp = x.shape[2]
        x = nn.AdaptiveMaxPool3d((t_comp, 1, 1))(x).squeeze(-1).squeeze(-1)
        print(x.shape)
        
        return x



# ======================================================================================================================
class MultiBranchTimeCNNModel(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shapes, output_shape, D=16):
        super().__init__()

        self.D = D
        self.branches = nn.ModuleList()
        self.output_shape = output_shape[0][0]
        # merged_dim = 0

        for shape in input_shapes:
            if len(shape) == 4:  # e.g., (2, T, 15, 17) images evolving in time
                input_len = shape[0]
                branch = Conv3DBranch(input_len, D)
                # merged_dim += input_len
            elif len(shape) == 3:  # e.g., (1, T, 15) profiles evolving in time
                input_len = shape[0]
                branch = Conv2DBranch(input_len, D)
                # merged_dim += input_len
            elif len(shape) == 2:  # e.g., (7, T, ) time series evolving in time
                input_len = shape[0]
                branch = Conv1DBranch(input_len, D)
                # merged_dim += shape[0]
            else:
                raise ValueError(f"Unsupported input shape: {shape}")
            self.branches.append(branch)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, *inputs):
        branch_outputs = []

        for branch, x in zip(self.branches, inputs):
            out = branch(x)
            branch_outputs.append(out)
        
        print('\nCommon Layer')     
        merged = torch.cat(branch_outputs, dim=1)
        print(merged.shape)   
        merged = Conv1DBranch(merged.shape[1], 2*self.D)(merged)
        merged = torch.flatten(merged,1)
        print(merged.shape)    

        self.fc = nn.Sequential(
            nn.BatchNorm1d(merged.shape[1]),
            nn.Linear(merged.shape[1], 4 * self.D),
            nn.ReLU(),
            nn.BatchNorm1d(4 * self.D),
            nn.Linear(4 * self.D, 2 * self.D),
            nn.ReLU(),
            nn.BatchNorm1d(2 * self.D),
            nn.Dropout(0.2),
            nn.Linear(2 * self.D, self.output_shape),
        )
        print('\nDNN Layer')    
        print(merged.shape)    
        merged = self.fc(merged)
        print(merged.shape)    

        return merged

    # ------------------------------------------------------------------------------------------------------------------



# ======================================================================================================================
# # Example usage
# batch_size = 8

# D = 16
# T = 100

# input_channels_1 = 5  # Number of input channels

# input_channels_2 = 1  
# height_length_2 = 54

# input_channels_3 = 1  # Number of input channels
# height_length_3 = 27
# width_length_3 = 33

# output_shape = [[7]]

# x_init = [ torch.randn(input_channels_1, T), # time series evolving in time
#       torch.randn(input_channels_2, T, height_length_2), # profiles evolving in time
#       torch.randn(input_channels_3, T, height_length_3, width_length_3) # images evolving in time
#       ]

# model = MultiBranchCNNModel([arr.shape for arr in x_init], output_shape, D)

# x = [ torch.randn(batch_size, input_channels_1, T), # time series evolving in time
#       torch.randn(batch_size, input_channels_2, T, height_length_2), # profiles evolving in time
#       torch.randn(batch_size, input_channels_3, T, height_length_3, width_length_3) # images evolving in time
# ]

# output = model(x)
