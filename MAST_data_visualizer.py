import zarr
import s3fs
import fsspec
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2

def plot_1d_profile(profile: xr.DataArray, ax=None):
    try:

        profile.plot(x='time', ax=ax)

        if ax is None:
            ax = plt.gca()

        ax.grid('on', alpha=0.5)
        ax.set_xlim(profile.time.min(), profile.time.max())

        plt.show()

    except Exception as e:
        print(f"Error: {e}")


def plot_1d_profiles(profiles: xr.Dataset):
    try:

        """Helper function for plotting 1D profiles"""
        n = int(np.ceil(len(profiles.data_vars) / 2))
        fig, axes = plt.subplots(n, 2, figsize=(10, 2*n))
        axes = axes.flatten()

        for i, name in enumerate(profiles.data_vars.keys()):
            plot_1d_profile(profiles[name], ax=axes[i])

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


def MAST_shot_data(shot_id =30421):
    try:

        endpoint_url = 'https://s3.echo.stfc.ac.uk'
        url = f's3://mast/test/level2/shots/{shot_id}.zarr'

        fs = fsspec.filesystem(
        **dict(
            protocol='simplecache',
            target_protocol="s3",
            target_options=dict(anon=True, endpoint_url=endpoint_url)
        )
        )
        store = zarr.storage.FSStore(fs=fs, url=url)

    except Exception as e:
        print(f"Error: {e}")
        return None

    return store


def plot_current(store):
    try:
        profiles = xr.open_zarr(store, group='summary')
        plot_1d_profile(profiles['plasma_current'])
        profiles['plasma_current']
    except Exception as e:
        print(f"Error: {e}")

def plot_NBI(store):
    try:
        profiles = xr.open_zarr(store, group='summary')
        plot_1d_profile(profiles['total_power'])
        profiles['total_power']
    except Exception as e:
        print(f"Error: {e}")

def plot_magnetic_field(store):
    try:
        profiles = xr.open_zarr(store, group='magnetics')

        fig, axes = plt.subplots(3, 2, figsize=(8, 10))
        axes = axes.flatten()

        for i, name in enumerate(profiles.data_vars.keys()):
            signal = profiles[name]
            for channel in range(signal.shape[0]):
                s = signal.isel({signal.dims[0]: channel})
                s.plot(x='time', ax=axes[i])
            axes[i].set_title(name)

        for ax in axes:
            ax.grid('on', alpha=0.5)
       
        plt.tight_layout()
        plt.show()
       
        profiles
    
    except Exception as e:
        print(f"Error: {e}")


def plot_Dalpha(store):
    try:
        profiles = xr.open_zarr(store, group='dalpha')
        plot_1d_profiles(profiles)
        profiles

    except Exception as e:
        print(f"Error: {e}")


def plot_CXRS(store):
    try:
        profiles = xr.open_zarr(store, group='charge_exchange')

        fig, axes = plt.subplots(2, 1)
        profiles['ti'].plot(x='time', y='major_radius', ax=axes[0], vmax=1000)
        profiles['vi'].plot(x='time', y='major_radius', ax=axes[1], vmax=1000)
        
        plt.tight_layout()
        plt.show()
        
        profiles

    except Exception as e:
        print(f"Error: {e}")


def plot_mirnov_coils(store):
    try:
        profiles = xr.open_zarr(store, group='mirnov')

        plot_1d_profiles(profiles)
        profiles

    except Exception as e:
        print(f"Error: {e}")


def plot_Cameras(store, time_frame):
    try:
        profiles = xr.open_zarr(store, group='cameras')

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        profiles['camera_a'].isel(time=time_frame).plot(ax=axes[0])
        profiles['camera_b'].isel(time=time_frame).plot(ax=axes[1])
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

def movie(store):
    # Define video writer parameters
    output_video_file = "output_video.mp4"
    fps = 10  # Frames per second
    frame_width, frame_height = 1000, 500  # Resolution matching figsize in pixels

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    try:
        profiles = xr.open_zarr(store, group='cameras')

        for t in range(1,len(profiles.camera_a.time),1):
            print(t)
            if t%5!=0:
                continue
            try:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                profiles['camera_a'].isel(time=t).plot(ax=axes[0])
                profiles['camera_b'].isel(time=t).plot(ax=axes[1])
            
                plt.tight_layout()

                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # Resize frame to match video resolution
                frame = cv2.resize(frame, (frame_width, frame_height))

                # Write the frame to the video
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                # Close the figure to free memory
                plt.close(fig)
            except Exception as e:
                print(f"Error {e}")
            
            # Release the video writer
        video_writer.release()

    except Exception as e:
        print(f"Error: {e}")


def plot_Thomson(store):
    try:

        profiles = xr.open_zarr(store, group='thomson_scattering')

        fig, axes = plt.subplots(3, 1)
        axes = axes.flatten()
        profiles.te.plot(x='time', y='major_radius', ax=axes[0])
        profiles.ne.plot(x='time', y='major_radius', ax=axes[1])
        profiles.pe.plot(x='time', y='major_radius', ax=axes[2])
        
        plt.tight_layout()
        plt.show()

        profiles
    
    except Exception as e:
        print(f"Error: {e}")

        
def main():
        try:
            shot_id =30421
            store = MAST_shot_data(shot_id)
            print("Created store variable")
            # plot_current(store)
            # plot_NBI(store)
            # plot_magnetic_field(store)
            # plot_Dalpha(store)
            # plot_Dalpha(store)
            # plot_CXRS(store)
            # plot_Thomson(store)
            # plot_mirnov_coils(store)
            movie(store)


        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()