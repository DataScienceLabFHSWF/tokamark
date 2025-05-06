from typing import Literal
import zarr
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cv2


def list_all_shots(location: Literal["local", "singularity", "S3"] = "local"):
    """Get a list of all shot indices.
    
    Parameters
    ----------
    location : str
        Where to search for the data.
        - If "local", tries to get data from
          /rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp/level2/
          which is the main CSD3 path.
        - If "singularity", tries to get data from "/srv".
        - If "S3", tries to get data from
          s3://mast/level2/shots/
          at the endpoint
          https://s3.echo.stfc.ac.uk

    Returns
    -------
    list
        List of shot ID integers.
    """
    location = "singularity"
    if location == "local":
        fs = fsspec.filesystem("file")
        all_files = fs.ls("/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp/level2/")
    elif location == "singularity":
            fs = fsspec.filesystem("file")
            all_files = fs.ls("/srv")
    elif location == "S3":
        fs = fsspec.filesystem(
            **dict(
                protocol="s3",
                anon=True,
                endpoint_url="https://s3.echo.stfc.ac.uk"
            )
        )
        all_files = fs.ls("s3://mast/level2/shots/")
    else:
        raise ValueError(f"Unknown location: {location}")

    shot_ids = [file.split("/")[-1].replace(".zarr", "") for file in all_files]
    shot_ids = [int(shot_id) for shot_id in shot_ids if shot_id.isdigit()]
    
    return shot_ids

def list_all_sources(URL = 'https://mastapp.site'):
    sources_df = pd.read_parquet(f"{URL}/parquet/level2/sources")
    return sources_df

def list_signals_per_shot(shot_id, URL = 'https://mastapp.site'):
    signals_df = pd.read_parquet(f"{URL}/parquet/level2/signals?shot_id={shot_id}")
    return signals_df

def make_store(shot_id, location: Literal["local", "singularity", "S3"] = "local"):
    """Create a filesystem store given a shot id.
    
    Parameters
    ----------
    shot_id : int
    location : str
        Where to search for the data.
        - If "local", tries to get data from
          /rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp/level2/
          which is the main CSD3 path.
        - If "singularity", tries to get data from "/srv".
        - If "S3", tries to get data from
          s3://mast/level2/shots/
          at the endpoint
          https://s3.echo.stfc.ac.uk

    Returns
    -------
    store
        FSStore with MAST data.
    """
    location = "singularity"
    if location == "S3":
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
            raise e
    elif location == "local":
        local_url = f"/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp/level2/{shot_id}.zarr"
        store = zarr.storage.FSStore(url=local_url)
    elif location == "singularity":
        local_url = f"/srv/{shot_id}.zarr"
        store = zarr.storage.FSStore(url=local_url)
    else:
        raise ValueError(f"Unknown location: {location}")
    
    return store

def print_store(store):
    """
    Print all signals, attributes and methods in the store
    """
    print("All signals (keys) in the store")
    for item in store:
        print(item)

    print(f"Attributes and methods of store: {dir(store)}")


def is_signal_in_store(store, signal:str):
    """
    Tells if an signal is available in the store. 
    """
    availability = False

    store_keys = list(store.keys())

    for key in store_keys:
        if signal in key:
            print(f"We found the following signal: {key}")
            availability = True
        
    return availability


def print_signals_in_group(store, group):
    """
    Shows all signals in the group. 
    """
    [print(key) for key in store.keys() if key.startswith(group)]


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
        plt.savefig(f"{xr.Dataset}.png") 
        plt.close()

    except Exception as e:
        print(f"Error: {e}")


def plot_plasma_current(store):
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
        plt.savefig("Magnetics.png")
       
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


def plot_signal_in_store(group, signal, store):
    try:
        profiles = xr.open_zarr(store, group=group)
        plot_1d_profiles(profiles[signal])
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


def plot_magnetics(store):
    profiles = xr.open_zarr(store, group='magnetics')

    fig, axes = plt.subplots(3, 2)
    axes = axes.flatten()

    # Flux loops
    profiles['flux_loop_flux'].plot.line(ax=axes[0], x='time', add_legend=False)
    axes[0].set_title('Flux Loops', fontsize='small')


    # Saddle coils
    profiles['b_field_tor_probe_saddle_voltage'].plot.line(ax=axes[1], x='time_saddle', add_legend=False)
    axes[1].set_title('Saddle Coils', fontsize='small')


    # Pickup coils - centre column poloidal
    profiles['b_field_pol_probe_ccbv_field'].plot.line(ax=axes[2], x='time', add_legend=False)
    axes[2].set_title('Pick Up Coils - Centre Column', fontsize='small')

    # Pickup coils - outboard
    profiles['b_field_pol_probe_obr_field'].plot.line(ax=axes[3], x='time', add_legend=False)
    axes[3].set_title('Pick Up Coils - Outboard', fontsize='small')

    profiles['b_field_pol_probe_obv_field'].plot.line(ax=axes[4], x='time', add_legend=False)
    axes[4].set_title('Pick Up Coils - Outboard', fontsize='small')

    plt.tight_layout()  
    plt.savefig("magnetics.png")

def main():
    try:
      
        shot_id =30421
        store = make_store(shot_id)
        # plot_plasma_current(store)
        # plot_NBI(store)
        # plot_magnetic_field(store)
        # plot_Dalpha(store)
        # plot_Dalpha(store)
        # plot_CXRS(store)
        # plot_Thomson(store)
        # plot_mirnov_coils(store)
        # movie(store)
        plot_magnetics(store)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()