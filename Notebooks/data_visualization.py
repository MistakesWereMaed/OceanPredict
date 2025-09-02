import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_currents(ds, time_idx=0, cmap='viridis'):
    print(ds.loss[time_idx].values)
    # Extract lat/lon grid
    lat = ds.latitude.values
    lon = ds.longitude.values
    Lon, Lat = np.meshgrid(lon, lat)

    # Extract u,v for predictions and targets (no sample dimension)
    u_pred = ds.predictions[0, time_idx].values
    v_pred = ds.predictions[1, time_idx].values
    u_target = ds.targets[0, time_idx].values
    v_target = ds.targets[1, time_idx].values

    # Compute magnitude
    mag_pred = np.sqrt(u_pred**2 + v_pred**2)
    mag_target = np.sqrt(u_target**2 + v_target**2)

    # Create figure
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), subplot_kw={'projection': ccrs.PlateCarree()}
    )

    for ax, M, title in zip(
        axes,
        [mag_pred, mag_target],
        ["Predicted Currents", "Target Currents"]
    ):
        mag_plot = ax.pcolormesh(Lon, Lat, M, shading='auto', cmap=cmap)
        fig.colorbar(mag_plot, ax=ax, orientation='vertical', label='Current speed')
        ax.coastlines()
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

def plot_height(zos, lat, lon):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    lon, lat = np.meshgrid(lon, lat)
    im = ax.pcolormesh(lon, lat, zos, cmap='Blues')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='dotted')

    plt.title('Sea Level Height')
    plt.show()

def plot_model_metrics_linegraph(pinn_path, gnn_path, fno_path):
    # Load all three CSVs
    df_pinn = pd.read_csv(pinn_path)
    df_gnn = pd.read_csv(gnn_path)
    df_fno = pd.read_csv(fno_path)

    # Check columns
    for df, name in zip([df_pinn, df_fno, df_gnn], ["PINN", "FNO", "GNN"]):
        if not {'gpu_count', 'avg_train_time', 'avg_val_loss'}.issubset(df.columns):
            raise ValueError(f"{name} CSV must contain 'gpu_count', 'avg_train_time', and 'avg_val_loss'.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot: Training Time
    ax1.plot(df_pinn['gpu_count'], df_pinn['avg_train_time'], marker='o', label='PINN')
    ax1.plot(df_fno['gpu_count'], df_fno['avg_train_time'], marker='o', label='FNO')
    ax1.plot(df_gnn['gpu_count'], df_gnn['avg_train_time'], marker='o', label='GNN')
    ax1.set_xlabel("GPU Count")
    ax1.set_ylabel("Training Time (seconds)")
    ax1.set_title("Training Time by GPU Count")
    ax1.legend()
    ax1.grid(True)

    # Plot: Validation Loss
    ax2.plot(df_pinn['gpu_count'], df_pinn['avg_val_loss'], marker='o', label='PINN')
    ax2.plot(df_fno['gpu_count'], df_fno['avg_val_loss'], marker='o', label='FNO')
    ax2.plot(df_gnn['gpu_count'], df_gnn['avg_val_loss'], marker='o', label='GNN')
    ax2.set_xlabel("GPU Count")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss by GPU Count")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_all_loss_histories(pinn_path, gnn_path, fno_path):
    # Load the CSV files
    df_pinn = pd.read_csv(pinn_path)
    df_gnn = pd.read_csv(gnn_path)
    df_fno = pd.read_csv(fno_path)

    # Create a figure with 2 subplots (Training Loss, Validation Loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot Training Loss
    ax1.plot(df_pinn['epoch'] + 1, df_pinn['train_loss'], label='PINN', marker='o')
    ax1.plot(df_fno['epoch'] + 1, df_fno['train_loss'], label='FNO', marker='s')
    ax1.plot(df_gnn['epoch'] + 1, df_gnn['train_loss'], label='GNN', marker='^')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss History')
    ax1.grid()
    ax1.legend()

    # Plot Validation Loss
    ax2.plot(df_pinn['epoch'] + 1, df_pinn['val_loss'], label='PINN', marker='o')
    ax2.plot(df_fno['epoch'] + 1, df_fno['val_loss'], label='FNO', marker='s')
    ax2.plot(df_gnn['epoch'] + 1, df_gnn['val_loss'], label='GNN', marker='^')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss History')
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_all_accuracy_over_time(pinn_path, gnn_path, fno_path):
    # Load NetCDF files using xarray
    ds_pinn = xr.open_dataset(pinn_path)
    ds_gnn = xr.open_dataset(gnn_path)
    ds_fno = xr.open_dataset(fno_path)

    # Extract loss values
    loss_pinn = ds_pinn["loss"].values
    loss_fno = ds_fno["loss"].values
    loss_gnn = ds_gnn["loss"].values

    # Generate x-axis as days (assuming each entry corresponds to a day)
    days = np.arange(1, len(loss_pinn) + 1)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(days, loss_pinn, label="PINN", marker='o', linestyle='-')
    plt.plot(days, loss_fno, label="FNO", marker='s', linestyle='--')
    plt.plot(days, loss_gnn, label="GNN", marker='^', linestyle='-.')
    plt.xlabel("Prediction Lead Time (Days)")
    plt.ylabel("Loss")
    plt.title("Model Loss Over Prediction Lead Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Close datasets
    ds_pinn.close()
    ds_fno.close()
    ds_gnn.close()