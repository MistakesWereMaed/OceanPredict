import os
import json
import shutil
import xarray as xr

import copernicusmarine
import cdsapi

PATH_DATA = "../Data"
PATH_SECRETS = "secrets.json"

LON_MIN = 10
LON_MAX = 38
LAT_MIN = -45
LAT_MAX = -32

def init():
    os.makedirs("./tmp", exist_ok=True)

    with open(PATH_SECRETS, 'r') as f:
        secrets = json.load(f)

    cmems_uname = secrets["CMEMS_UNAME"]
    cmems_pwd = secrets["CMEMS_PWD"]

    return cmems_uname, cmems_pwd

def get_cmems(cmems_uname, cmems_pwd):
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["uo", "vo"],
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime="2024-01-01T00:00:00",
        end_datetime="2024-12-28T00:00:00",
        minimum_depth=0.49402499198913574,
        maximum_depth=0.49402499198913574,
        username=cmems_uname,
        password=cmems_pwd,
        output_filename="ssc",
        output_directory="./tmp"
    )

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["zos"],
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime="2024-01-01T00:00:00",
        end_datetime="2024-12-28T00:00:00",
        minimum_depth=0.49402499198913574,
        maximum_depth=0.49402499198913574,
        username=cmems_uname,
        password=cmems_pwd,
        output_filename="ssh",
        output_directory="./tmp"
    )

def get_era():
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind"
        ],
        "year": ["2024"],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": ["00:00", "12:00", "23:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX]
    }

    client = cdsapi.Client()
    name = client.retrieve(dataset, request).download()

    os.rename(name, "./tmp/ssw.nc")

def process_files():
    ssc = xr.open_dataset("./tmp/ssc.nc")
    ssh = xr.open_dataset("./tmp/ssh.nc")
    ssw = xr.open_dataset("./tmp/ssw.nc")

    ssw = ssw[['u10', 'v10']].resample(valid_time="1D").mean()
    ssw = ssw.rename({'valid_time': 'time'}).drop_vars('number')
    ssw = ssw.interp(
        latitude=ssc.latitude, 
        longitude=ssc.longitude, 
        method = 'nearest'
    )

    ssc_ssh = xr.merge([ssc, ssh])
    full = xr.merge([ssc_ssh, ssw])

    mean = full.mean(skipna=True)
    full = full.fillna(mean).sel(depth=0, method='nearest').drop('depth')

    print(full)

    train = full.sel(time=slice("2024-01-01", "2024-10-31"))
    val = full.sel(time=slice("2024-11-01", "2024-11-30"))
    test = full.sel(time=slice("2024-12-01", "2024-12-31"))

    os.makedirs(PATH_DATA, exist_ok=True)

    train.to_netcdf(f"{PATH_DATA}/train.nc")
    val.to_netcdf(f"{PATH_DATA}/val.nc")
    test.to_netcdf(f"{PATH_DATA}/test.nc")

    shutil.rmtree("./tmp")

def main():
    cmems_uname, cmems_pwd = init()

    get_cmems(cmems_uname, cmems_pwd)
    get_era()

    process_files()

if __name__ == "__main__":
    main()