# Planetary Analog Finder: builds data.js (Earth, Moon and Mars grids).
# Run in Google Colab. This is the version that ran successfully on 2026-11-14.
import ee, json, io, time, requests, numpy as np

ee.Authenticate()
ee.Initialize(project='skincare-ai-453321')

STEP = 0.5
PROJ = ee.Projection('EPSG:4326', [STEP, 0, -180, 0, -STEP, 90])
YEAR = ('2023-01-01', '2024-01-01')
COLS = ['slope', 'tmean', 'trange', 'ice', 'arid', 'rock']

def yearly(cid, band, dates=YEAR):
    return ee.ImageCollection(cid).filterDate(*dates).select(band).mean()

def coarse(img):
    return img.reduceResolution(ee.Reducer.mean(), True, 65536).reproject(PROJ)

# Mobility: regional slope (degrees) from the ETOPO1 global relief model
slope = coarse(ee.Terrain.slope(ee.Image('NOAA/NGDC/ETOPO1').select('ice_surface')))

# Thermal: MODIS land surface temperature, day and night (scale 0.02 K)
day = yearly('MODIS/061/MOD11A2', 'LST_Day_1km').multiply(0.02).subtract(273.15)
night = yearly('MODIS/061/MOD11A2', 'LST_Night_1km').multiply(0.02).subtract(273.15)
tmean = day.add(night).divide(2)
trange = day.subtract(night)

# Water: MODIS snow cover (0-1) and aridity from NASA GPM IMERG rainfall
ice = yearly('MODIS/061/MOD10A1', 'NDSI_Snow_Cover').divide(100).unmask(0)
imerg = yearly('NASA/GPM_L3/IMERG_MONTHLY_V06', 'precipitation', ('2020-01-01', '2021-01-01')).multiply(24 * 365)
# IMERG has gaps over ice sheets: fill them with ERA5 (Copernicus) annual precipitation, in mm
era5 = ee.ImageCollection('ECMWF/ERA5/MONTHLY').filterDate('2019-01-01', '2020-01-01').select('total_precipitation').sum().multiply(1000)
rain = imerg.unmask(era5)
arid = rain.divide(1500).min(1).multiply(-1).add(1)

# Geology: share of bare ground from MODIS MCD12C1 (0.05 degree; class 16 barren, class 0 water)
lc = ee.ImageCollection('MODIS/061/MCD12C1').filterDate('2022-01-01', '2023-01-01').first().select('Majority_Land_Cover_Type_1')
rock = coarse(lc.eq(16).unmask(0))
water = coarse(lc.eq(0).unmask(0))

stack = ee.Image.cat([slope, tmean, trange, ice, arid, rock, water]) \
    .rename(COLS + ['water']).toFloat().unmask(-9999)

def tile(w, s, e, n):
    nx, ny = int((e - w) / STEP), int((n - s) / STEP)
    url = stack.getDownloadURL({'region': ee.Geometry.Rectangle([w, s, e, n], None, False),
                                'dimensions': f'{nx}x{ny}', 'crs': 'EPSG:4326', 'format': 'NPY'})
    for attempt in range(4):
        r = requests.get(url, timeout=900)
        if r.ok:
            return np.load(io.BytesIO(r.content))
        time.sleep(15)
    raise RuntimeError(r.text[:400])

# ---- Moon (LOLA) and Mars (MOLA): 16 pixel/degree global elevation from NASA PDS
import os
from google.colab import files

def body(url, dtype, scale, R, c, checks):
    r = requests.get(url, timeout=900)
    print(url.split('/')[-1], r.status_code, len(r.content))
    r.raise_for_status()
    H, W = 2880, 5760
    z = np.frombuffer(r.content, dtype=dtype).astype('f4').reshape(H, W) * scale
    z = np.roll(z, W // 2, axis=1)                       # 0..360E -> -180..180
    lat = (90 - (np.arange(H) + 0.5) / 16)[:, None] * np.pi / 180
    d = np.pi / 180 / 16 * R
    dzdy = np.gradient(z, axis=0) / d
    dzdx = (np.roll(z, -1, 1) - np.roll(z, 1, 1)) / (2 * d * np.maximum(np.cos(lat), 0.002))
    sl = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    h, w = H // c, W // c
    blk = lambda x: x.reshape(h, c, w, c)
    for nm, la, lo in checks:
        print('  ', nm, round(float(z[int((90 - la) * 16), int((lo + 180) * 16)])), 'm')
    S = blk(sl).mean((1, 3)); Rl = blk(z).std((1, 3))
    return {'step': c / 16, 'nlat': h, 'nlon': w, 'elev': blk(z).mean((1, 3)).round(0).astype(int).ravel().tolist(),
            'slope': S.round(2).ravel().tolist(), 'relief': Rl.round(0).astype(int).ravel().tolist()}

PDS = 'https://pds-geosciences.wustl.edu/'
# Moon cells are 2 degrees (about 60 km), Mars cells 1 degree (about 59 km), to match Earth's 55 km cells
moon = body(PDS + 'lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/ldem_16.img', '<i2', 0.5, 1737400, 32,
            [('Mare Imbrium', 32.8, -15.6), ('SPA basin', -53, -169)])
mars = body(PDS + 'mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/megt90n000eb.img', '>i2', 1.0, 3396000, 16,
            [('Olympus Mons', 18.65, -133.8), ('Hellas', -42.4, 70.5)])

# ---- Earth: same grid, plus mean elevation and relief (std of elevation inside each cell)
dem = ee.Image('NOAA/NGDC/ETOPO1').select('ice_surface')
elev = coarse(dem)
relief = dem.reduceResolution(ee.Reducer.stdDev(), True, 65536).reproject(PROJ)
COLS = ['slope', 'tmean', 'trange', 'ice', 'arid', 'rock', 'elev', 'relief']
stack = ee.Image.cat([slope, tmean, trange, ice, arid, rock, elev, relief, water]).rename(COLS + ['water']).toFloat().unmask(-9999)
LA, LO, VV = [], [], []
for n in (90, 45, 0, -45):
    for w in range(-180, 180, 90):
        a = tile(w, n - 45, w + 90, n)
        arr = np.stack([a[k].astype('f4') for k in COLS + ['water']])
        ii, jj = np.nonzero((arr[:8].min(0) > -9000) & (arr[8] < 0.5))
        LA.append(n - (ii + 0.5) * STEP); LO.append(w + (jj + 0.5) * STEP); VV.append(arr[:8][:, ii, jj])
        print('tile', w, n, len(ii))
LA = np.concatenate(LA); LO = np.concatenate(LO); VV = np.concatenate(VV, axis=1)
dec = [2, 1, 1, 2, 2, 2, 0, 0]
earth = {'step': STEP, 'cols': COLS, 'lat': LA.round(2).tolist(), 'lon': LO.round(2).tolist(),
         'v': [[int(x) for x in VV[k].round(0)] if dec[k] == 0 else [round(float(x), dec[k]) for x in VV[k]] for k in range(8)]}
open('data.js', 'w').write('window.ANALOG_DATA=' + json.dumps({'built': '2026-11-14', 'earth': earth, 'moon': moon, 'mars': mars}, separators=(',', ':')) + ';')
print(len(LA), 'Earth cells,', round(os.path.getsize('data.js') / 1e6, 1), 'MB')
files.download('data.js')
