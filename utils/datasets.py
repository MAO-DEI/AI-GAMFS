"""Archive names, local paths and metadata checks for AI-GAMFS datasets."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys


@dataclass(frozen=True)
class Product:
    name: str
    collection: str
    cadence: int
    markers: tuple[str, ...]
    merra_host: int = 0

    @property
    def directory(self):
        if self.merra_host:
            return Path("MERRA-2", self.name)
        return Path(f"GEOS.fp.asm.{self.collection}")

    def filename(self, stamp, stream="400"):
        if self.merra_host:
            return f"MERRA2_{stream}.{self.collection}.{stamp:%Y%m%d}.nc4"
        return f"{self.directory.name}.{stamp:%Y%m%d_%H%M}.V01.nc4"

    def url(self, stamp, stream="400"):
        if self.merra_host:
            base = f"https://goldsmr{self.merra_host}.gesdisc.eosdis.nasa.gov/data/MERRA2"
            folder = f"{self.name}.5.12.4/{stamp:%Y/%m}"
        else:
            base = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das"
            folder = f"{stamp:Y%Y/M%m/D%d}"
        return f"{base}/{folder}/{self.filename(stamp, stream)}"


PRODUCTS = (
    Product("tavg3_2d_aer_Nx", "tavg3_2d_aer_Nx", 3, ("TOTEXTTAU", "DUEXTTAU", "DUSMASS")),
    Product("tavg1_2d_flx_Nx", "tavg1_2d_flx_Nx", 1, ("QLML", "TLML", "ULML", "VLML", "PRECTOT")),
    Product("tavg3_3d_asm_Nv", "tavg3_3d_asm_Nv", 3, ("QV", "SLP", "T", "U", "V", "lev")),
    Product("M2T1NXAER", "tavg1_2d_aer_Nx", 1, ("TOTEXTTAU", "DUEXTTAU", "DUSMASS"), 4),
    Product("M2T1NXFLX", "tavg1_2d_flx_Nx", 1, ("PRECTOT", "EVAP"), 4),
    Product("M2T3NVASM", "tavg3_3d_asm_Nv", 3, ("QV", "T", "U", "V", "lev"), 5),
)
ALIASES = {p.name.casefold(): p for p in PRODUCTS}
ALIASES.update({p.directory.name.casefold(): p for p in PRODUCTS if not p.merra_host})
GEOS_NAME = re.compile(
    r"GEOS\.fp\.asm\.(?P<collection>[^.]+)\.(?P<stamp>\d{8}_\d{4})\.V\d{2}"
    r"(?P<copy> \(\d+\))?\.(?:hdf|nc4|nc)", re.IGNORECASE
)
MERRA_NAME = re.compile(
    r"MERRA2_\d{3}\.(?P<collection>[^.]+)\.(?P<stamp>\d{8})(?P<copy> \(\d+\))?\.nc4",
    re.IGNORECASE,
)


def classify(name):
    """Return product, timestamp and browser-copy flag for a supported filename."""
    for merra, pattern in ((False, GEOS_NAME), (True, MERRA_NAME)):
        match = pattern.fullmatch(name)
        if match:
            collection = match["collection"].casefold()
            for product in PRODUCTS:
                if bool(product.merra_host) == merra and product.collection.casefold() == collection:
                    try:
                        stamp = datetime.strptime(match["stamp"], "%Y%m%d" if merra else "%Y%m%d_%H%M")
                    except ValueError:
                        return None
                    return product, stamp, bool(match["copy"])
    return None


def geos_candidates(root, product, stamp):
    paths = set()
    for folder in (root, root / product.directory):
        if folder.is_dir():
            for path in folder.iterdir():
                if path.is_file() and classify(path.name) == (product, stamp, False):
                    paths.add(path.resolve())
    return sorted(paths)


def reference_path(directory, name):
    path = directory / name
    if Path(name).parent != Path("."):
        return path
    identified = classify(path.name)
    if identified and identified[0].merra_host:
        preferred = directory / identified[0].directory / path.name
        if preferred.is_file() and path.is_file() and preferred.resolve() != path.resolve():
            raise ValueError(f"Duplicate reference file: {preferred} and {path}")
        if preferred.is_file() or not path.is_file():
            return preferred
    return path


def require_netcdf4():
    try:
        import netCDF4
    except ImportError as error:
        raise ValueError(
            f"Cannot import netCDF4 with {sys.executable}: {error}. "
            "Run download_data.py using the project's Python environment (e.g. conda activate gamfs)."
        ) from error
    return netCDF4


def validate_file(path, product, stamp):
    """Read NetCDF metadata only. Reject wrong products, timestamps and partial day files."""
    netCDF4 = require_netcdf4()

    with netCDF4.Dataset(path) as ds:
        missing = (set(product.markers) | {"time", "lat", "lon"}) - ds.variables.keys()
        if missing:
            raise ValueError(f"{path.name}: missing {', '.join(sorted(missing))}")
        clock = ds.variables["time"]
        # NASA's time metadata can carry a valid_range outside its integer dtype.
        clock.set_auto_mask(False)
        times = netCDF4.num2date(clock[:], clock.units, getattr(clock, "calendar", "standard"),
                                 only_use_cftime_datetimes=False)
        actual = [t.strftime("%Y%m%d_%H%M%S") for t in times]
        if product.merra_host:
            first_hour = 0 if product.cadence == 1 else 1
            expected = [f"{stamp:%Y%m%d}_{hour:02d}3000"
                        for hour in range(first_hour, 24, product.cadence)]
        else:
            expected = [stamp.strftime("%Y%m%d_%H%M%S")]
        if actual != expected:
            raise ValueError(f"{path.name}: unexpected time axis: {actual}; expected {expected}")
