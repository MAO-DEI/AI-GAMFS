"""Organize inputs or batch-download the supported NASA archives."""

from datetime import datetime, timedelta
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from utils.datasets import ALIASES, PRODUCTS, classify, geos_candidates, require_netcdf4, validate_file


# 编辑这里，然后运行 python download_data.py
DATA_ROOT = Path(__file__).resolve().parent / "temp_asm"
ACTION = "geos-fp"  # geos-fp / merra-2 / organize
DRY_RUN = False    # True 预览，False 下载或移动文件

GEOS_FP = {
    "products": ["tavg3_2d_aer_Nx", "tavg1_2d_flx_Nx", "tavg3_3d_asm_Nv"],
    "start": "20240726",
    "end": "20240802",
    "daily_time": "2230",  # 设为 None 时获取区间内所有原生时次
}

MERRA_2 = {
    "products": ["M2T1NXAER", "M2T1NXFLX", "M2T3NVASM"],
    "start": "20240726",
    "end": "20240802",
    "netrc": None,   # None 使用主目录的 .netrc 或 _netrc，也可填 Path(...)
    "stream": None,  # None 按年份选择，也可指定 "401" 等档案流号
}


def parse_stamp(value, date_only=False):
    formats = ("%Y%m%d", "%Y-%m-%d")
    if not date_only:
        formats += ("%Y%m%d_%H%M", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%MZ")
    for fmt in formats:
        try:
            stamp = datetime.strptime(value, fmt)
        except ValueError:
            continue
        if stamp.strftime(fmt) == value:
            return stamp
    raise ValueError(f"Invalid UTC {'date' if date_only else 'date/time'}: {value}")


def plan(products, start, end, daily_time=None):
    merra = bool(products[0].merra_host)
    date_only = merra or daily_time is not None
    first = parse_stamp(start, date_only)
    last = parse_stamp(end, date_only)
    if not date_only and len(end) in (8, 10):
        last += timedelta(hours=23, minutes=59)
    if last < first:
        raise ValueError("end precedes start")
    if daily_time is not None:
        clock = datetime.strptime(daily_time, "%H%M")
        if clock.strftime("%H%M") != daily_time or clock.minute != 30:
            raise ValueError("daily_time must be a GEOS-FP archive time in HHMM, e.g. 2230")
        for p in products:
            if p.cadence == 3 and clock.hour % 3 != 1:
                raise ValueError(f"{p.name} is not published at {daily_time} UTC")

    day = first.replace(hour=0, minute=0)
    while day <= last:
        for product in products:
            if merra:
                stamps = [day]
            elif daily_time is not None:
                stamps = [day.replace(hour=clock.hour, minute=clock.minute)]
            else:
                first_hour = 1 if product.cadence == 3 else 0
                stamps = [day.replace(hour=h, minute=30)
                          for h in range(first_hour, 24, product.cadence)]
            for stamp in stamps:
                if first <= stamp <= last or daily_time is not None:
                    yield product, stamp
        day += timedelta(days=1)


def merra_stream(stamp):
    for year, stream in ((2011, "400"), (2001, "300"), (1992, "200"), (1980, "100")):
        if stamp.year >= year:
            return stream
    raise ValueError("MERRA-2 starts in 1980")


def move_without_overwrite(source, target):
    if os.name == "nt":
        source.rename(target)
    else:
        target.hardlink_to(source)
        source.unlink()


def fetch(root, product, stamp, stream="400", netrc=None, cookie=None):
    require_netcdf4()
    target = root / product.directory / product.filename(stamp, stream)
    candidates = ([p for p in (target, root / target.name) if p.is_file()]
                  if product.merra_host else geos_candidates(root, product, stamp))
    if len(candidates) > 1:
        raise ValueError(f"Duplicate files; organize them before downloading: {candidates}")
    if candidates:
        try:
            validate_file(candidates[0], product, stamp)
        except (ValueError, OSError, AttributeError, TypeError) as error:
            raise ValueError(f"Existing file failed validation; use ACTION = 'organize' first: {error}") from error
        print(f"EXISTS {candidates[0]}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".download-", dir=target.parent) as workspace:
        partial = Path(workspace) / target.name
        command = ["curl", "-q", "--fail", "--location", "--progress-bar", "--show-error",
                   "--proto", "=https", "--proto-redir", "=https",
                   "--retry", "3", "--connect-timeout", "30"]
        if product.merra_host:
            command += ["--netrc", "--netrc-file", str(netrc),
                        "--cookie", str(cookie), "--cookie-jar", str(cookie)]
        print(f"DOWNLOADING {target.name}", flush=True)
        subprocess.run(command + ["--output", str(partial), product.url(stamp, stream)], check=True)
        validate_file(partial, product, stamp)
        move_without_overwrite(partial, target)
    print(f"SAVED {target}", flush=True)


def organize(root, apply):
    folders = {root, *(root / p.directory for p in PRODUCTS)}
    files = sorted(path for folder in folders if folder.is_dir() for path in folder.iterdir() if path.is_file())
    counts = {"organized": 0, "duplicates": 0, "quarantined": 0, "conflicts": 0}
    planned = set()
    for path in files:
        identified = classify(path.name)
        if identified is None:
            continue
        product, stamp, browser_copy = identified
        group = "organized"
        target = root / product.directory / path.name
        try:
            validate_file(path, product, stamp)
        except (ValueError, OSError, AttributeError, TypeError) as error:
            print(f"INVALID {error}", flush=True)
            group = "quarantined"
            target = root / "_quarantine" / product.directory / path.name
        else:
            if browser_copy:
                group = "duplicates"
                target = root / "_duplicates" / product.directory / path.name
        if path == target:
            continue
        if target.exists() or target in planned:
            print(f"CONFLICT {path} -> {target}", flush=True)
            counts["conflicts"] += 1
            continue
        if not target.resolve().is_relative_to(root) or not path.resolve().is_relative_to(root):
            raise ValueError(f"Dataset path escapes {root}: {path} -> {target}")
        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            move_without_overwrite(path, target)
        planned.add(target)
        counts[group] += 1
        print(f"{'MOVE' if apply else 'PLAN'} {path} -> {target}", flush=True)
    print(counts, flush=True)
    return 1 if counts["conflicts"] else 0


def main():
    root = Path(DATA_ROOT).expanduser().resolve()
    if ACTION == "organize":
        require_netcdf4()
        if not root.is_dir():
            raise ValueError(f"Dataset directory not found: {root}")
        return organize(root, apply=not DRY_RUN)

    settings = {"geos-fp": GEOS_FP, "merra-2": MERRA_2}.get(ACTION)
    if settings is None:
        raise ValueError("ACTION must be 'geos-fp', 'merra-2' or 'organize'")
    products = []
    for name in settings["products"]:
        product = ALIASES.get(name.casefold())
        if product is None or bool(product.merra_host) != (ACTION == "merra-2"):
            raise ValueError(f"Unknown {ACTION} product: {name}")
        if product not in products:
            products.append(product)
    if not products:
        raise ValueError("Configure at least one product")

    override = settings.get("stream")
    if override not in (None, "100", "200", "300", "400", "401"):
        raise ValueError("stream must be None or a string such as '400' or '401'")
    jobs = [(product, stamp, (override or merra_stream(stamp)) if product.merra_host else "400")
            for product, stamp in plan(products, settings["start"], settings["end"], settings.get("daily_time"))]
    if not jobs:
        raise ValueError("No native archive timestamps fall within the requested range")
    if DRY_RUN:
        for product, stamp, stream in jobs:
            target = root / product.directory / product.filename(stamp, stream)
            print(f"{product.url(stamp, stream)} -> {target}")
        print(f"{len(jobs)} scheduled (preview)")
        return 0

    require_netcdf4()
    netrc = None
    if ACTION == "merra-2":
        configured = settings["netrc"]
        netrc = Path(configured).expanduser() if configured else next(
            (p for p in (Path.home() / ".netrc", Path.home() / "_netrc") if p.is_file()), None)
        if netrc is None or not netrc.is_file():
            raise ValueError("MERRA-2 requires an Earthdata credentials file in MERRA_2['netrc'] or ~/.netrc")
    failures = 0
    with TemporaryDirectory(prefix="gamfs-earthdata-") as temp:
        cookie = Path(temp) / "cookies.txt"
        for number, (product, stamp, stream) in enumerate(jobs, start=1):
            print(f"[{number}/{len(jobs)}] {product.name} {stamp:%Y-%m-%d %H:%M} UTC", flush=True)
            try:
                fetch(root, product, stamp, stream, netrc, cookie)
            except (OSError, ValueError, AttributeError, TypeError, subprocess.CalledProcessError) as error:
                failures += 1
                print(f"FAILED {product.name} {stamp:%Y-%m-%d %H:%M}: {error}", flush=True)
    print(f"{len(jobs)} scheduled, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Edit the settings at the top of download_data.py and run it without arguments.")
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
