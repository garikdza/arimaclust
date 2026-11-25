"""
McCracken and Ng's FRED-MD, a monthly panel of US macroeconomic series, prepared for clustering.

A fixed vintage is downloaded once into a cache, each series is made stationary with the transformation code FRED-MD
publishes for it, and the series are cut to a fixed window. No matplotlib.
"""

import http.client
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

VINTAGE_URL = 'https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2026-01-md.csv'
# the series are cut to these months, both included, after they are transformed
START, END = '1960-01', '2019-12'

GROUP_NAMES: dict[int, str] = {
    1: 'output and income',
    2: 'labour market',
    3: 'housing',
    4: 'consumption, orders and inventories',
    5: 'money and credit',
    6: 'interest and exchange rates',
    7: 'prices',
    8: 'stock market',
}
# the series of each group kept in the window, from the FRED-MD appendix (FRED-MD_updated_appendix.csv) in the
# vintage's spelling and order; hard-coded because the appendix's download address carries a hash. Only the stock
# market names have spaces, so the other groups are written as space-separated names
GROUP_SERIES: dict[int, tuple[str, ...]] = {
    1: tuple(
        'RPI W875RX1 INDPRO IPFPNSS IPFINAL IPCONGD IPDCONGD IPNCONGD IPBUSEQ IPMAT IPDMAT IPNMAT '
        'IPMANSICS IPB51222S IPFUELS CUMFNS'.split()
    ),
    2: tuple(
        'HWI HWIURATIO CLF16OV CE16OV UNRATE UEMPMEAN UEMPLT5 UEMP5TO14 UEMP15OV UEMP15T26 UEMP27OV '
        'CLAIMSx PAYEMS USGOOD CES1021000001 USCONS MANEMP DMANEMP NDMANEMP SRVPRD USTPU USWTRADE '
        'USTRADE USFIRE USGOVT CES0600000007 AWOTMAN AWHMAN CES0600000008 CES2000000008 CES3000000008'.split()
    ),
    3: tuple('HOUST HOUSTNE HOUSTMW HOUSTS HOUSTW PERMIT PERMITNE PERMITMW PERMITS PERMITW'.split()),
    4: tuple('DPCERA3M086SBEA CMRMTSPLx RETAILx AMDMNOx AMDMUOx BUSINVx ISRATIOx'.split()),
    5: tuple(
        'M1SL M2SL M2REAL BOGMBASE TOTRESNS NONBORRES BUSLOANS REALLN NONREVSL CONSPI DTCOLNVHFNM '
        'DTCTHFNM INVEST'.split()
    ),
    6: tuple(
        'FEDFUNDS CP3Mx TB3MS TB6MS GS1 GS5 GS10 AAA BAA COMPAPFFx TB3SMFFM TB6SMFFM T1YFFM T5YFFM '
        'T10YFFM AAAFFM BAAFFM EXSZUSx EXJPUSx EXUSUKx EXCAUSx'.split()
    ),
    7: tuple(
        'WPSFD49207 WPSFD49502 WPSID61 WPSID62 OILPRICEx PPICMM CPIAUCSL CPIAPPSL CPITRNSL CPIMEDSL '
        'CUSR0000SAC CUSR0000SAD CUSR0000SAS CPIULFSL CUSR0000SA0L2 CUSR0000SA0L5 PCEPI DDURRG3M086SBEA '
        'DNDGRG3M086SBEA DSERRG3M086SBEA'.split()
    ),
    8: ('S&P 500', 'S&P div yield', 'S&P PE ratio'),
}


def fetch_vintage(path: Path, url: str = VINTAGE_URL) -> None:
    """
    Download a FRED-MD vintage to ``path``, unless a file is already there.

    Parameters
    ----------
    path : Path
        where the vintage's CSV file is cached; its folder is created
    url : str
        address of the vintage

    Raises
    ------
    RuntimeError
        if the download fails or what it gets is not a FRED-MD vintage (a moved file is answered with an HTML page);
        no file is left at ``path``
    """
    if path.exists():
        print(f'using cached {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    # URLError and TimeoutError are OSErrors, as is a connection reset during the read; a truncated body is an
    # HTTPException
    except (OSError, http.client.HTTPException) as error:
        raise RuntimeError(
            f'could not download the FRED-MD vintage from {url}: {error}; check the network connection, or whether '
            'the St. Louis Fed has moved the file, and run again'
        ) from error
    # a vintage starts with its sasdate header; for a path it no longer serves, the host answers 200 with an HTML page
    if not data.startswith(b'sasdate'):
        raise RuntimeError(
            f'{url} does not serve a FRED-MD vintage (it answered with something other than the CSV, such as a web '
            'page); the St. Louis Fed may have moved the file, so find its current address on the FRED-MD page and '
            'set it as VINTAGE_URL in fred_md.py'
        )
    # write beside it and swap in, so an interrupted write leaves no file the next run would take as the cache
    part = path.with_suffix('.part')
    try:
        part.write_bytes(data)
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    print(f'downloaded {path}')


def read_vintage(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Read a FRED-MD vintage: a CSV with a ``sasdate`` column, a ``Transform:`` row of codes, then one row per month.

    Parameters
    ----------
    path : Path
        the vintage's CSV file

    Returns
    -------
    data : pd.DataFrame
        the series in levels, one column per series in the file's order, indexed by the first day of each month;
        NaN where the vintage has no value
    tcodes : pd.Series
        the transformation code of each series, indexed by series name

    Raises
    ------
    ValueError
        if the file is not a FRED-MD vintage, for example an HTML page saved in its place
    """
    try:
        raw = pd.read_csv(path)
    except (ValueError, UnicodeDecodeError) as error:
        raise _not_a_vintage(path, f'the CSV does not parse: {error}') from error
    if raw.columns[0] != 'sasdate':
        raise _not_a_vintage(path, 'its first column is not sasdate')
    if raw.empty or raw.iloc[0, 0] != 'Transform:':
        raise _not_a_vintage(path, 'its second row is not the Transform: row')
    try:
        tcodes = raw.iloc[0, 1:].astype(float)
    except ValueError as error:
        raise _not_a_vintage(path, f'a transformation code is not a number: {error}') from error
    if not (tcodes % 1 == 0).all():
        raise _not_a_vintage(path, 'a transformation code is missing or not an integer')
    # a row of empty cells, such as a trailing one, has no date
    data = raw.iloc[1:].dropna(subset=['sasdate']).set_index('sasdate')
    try:
        data.index = pd.to_datetime(data.index, format='%m/%d/%Y')
    except ValueError as error:
        raise _not_a_vintage(path, f'a date is not m/d/YYYY: {error}') from error
    return data.astype(float), tcodes.astype(int)


def _not_a_vintage(path: Path, reason: str) -> ValueError:
    return ValueError(f'{path} is not a FRED-MD vintage ({reason}); delete it to download it again')


def transform(x: pd.Series, tcode: int) -> pd.Series:
    """
    Make a series stationary with its FRED-MD transformation code.

    The codes are McCracken and Ng's (2016, table in the appendix): 1 the level x, 2 the difference Δx, 3 the second
    difference Δ²x, 4 the log, 5 the log difference Δlog x, 6 the second log difference Δ²log x, and 7 the difference
    of the growth rate, Δ(x_t / x_{t−1} − 1).

    Parameters
    ----------
    x : pd.Series
        the series in levels
    tcode : int
        its transformation code, 1 to 7

    Returns
    -------
    transformed : pd.Series
        the transformed series on the same index, NaN for the first values that a difference needs

    Raises
    ------
    ValueError
        if ``tcode`` is not one of the seven codes
    """
    if tcode == 1:
        return x
    if tcode == 2:
        return x.diff()
    if tcode == 3:
        return x.diff().diff()
    if tcode == 4:
        return np.log(x)
    if tcode == 5:
        return np.log(x).diff()
    if tcode == 6:
        return np.log(x).diff().diff()
    if tcode == 7:
        return x.pct_change(fill_method=None).diff()
    raise ValueError(f'unknown FRED-MD transformation code {tcode}; expected 1 to 7')


def prepare(
    data: pd.DataFrame, tcodes: pd.Series, start: str = START, end: str = END
) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform every series, cut them to a window and drop those with a gap in it.

    The series are transformed before they are cut, so a differenced series' first month in the window is computed
    from the months before it.

    Parameters
    ----------
    data : pd.DataFrame
        the series in levels, as ``read_vintage`` returns them
    tcodes : pd.Series
        the transformation code of each series
    start, end : str
        first and last month of the window, both included

    Returns
    -------
    series : pd.DataFrame
        the transformed series with a value at every month of the window, in the order of ``data``
    dropped : list[str]
        the series with a missing value in the window, in the order of ``data``
    """
    transformed = pd.DataFrame({name: transform(data[name], tcodes[name]) for name in data.columns})
    window = transformed.loc[start:end]
    gaps = window.isna().any()
    return window.loc[:, ~gaps], list(window.columns[gaps])


def group_ids(names: Iterable[str]) -> np.ndarray:
    """
    Look up the FRED-MD group of each series.

    Parameters
    ----------
    names : Iterable[str]
        series names as the vintage spells them

    Returns
    -------
    groups : np.ndarray
        the group of each series in order, a key of ``GROUP_NAMES``

    Raises
    ------
    ValueError
        if a series is in no group, naming every such series
    """
    group_of = {name: group for group, group_names in GROUP_SERIES.items() for name in group_names}
    names = list(names)
    missing = [name for name in names if name not in group_of]
    if missing:
        raise ValueError(f'no FRED-MD group for {missing}; add it to GROUP_SERIES')
    return np.array([group_of[name] for name in names])
