"""
ERT PROCESSING
it gets arguments from cmd-line and/or dictionary (e.g., ert manager script)
it uses a ERT processing class that delegates to two dataframes for data and elec tables
it filters the data based on args; set very loose args values to avoid filtering.
"""

import os
import argparse
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from .symlog_utils import MinorSymLogLocator
from .symlog_utils import find_threshold_minnonzero
from .symlog_utils import find_best_yscale
from matplotlib.ticker import SymmetricalLogLocator

try:
    from numba import jit
except ImportError:
    numba_opt = False
else:
    numba_opt = True


def get_cmd():
    """ get command line arguments for data processing """
    print('parsing arguments')
    parse = argparse.ArgumentParser()
    main = parse.add_argument_group('main')
    filters = parse.add_argument_group('filters')
    adjustments = parse.add_argument_group('adjustments')
    outputs = parse.add_argument_group('output')
    # MAIN
    main.add_argument('-fName', type=str, help='data file to process', nargs='+')
    main.add_argument('-fType', type=str, help='data type to process', default='labrecque')
    main.add_argument('-dir_proc', type=str, help='output directory', default='processing')
    # FILTERS
    filters.add_argument('-ctc', type=float, default=1E+5, help='max ctc, ohm')
    filters.add_argument('-stk', type=float, default=20, help='max stacking err, pct')
    filters.add_argument('-v', type=float, default=1E-5, help='min voltage, V')
    filters.add_argument('-rec', type=float, default=5, help='max reciprocal err, pct')
    filters.add_argument('-rec_couple', action='store_true', default=True, help='couple reciprocals')
    filters.add_argument('-rec_unpaired', action='store_true', default=True, help='keep unpaired')
    filters.add_argument('-k', type=float, default=1E+6, help='max geometrical factor, m')
    filters.add_argument('-k_file', type=str, help='file with geom factors and activates k and rhoa')
    filters.add_argument('-rhoa', default=[0, 1E+5], type=float, help='rhoa (min, max)', nargs='+')
    filters.add_argument('-badelec', type=int, help='electrodes to remove', nargs='+')
    filters.add_argument('-dist_amb', action='store_true', default=False, help='check if m is closer to a than b')
    # OUTPUT
    outputs.add_argument('-w_rhoa', action='store_true', help='if true, write rhoa')
    outputs.add_argument('-w_ip', action='store_true', help='if true, write phase')
    outputs.add_argument('-w_err', type=str, default=None, help='error to write')
    outputs.add_argument('-plot', action='store_true', default=True, help='plot data quality figures')
    outputs.add_argument('-odf', type=str, default='bert', help='output data format', choices=['bert', 'ubc'])
    # ADJUSTMENTS
    adjustments.add_argument('-s_abmn', type=int, default=0, help='shift abmn')
    adjustments.add_argument('-s_meas', type=int, default=0, help='shift measurement number')
    adjustments.add_argument('-s_elec', type=int, default=0, help='shift electrode number')
    adjustments.add_argument('-f_elec', type=str, default=None, help='electrode coordinates: x y z')
    # GET ARGS
    args = parse.parse_args()
    return(args)


def update_args(cmd_args, dict_args):
    """ update cmd-line args with args from dict """
    args = get_cmd()
    for key, val in dict_args.items():
        if not hasattr(args, key):
            raise AttributeError('unrecognized option: ', key)
        else:
            setattr(args, key, val)
    return(args)


def check_args(args):
    """ check consistency of args """
    if isinstance(args.fName, str):
        args.fName = [args.fName]
    if (args.k_file is None and args.w_rhoa is True):
        error = """cannot calculate and write rhoa without k_file with geometric factors,
                this wont break the code (default rhoa=None) but it may be and argument err"""
        raise ValueError(error)
    return(args)


def output_file(old_fname, new_ext='.dat', directory='.'):
    """ return name for the output file and clean them if already exist """
    f, old_ext = os.path.splitext(old_fname)
    new_fname = f + new_ext
    new_dfname = os.path.join(directory, new_fname)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    elif os.path.exists(new_dfname):
        os.remove(new_dfname)
    return(new_dfname)


def badlines_function(bl):
    print(bl)
    return(None)


def read_labrecque(f):
    """
    Read a labrecque data file an return data and electrode dataframes.
    """
    FreDom = False
    AppRes = False
    TW = 0
    SP = 0
    with open(f) as fid:
        enumerated_lines = enumerate(fid)
        i, l = next(enumerated_lines)
        while 'elec_start' not in l:
            if 'FStcks' in l:
                FreDom = True
            elif '#SAprs' in l:
                AppRes = True
            elif '#TW' in l:
                TW += 2  # for each window, adds IP Window n and its associated Std
            elif l.startswith('#SCltSP'):
                SP = 1
            i, l = next(enumerated_lines)
        es = i + 1
        while 'elec_end' not in l:
            i, l = next(enumerated_lines)
        ee = i - 1
        while 'data_start' not in l:
            i, l = next(enumerated_lines)
        ds = i + 3
        while 'data_end' not in l:
            i, l = next(enumerated_lines)
        de = i
    print('FreDom: ', FreDom, '    AppRes: ', AppRes, '    TW: ', TW, '    SP: ', SP)
    # DATA without using header because it is too inconsistent
    col_name_num = {'meas': 0, 'a': 2, 'b': 4, 'm': 6, 'n': 8}
    col_name_num.update({'r': 9, 'stk': 10, 'v': 11, 'curr': 13 + TW + SP, 'ctc': 14 + TW + SP, 'datetime': 15 + TW + SP})
    if (FreDom) and (not AppRes):
        col_name_num.update({'r': 9, 'ip': 10, 'v': 13, 'stk': 14, 'curr': 17, 'ctc': 20, 'datetime': 21 + SP})
    elif (not FreDom) and (AppRes):
        # col_name_num.update({'r': 10, 'stk': 11, 'v': 12, 'ctc': 16 + TW + SP, 'datetime': 17 + TW + SP})
        col_name_num.update({'r': 10, 'stk': 11, 'v': 12, 'curr':14 + TW + SP, 'ctc': 15 + TW + SP, 'datetime': 16 + TW + SP})
    elif (FreDom) and (AppRes):
        col_name_num.update({'r': 10, 'ip': 11, 'v': 14, 'stk': 15, 'curr': 18, 'ctc': 21, 'datetime': 22 + SP})
    col_names_dtypes = {
        'meas': 'Int16', 'a': 'Int16', 'b': 'Int16', 'm': 'Int16', 'n': 'Int16',
        'r': float, 'ip': float, 'v': float, 'curr': float, 'ctc': float, 'stk': float, 'datetime': 'datetime64[ns]'
    }
    (col_nums, col_names, col_dtypes) = zip(
        *[(v, k, col_names_dtypes[k]) for k, v in sorted(col_name_num.items(), key=lambda item: item[1])]
    )
    dn = de - ds
    sep = r"\s+|,"
    error_strings = ['*', 'TX', 'Resist.', 'out', 'of', 'range', 'Error_Zero_Current', 'Raw_Voltages:', 'Run', 'Complete']
    print(col_name_num)
    data = pd.read_csv(
        f,
        header=None,
        index_col=False,
        nrows=dn,
        skiprows=ds,
        usecols=col_nums,
        names=col_names,
        dtype=col_names_dtypes,
        na_values=error_strings,
        parse_dates=['datetime'],
        date_parser=lambda d: pd.to_datetime(d, format="%Y%m%d_%H%M%S", errors="coerce"),
        sep=sep,
        engine='python',
        on_bad_lines='warn',
        # comment='*',
    )
    # print('data set:\n{}'.format(data))
    invalid_data = data.loc[data['r'].isna()]
 #   if not invalid_data.empty:
 #       print('\n!!! found invalid data\n', invalid_data)
 #       data = data.drop(index=invalid_data.index)
  #      data = data.reset_index()
#        print('data set:\n{}'.format(data))
    if not FreDom:
        data['ip'] = 1
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y%m%d_%H%M%S')
    data = data.astype(col_names_dtypes)
    data['stk'] = data['stk'] / np.abs(data['v']) * 100
    # elec using headers
    ec = {'El#': 'num', 'Elec-X': 'x', 'Elec-Y': 'y', 'Elec-Z': 'z'}
    et = {'num': 'Int16', 'x': float, 'y': float, 'z': float}
    en = ee - es
    elec = pd.read_csv(
        f,
        skiprows=es,
        usecols=list(ec),
        nrows=en,
        header=0,
        sep=r',|\s+',
        index_col=False,
        engine='python',
    )
    elec = elec.rename(columns=ec)
    elec = elec.astype(et)
    # print('electrodes:\n{}'.format(elec))
    return(elec, data)


def read_bert(k_file=None):
    """read bert-type file and return elec and data"""
    with open(k_file) as fid:
        lines = fid.readlines()
    elec_num = int(lines[0])
    data_num = int(lines[elec_num + 2])
    elec_raw = pd.read_csv(k_file, delim_whitespace=True, skiprows=1, nrows=elec_num, header=None)
    elec = elec_raw[elec_raw.columns[:-1]]
    elec.columns = elec_raw.columns[1:]
    data_raw = pd.read_csv(k_file, delim_whitespace=True, skiprows=elec_num + 3, nrows=data_num)
    data = data_raw[data_raw.columns[:-1]]
    data.columns = data_raw.columns[1:]
    return(elec, data)


def read_res2dinv_gen(f):
    # read info lines
    lines_descriptions = {0: 'name', 1: 'spacing', 5: 'type', 6: 'num_meas'}
    file_dict = {}
    with open(f) as fin:
        for fin_ind, fin_line in enumerate(fin):
            fin_line = fin_line.strip()
            if fin_ind in lines_descriptions.keys():
                file_dict[lines_descriptions[fin_ind]] = fin_line
            if fin_ind == 9:
                break
    file_dict['spacing'] = float(file_dict['spacing'])
    file_dict['type'] = int(file_dict['type'])
    file_dict['num_meas'] = int(file_dict['num_meas'])

    # read data
    data_skiprows = 9
    data = pd.read_csv(
        f,
        header=None,
        delim_whitespace=True,
        skiprows=data_skiprows,
        nrows=file_dict['num_meas'],
        usecols=[1, 3, 5, 7, 9],
        names=['a', 'b', 'm', 'n', 'rhoa'],
    )
    print(data)
    # find unique coordinates
    unique_a = data['a'].unique()
    unique_b = data['b'].unique()
    unique_m = data['m'].unique()
    unique_n = data['n'].unique()
    unique = sorted(set([*unique_a, *unique_b, *unique_m, *unique_n]))
    # find number of unique coordinates and thus electrodes
    num_unique = len(unique)
    # init elec df
    elec_num = np.arange(1, num_unique + 1, dtype=np.int16)
    elec_x = np.array(unique)
    zeros = np.zeros_like(elec_x)
    elec = pd.DataFrame(data={'num': elec_num, 'x': elec_x, 'y': zeros, 'z': zeros})
    print(elec)
    # remap data based on elec numbering and coordinates
    elec_dict = elec.set_index('x').to_dict()['num']
    print(elec_dict)
    map_list = ['a', 'b', 'm', 'n']
    for column in map_list:
        data[column] = data[column].map(elec_dict)
    print(data)
    return(elec, data)


def fun_rec(a: np.ndarray, b: np.ndarray, m: np.ndarray, n: np.ndarray, x: np.ndarray):
    len_sequence = int(len(x))
    rec_num = np.zeros_like(x, dtype=np.int64)
    rec_avg = np.zeros_like(x, dtype=np.float64)
    rec_err = np.zeros_like(x, dtype=np.float64)
    rec_fnd = np.zeros_like(x, dtype=np.int64)
    for i in range(len_sequence):
        if rec_num[i] != 0:
            continue
        for j in range(i + 1, len_sequence):
            if (a[i] == m[j] and b[i] == n[j] and m[i] == a[j] and n[i] == b[j]):
                avg = (x[i] + x[j]) / 2
                err = abs(x[i] - x[j]) / abs(avg) * 100
                rec_num[i] = j + 1
                rec_num[j] = i + 1
                rec_avg[i] = avg
                rec_avg[j] = avg
                rec_err[i] = err
                rec_err[j] = err
                rec_fnd[i] = 1  # mark meas as direct
                rec_fnd[j] = 2  # mark meas as reciprocal (keep 0 for unpaired)
                break
    return(rec_num, rec_avg, rec_err, rec_fnd)


if numba_opt:
    s = 'Tuple((int64[:],float64[:],float64[:],int64[:]))(int64[:],int64[:],int64[:],int64[:],float64[:])'
    fun_rec = jit(signature_or_function=s, nopython=True,
                  parallel=False, cache=True, fastmath=True, nogil=True)(fun_rec)


class ERTdataset():
    """ A dataset class composed of two dataframes data and elec.
    delegation to pandas dataframes is use for data and elec tables """

    data_headers = [
        'meas', 'a', 'b', 'm', 'n',
        'r', 'k', 'rhoa', 'ip',
        'v', 'curr', 'ctc', 'stk', 'datetime',
        'rec_num', 'rec_fnd', 'rec_avg', 'rec_err',
        'rec_ip_avg', 'rec_ip_err',
        'rec_valid', 'k_valid', 'rhoa_valid', 'v_valid',
        'ctc_valid', 'stk_valid', 'elec_valid', 'valid',
    ]
    data_dtypes = {
        'meas': 'Int16', 'a': 'Int16', 'b': 'Int16', 'm': 'Int16', 'n': 'Int16',
        'r': float, 'k': float, 'rhoa': float, 'ip': float,
        'v': float, 'curr': float, 'ctc': float, 'stk': float, 'datetime': 'datetime64[ns]',
        'rec_num': 'Int16', 'rec_fnd': 'Int16', 'rec_avg': float, 'rec_err': float,
        'rec_ip_avg': float, 'rec_ip_err': float,
        'rec_valid': bool, 'k_valid': bool, 'rhoa_valid': bool, 'v_valid': bool,
        'ctc_valid': bool, 'stk_valid': bool, 'elec_valid': bool, 'valid': bool,
    }
    elec_headers = ['num', 'x', 'y', 'z']
    elec_dtypes = {'num': 'Int16', 'x': float, 'y': float, 'z': float}

    def __init__(self, data=None, elec=None):
        self.data = None
        self.elec = None

        if data is not None:
            self.init_EmptyData(data_len=len(data))
            self.data.update(data)
            self.data = self.data.astype(self.data_dtypes)

        if elec is not None:
            self.init_EmptyElec(elec_len=len(elec))
            self.elec.update(elec)
            self.elec = self.elec.astype(self.elec_dtypes)

    def init_EmptyData(self, data_len=None):
        """ wrapper to create empty (None) data dataframe with the proper headers and datatypes."""
        self.data = pd.DataFrame(None, index=range(data_len), columns=self.data_headers)
        self.data = self.data.astype(self.data_dtypes)

    def init_EmptyElec(self, elec_len=None):
        """ wrapper to create empty (None) data dataframe with the proper headers and datatypes."""
        self.elec = pd.DataFrame(None, index=range(elec_len), columns=self.elec_headers)
        self.elec = self.elec.astype(self.elec_dtypes)

    def default_types(self):
        self.data = self.data.astype(self.data_dtypes)

    def process_rec(self, fun_rec=fun_rec, x='r', x_avg='rec_avg', x_err='rec_err'):
        a = self.data['a'].to_numpy(dtype=np.int64)
        b = self.data['b'].to_numpy(dtype=np.int64)
        m = self.data['m'].to_numpy(dtype=np.int64)
        n = self.data['n'].to_numpy(dtype=np.int64)
        x = self.data[x].to_numpy(dtype=np.float64)
        rec_num, rec_avg, rec_err, rec_fnd = fun_rec(a, b, m, n, x)
        self.data['rec_num'] = rec_num
        self.data['rec_fnd'] = rec_fnd
        self.data[x_avg] = rec_avg
        self.data[x_err] = rec_err

    def get_k(self, data_k):
        if len(self.data) == len(data_k):
            self.data['k'] = data_k['k']
        elif len(data_k) < len(self.data):
            raise IndexError('!!! len k < len data, make sure the right k file is used')
        elif len(self.data) < len(data_k):
            print('data len {dl} does not match the k len {kl}'.format(dl=len(self.data), kl=len(data_k)))
            warnings.warn('len k != len data; check if it is the right k and/or invalid data were discharged')
            abmn = ['a', 'b', 'm', 'n']
            abmnk = ['a', 'b', 'm', 'n', 'k']
            self.data = self.data.merge(data_k[abmnk], on=abmn, how='left', suffixes=('', '_'))
            self.data['k'] = self.data['k_']
            self.data.drop(columns='k_', inplace=True)

    def check_dist_amb(self):
        for (i, (ri, r)) in enumerate(self.data.iterrows()):
            elec_a = self.elec.loc[self.elec['num'] == r['a'], ['x', 'y', 'z']].to_numpy()
            elec_m = self.elec.loc[self.elec['num'] == r['m'], ['x', 'y', 'z']].to_numpy()
            elec_b = self.elec.loc[self.elec['num'] == r['b'], ['x', 'y', 'z']].to_numpy()
            dist_am = np.sum(np.abs(elec_m - elec_a))
            dist_mb = np.sum(np.abs(elec_b - elec_m))
            print(dist_am, dist_mb)
            if dist_am > dist_mb:
                self.data.loc[ri, 'dist_amb_valid'] = False

    def couple_rec(self, rec_couple=False, rec_unpaired=False,
                   dir_mark=1, rec_mark=2, unpaired_mark=0):
        if (rec_couple and rec_unpaired):
            direct = self.data.loc[self.data['rec_fnd'] == dir_mark]
            unpaired = self.data.loc[self.data['rec_fnd'] == unpaired_mark]
            self.data = pd.concat([direct, unpaired])
        elif (rec_couple and not rec_unpaired):
            direct = self.data.loc[self.data['rec_fnd'] == dir_mark]
            self.data = direct
        elif (not rec_couple and rec_unpaired):
            self.data = self.data

    def calc_2dgeometrical_factors(self):
        elec_dict = self.elec.set_index('num').to_dict()['x']
        data_coord = self.data[['a', 'b', 'm', 'n']]
        print('data coord num', data_coord)
        map_list = ['a', 'b', 'm', 'n']
        for column in map_list:
            data_coord[column] = data_coord[column].map(elec_dict)
        print('data coord coord', data_coord)
        data_coord = data_coord.to_numpy()
        print('data coord numpy', data_coord)
        k = 2 * 3.14 * (
                (1 / abs(data_coord[:, 0] - data_coord[:, 2]))
                - (1 / abs(data_coord[:, 0] - data_coord[:, 3]))
                - (1 / abs(data_coord[:, 1] - data_coord[:, 2]))
                + (1 / abs(data_coord[:, 1] - data_coord[:, 3]))
            ) ** -1
        self.data['k'] = k

    def to_bert(self, fname, w_rhoa, w_ip, w_err, data_cols, elec_cols, rounding=6):
        if w_rhoa:
            data_cols.extend(['rhoa', 'k'])
        if w_ip:
            data_cols.append('ip')
        if w_err:
            data_cols.append(w_err)
        with open(fname, 'w') as file_handle:
            file_handle.write(str(len(self.elec)) + '\n')
            file_handle.write('# ' + ' '.join(elec_cols) + '\n')
            self.elec[elec_cols].to_csv(file_handle, sep=' ', index=False, header=False, line_terminator='\n')
            # data_wrt = self.data[self.data.valid == 1][data_cols]
            data_wrt = self.data.loc[self.data['valid'] == 1, data_cols]
            file_handle.write(str(len(data_wrt)) + '\n')
            file_handle.write('# ' + ' '.join(data_cols) + '\n')
            data_wrt.to_csv(file_handle, sep=' ', index=False, header=False, float_format='%g', line_terminator='\n')

    def format_elec_coord(self, e, d=2):
        if d == 2:
            ex = self.elec.loc[self.elec['num'] == e, 'x'].to_numpy()[0]
            ez = self.elec.loc[self.elec['num'] == e, 'z'].to_numpy()[0]
            str_elec_coord = "{:10.2f}".format(ex) + " " + "{:10.2f}".format(ez)
        return(str_elec_coord)

    def to_ubc(self, fname, meas_col, err_col, w_err, rounding=6, sep=" "):
        data_ubc = self.data.copy()
        data_ubc = data_ubc.sort_values(by=['a', 'b'])
        inj_groups = data_ubc.groupby(['a', 'b'])
        with open(fname, 'w') as file_handle:
            for ab, g in inj_groups:
                file_handle.write(self.format_elec_coord(ab[0]))
                file_handle.write(sep)
                file_handle.write(self.format_elec_coord(ab[1]))
                file_handle.write(sep)
                file_handle.write("{:4.0f}".format(len(g)))
                file_handle.write("\n")
                for i, r in g.iterrows():
                    file_handle.write(self.format_elec_coord(r['m']))
                    file_handle.write(sep)
                    file_handle.write(self.format_elec_coord(r['n']))
                    file_handle.write(sep)
                    file_handle.write("{:8.8f}".format(r[meas_col]))
                    if w_err:
                        file_handle.write(sep)
                        file_handle.write("{:8.8f}".format(r[err_col]))
                    file_handle.write("\n")
                file_handle.write("\n")

    def plot_together(self, fname, plot_columns, valid_column='valid', dir_proc='.'):
        groupby_df = self.data.groupby(self.data[valid_column])
        try:
            group_valid = groupby_df.get_group(True)
        except KeyError:
            some_valid = False
        else:
            some_valid = True
        try:
            group_invalid = groupby_df.get_group(False)
        except KeyError:
            some_invalid = False
        else:
            some_invalid = True
        for col in plot_columns:
            fig, ax = plt.subplots()
            if some_valid:
                nmeas_valid = group_valid['meas'].to_numpy()
                ax.plot(nmeas_valid, group_valid[col].to_numpy(), 'o', color='b', markersize=4)
            if some_invalid:
                nmeas_invalid = group_invalid['meas'].to_numpy()
                ax.plot(nmeas_invalid, group_invalid[col].to_numpy(), 'o', color='r', markersize=4)
            scale, vstdmedian, vskewness = find_best_yscale(self.data[col])
            if scale == 'symlog':
                threshold = find_threshold_minnonzero(self.data[col])
                plt.minorticks_on()
                ax.set_yscale('symlog', linscale=0.2, linthresh=threshold, base=10)
                ax.yaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=threshold))
                ax.yaxis.set_minor_locator(MinorSymLogLocator(threshold))
            else:
                ax.set_yscale(scale)
                plt.minorticks_on()
            ax.grid(which='both', axis='both')
            plt.ylabel(col)
            plt.xlabel('measurement num')
            plt.tight_layout()
            os.makedirs(dir_proc, exist_ok=True)
            fig_fname = fname.replace('.Data', '_') + col + '.png'
            fig_dirfname = os.path.join(dir_proc, fig_fname)
            plt.savefig(fig_dirfname, dpi=100)
            plt.close()

    def plot(self, fname, plot_columns, valid_column='valid', dir_proc='.'):
        colors_validity = {1: 'b', 0: 'r'}
        labels_validity = {1: 'Valid', 0: 'Invalid'}
        groupby_df = self.data.groupby(self.data['valid'])
        for key in groupby_df.groups.keys():  # for group 1 (valid) and group 0 (invalid)
            meas = groupby_df.get_group(key)['meas'].to_numpy(dtype=int)
            for c in plot_columns:
                fig_fname = fname.replace('.Data', '_') + labels_validity[key] + '_' + c + '.png'
                fig_dfname = os.path.join(dir_proc, fig_fname)
                y = groupby_df.get_group(key)[c].to_numpy()
                _ = plt.figure(figsize=(10, 10))
                plt.plot(meas, y, 'o', color=colors_validity[key], markersize=4)
                plt.ylabel(c)
                plt.yscale('log')
                plt.xlabel('measurement num')
                plt.tight_layout()
                plt.savefig(fig_dfname)
                plt.close()

    def report(self, cols=['valid']):
        for c in cols:
            print('-----\n', self.data[c].value_counts())


def naive_pairs(l):
    for c in l:
        for r in l:
            if c == r:
                pass
            else:
                yield((c, r))


def crossvalidity(df, summary_col_header='valid'):
    """
    1 is valid and 0 is invalid, the table will report the number of rejections
    """
    if summary_col_header not in df.columns.tolist():
        df[summary_col_header] = df.all(axis=1)

    cols = df.columns.tolist()

    dfc = pd.DataFrame(index=cols, columns=cols)

    # off-diagonal: number of data points that both filters reject
    for (p0, p1) in naive_pairs(cols):
        common_rejections = np.sum((~df[[p0, p1]]).all(axis=1))
        dfc.loc[p0, p1] = common_rejections
        dfc.loc[p1, p0] = common_rejections

    # diagonal: number of data points that only that filter reject
    # remove the summary column, it would always match the other columns
    # ns: no summary
    cols.remove(summary_col_header)
    dfns = ~df[cols]
    sum_reject = dfns.sum(axis=1)
    dfns_unique = dfns[sum_reject == 1]
    colsns_unique_reject = dfns_unique.sum(axis=0)
    for c in cols:
        dfc.loc[c, c] = colsns_unique_reject[c]
    dfc.loc[summary_col_header, summary_col_header] = np.sum(~df[summary_col_header])
    return(dfc)


def __process__(f, args):
    """ process ERT file """
    if args.fType == 'labrecque':
        elec, data = read_labrecque(f)
    # pass to ERTdataset class
    ds = ERTdataset(data=data, elec=elec)
    # adjust
    ds.data[['a', 'b', 'm', 'n']] += args.s_abmn
    ds.data['meas'] += args.s_meas
    ds.elec['num'] += args.s_elec
    if args.f_elec is not None:
        ds.elec = pd.read_csv(args.f_elec, sep=r'[\s,]{1,20}', engine='python')
    # filters
    ds.process_rec(x='r', x_avg='rec_avg', x_err='rec_err')
    ds.data['rec_valid'] = ds.data['rec_err'] < args.rec
    ds.data['ctc_valid'] = ds.data['ctc'] < args.ctc
    ds.data['stk_valid'] = ds.data['stk'] < args.stk
    if args.badelec is not None:
        print('removing the data associated with these electrodes: ', args.badelec)
        ds.data['elec_valid'] = ~np.isin(ds.data[['a', 'b', 'm', 'n']], args.badelec).any(axis=1)
    ds.data['v_valid'] = ds.data['v'].abs() > args.v
    if args.dist_amb:
        ds.check_dist_amb()
    else:
        ds.data['dist_amb_valid'] = True
    if all(ds.data['ip'].notnull()):
        ds.process_rec(x='ip', x_avg='rec_ip_avg', x_err='rec_ip_err')
    if args.k_file:
        elec_kfile, data_kfile = read_bert(k_file=args.k_file)
        if not any(data_kfile['k']):
            print('!!! calculating k from r, assuming rho == 1')
            data_kfile['k'] = 1 / data_kfile['r']
        ds.get_k(data_kfile)
        ds.data['k_valid'] = ds.data['k'].abs() < args.k
        ds.data['rhoa'] = ds.data['r'] * ds.data['k']
        ds.data['rhoa_valid'] = ds.data['rhoa'].between(args.rhoa[0], args.rhoa[1])
    # combine filters
    filters = ['rec_valid', 'k_valid', 'rhoa_valid', 'v_valid', 'ctc_valid', 'stk_valid', 'elec_valid']
    ds.data['valid'] = ds.data[filters].all(axis='columns')
    # output csv
    ds.default_types()
    fcsv = output_file(f, new_ext='.csv', directory=args.dir_proc)
    ds.data.to_csv(fcsv, float_format='%#8g', index=False)
    ds.couple_rec(rec_couple=args.rec_couple, rec_unpaired=args.rec_unpaired)
    # output dat
    data_cols = ['a', 'b', 'm', 'n', 'r']
    elec_cols = ['x', 'y', 'z']
    if args.odf == 'bert':
        fdat = output_file(f, new_ext='.dat', directory='.')
        ds.to_bert(fdat, args.w_rhoa, args.w_ip, args.w_err, data_cols, elec_cols)
    elif args.odf == 'ubc':
        fdat = output_file(f, new_ext='.obs', directory='.')
        ds.to_ubc(fdat, meas_col='r', err_col='rec_err', w_err=True)
    else:
        raise ValueError('invalid output data format')
    print('output data file: ', fdat)
    # report
    report_columns = ['rec_valid', 'k_valid', 'rhoa_valid', 'v_valid', 'ctc_valid', 'stk_valid', 'elec_valid', 'valid']
    ds.report(cols=report_columns)
    df_crossvalidity = crossvalidity(ds.data[report_columns], summary_col_header='valid')
    print('\ncross-rejection table\n', df_crossvalidity)

    # plot
    plot_columns = ['ctc', 'stk', 'v', 'rec_err', 'k', 'rhoa']
    ds.plot_together(f, plot_columns=plot_columns, valid_column='valid', dir_proc=args.dir_proc)
    # ds.plot(f, plot_columns=plot_columns, valid_column='valid', dir_proc=args.dir_proc)
    # datetime
    fdatetime = ds.data.loc[0, 'datetime']
    return(fcsv, fdat, fdatetime)


def process(**kargs):
    """
    get and merge args from CMDLINE and KARGS, then process files
    """
    cmd_args = get_cmd()
    args = update_args(cmd_args, dict_args=kargs)
    args = check_args(args)
    for f in args.fName:
        print('\n', '-' * 80, '\n', f)
        print(f)
        fcsv, fdat, datetime = __process__(f, args)
        yield(fcsv, fdat, datetime)


if __name__ == '__main__':
    print('CLI-processing')
    fyield = process()
    for (fcsv, finv, datetime) in fyield:
        print('processed data saved to ', finv)
