"""
region analysis of vtk files
"""

import os
import argparse
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import seaborn
import datetime


def get_cmd():
    """ get command line arguments for data processing """
    parse = argparse.ArgumentParser()
    main = parse.add_argument_group('main')
    ertds = parse.add_argument_group('ertds')
    reg = parse.add_argument_group('reg')
    vtk = parse.add_argument_group('vtk')
    output = parse.add_argument_group('output')
    # MAIN
    main.add_argument('-group_analysis', action='store_true', help='perform group analysis, based on regs group column')
    main.add_argument('-change_dir', type=str, help='change directory of vtk files from the table one')
    # ERTDS FILE
    ertds.add_argument('-csv_datasets', type=str, help='ertds csv file with information on the ERT datasets')
    ertds.add_argument('-datetime_col', type=str, help='header of datetime column in the csv_datasets')
    ertds.add_argument('-vtk_col', type=str, help='header of the column with the vtk files in csv_datasets')
    # VTK FILES
    vtk.add_argument('-rho', type=str, default='res', help='resistivity name in vtk files')
    # REGION FILE
    reg.add_argument('-csv_reg', type=str, help='csv file with regions-volumes names and coords')
    reg.add_argument('-name_col', type=str, help='header of datetime column in the csv_datasets', default='name')
    reg.add_argument('-group_col', type=str, help='header of datetime column in the csv_datasets', default='group')
    reg.add_argument(
        '-coords',
        type=str,
        default=['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'],
        help='coord headers',
        nargs='+',
        )
    reg.add_argument(
        '-stats',
        type=str,
        default=['avg', 'std', 'min', 'max', 'num'],
        help='statistics headers',
        nargs='+',
        )
    # OUTPUT
    output.add_argument('-add_MultiIndex', type=str, default=None, help='add str as highest level column MultiIndex')
    output.add_argument('-dir_analysis', type=str, help='output directory', default='analysis')
    output.add_argument('-file_analysis', type=str, help='output file', default='ert_analysis.csv')
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


def combine_ert_regions(
        ds, regs,
        vtk_col='fvtk', datetime_col='datetime',
        reg_name_col='name', reg_group_col='group',
        coords=['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'],
        stat_meas=['avg', 'std', 'min', 'max', 'num']
        ):
    """
    MultiIndex-Column Table
    Region names: name of the ERT regions to analyse
    Statistical measurements for each region: avg, std, min, and max
    Arguments:
    * ds: dataframe with information on the ERT datasets
    * regs: dataframe with information on the regions to analyze
    * vtk_col: header of the ds column with the name of the ERT vtk files
    * datetime_col: header of the ds column with the datetime of each dataset
    * reg_name_col: header of the regs column with the names of each region
    * reg_group_col: header of the regs column with the group of each region
    * coords: headers of the regs columns with the coordinates of each region
    * stat_meas: statistical measurements to include
    Return:
    * dataframe: each ERT-datetime is a row (datetime index)
    """
    # columns
    # region MultiIndex
    region_names = regs[reg_name_col]
    region_cols = [reg_group_col] + coords + stat_meas  # lists of columns regarding the regions
    MI_regions = pd.MultiIndex.from_product([region_names, region_cols])
    # ERT dataset info
    MI_ert = pd.MultiIndex.from_product([['ERT_dataset'], [vtk_col, datetime_col]])
    # combine MIs
    MIcols = MI_ert.append(MI_regions)
    # init df
    df = pd.DataFrame(data=None, index=ds.index, columns=MIcols)
    # update df
    # from ERT ds
    df[MI_ert] = ds[[vtk_col, datetime_col]]
    # from regs region information
    regs = regs.set_index('name')
    regs = regs.stack()
    for index in regs.index:
        df.loc[:, index] = regs.loc[index]
    return(df)


def add_stat_measurements(df, vtk_col, reg_coords, rho, change_dir):
    regions = [col for col in df.columns.unique(level=0) if
               all(coord in df[col].columns.get_level_values(0).tolist() for coord in reg_coords)]
    for i, row in df.iterrows():  # each row is a file, one xyzr
        fname = row[('ERT_dataset', vtk_col)]
        if change_dir is not None:
            basename = os.path.basename(fname)
            fname = os.path.join(change_dir, basename)
        xyzr = get_vtk_xyzr(fname, rho)
        for reg in regions:
            row_reg = row[reg]
            rho_avg, rho_std, rho_min, rho_max, rho_num = get_region_stats(xyzr, row_reg, reg_coords)
            df.loc[i, (reg, 'avg')] = rho_avg
            df.loc[i, (reg, 'std')] = rho_std
            df.loc[i, (reg, 'num')] = rho_num
            df.loc[i, (reg, 'min')] = rho_min
            df.loc[i, (reg, 'max')] = rho_max
    return(df)


def group_analysis(regs, df, args):
    """ combines area of one group together """
    regs_names = list(regs[args.name_col])
    regs_groups_all = list(regs[args.group_col])
    regs_groups_unique = list(regs[args.group_col].unique())
    print(args.stats)
    print(regs_groups_unique)
    columns = pd.MultiIndex.from_product([regs_groups_unique, args.stats])
    df_groups = pd.DataFrame(index=df.index, columns=columns)
    for g in regs_groups_unique:
        names_in_group = [ni for ni, gi in zip(regs_names, regs_groups_all) if gi == g]
        print(names_in_group)
        df_in_group = df.loc[:, (names_in_group, args.stats)]
        df_in_group = df_in_group.apply(pd.to_numeric)
        print(df_in_group.info())
        df_mean = df_in_group.mean(level=1, axis='columns')
        for c in args.stats:
            df_groups.loc[:, (g, c)] = df_mean.loc[:, c]
    return(df_groups)


def get_vtk_xyzr(fName, rho):
    """ read vtk and extract value and center of each cell in it """
    mesh = pv.read(fName)
    rho = mesh[rho]
    centers = mesh.cell_centers().points
    xyzr = np.column_stack((centers, rho))
    xyzr = pd.DataFrame(xyzr, columns=['x', 'y', 'z', 'r'])
    return(xyzr)


def get_csv_xzrs(fName, rho):
    """ read csv file with x, z, resistivity """
    xzr = pd.read_csv(fName, header=None, index_col=[0, 1, 2], names=['x', 'z', 'r'])
    return(xzr)


def get_region_stats(xyzr, row_reg, reg_coords):
    """ xyzr contains the data, table contains the regions """
    region_cells = xyzr[(xyzr['x'].between(row_reg[reg_coords[0]], row_reg[reg_coords[1]])) &
                        (xyzr['y'].between(row_reg[reg_coords[2]], row_reg[reg_coords[3]]))]
    rho_avg = region_cells['r'].mean()
    rho_std = region_cells['r'].std()
    rho_min = region_cells['r'].min()
    rho_max = region_cells['r'].max()
    rho_num = len(region_cells)
    return(rho_avg, rho_std, rho_min, rho_max, rho_num)


def plot_seaborn(df, x, y, hue, output_name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetimems'] = pd.to_datetime(df['datetime'], unit='ms')
    df_space = df.loc[df['reg_group'] == 'space']
    df_space_avg = df_space.groupby('datetimems').mean()
    df_crop = df.loc[df['reg_group'] == 'crop']
    df_crop_avg = df_crop.groupby('datetimems').mean()
    seaborn.scatterplot(
        data=df_crop, x=x, y=y, ax=ax, s=90, hue='reg_name', palette='Greens',
        alpha=1, legend='full',  edgecolor='k',
        )
    seaborn.scatterplot(
        data=df_space, x=x, y=y, ax=ax, s=90, hue='reg_name', palette='Reds',
        alpha=1, legend='full', edgecolor='k',
        )
    seaborn.lineplot(x=df_space_avg.index, y=df_space_avg['ravg'], ax=ax, color='darkred', linewidth=5.5)
    seaborn.lineplot(x=df_crop_avg.index, y=df_crop_avg['ravg'], ax=ax, color='darkgreen', linewidth=5.5)
    locator = matplotlib.dates.AutoDateLocator(minticks=5, maxticks=20)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    ax.set_xlim([datetime.date(2019, 11, 15), datetime.date(2020, 5, 1)])
    ax.set_xlabel('datetime')
    ax.set_ylabel('rho [ohm m]')
    ax.legend(loc='center left')
    plt.tight_layout()
    plt.savefig(output_name, dpi=600)
    plt.show()


def plot_stats(df, output_name, ylabel='rho'):
    print('\ndf for plotting:\n', df)
    # ndata = df.shape[1]
    colors = plt.cm.get_cmap('Set1', None).colors
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    for column, color in zip(df.columns, colors):
        plt.plot(df.index, df[column], '-o', color=color, label=column, linewidth=6, markersize=8)
    locator = matplotlib.dates.AutoDateLocator(minticks=5, maxticks=20)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    # ax.set_xlim([datetime.date(2019, 11, 15), datetime.date(2020, 7, 1)])
    ax.set_xlabel('datetime')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.grid()
    plt.savefig(output_name, dpi=600)
    plt.show()


def _analysis_(args):
    # csv with ert data sets
    ds = pd.read_csv(args.csv_datasets)
    ds[args.datetime_col] = pd.to_datetime(ds[args.datetime_col])
    ds = ds.set_index(args.datetime_col, drop=False)
    ds = ds.dropna(axis=0, how='any')
    print('\ndf with ert datasets:\n', ds)

    # csv with regions
    freg = os.path.join(args.dir_analysis, args.csv_reg)
    regs = pd.read_csv(freg, index_col=False)  # df with regions (contains coordinates of the regions)
    print('\ndf with regions:\n', regs)

    # combine ert datasets and regions
    df = combine_ert_regions(ds, regs, vtk_col=args.vtk_col, datetime_col=args.datetime_col, stat_meas=args.stats)
    df = add_stat_measurements(df, args.vtk_col, args.coords, args.rho, args.change_dir)
    print('\ndf combined with statistical analysis of the regions:\n', df)

    # optional, analyze the area in groups
    if args.group_analysis:
        df_group = group_analysis(regs, df, args)
        df_group_stat = df_group.loc[:, (slice(None), 'avg')]
        png_out = os.path.join(args.dir_analysis, 'group_analysis.pdf')
        plot_stats(df_group_stat, png_out)

    df_stat = df.loc[:, (slice(None), 'avg')]
    png_out = os.path.join(args.dir_analysis, 'analysis_plot.pdf')
    plot_stats(df_stat, png_out)
    csv_out = os.path.join(args.dir_analysis, args.file_analysis)
    df = df.infer_objects()
    df = df.round(3)
    df.to_csv(csv_out)


def analysis(**kargs):
    cmd_args = get_cmd()
    args = update_args(cmd_args, dict_args=kargs)
    pv.set_plot_theme("document")
    _analysis_(args)


if __name__ == '__main__':
    analysis()
