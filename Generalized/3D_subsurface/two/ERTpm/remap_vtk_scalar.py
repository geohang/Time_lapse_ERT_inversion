import argparse
import pyvista as pv
import numpy as np
# from IPython import embed
import os


def get_cmdline():
    """
    get CLI arguments:
    * a common parser for general arguments, such as names of vtk file and scalar data therein.
    * subparsers for choosing the model and its parameters
    """

    common_parser = argparse.ArgumentParser()
    common_parser.add_argument('-fnames', type=str, help='file names (vtk)', nargs='+')
    common_parser.add_argument('-sn', type=str, help='scalar name to ', default='res')
    common_parser.add_argument('-out_dir', type=str, help='output directory', default='remapped')

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='remap model',
        dest='model',
        )

    nick_model = subparsers.add_parser(
        'archie',
        add_help=False,
        description='Archie model: rho = rho_sat * eff_sat ** (- n)',
        parents=[common_parser],
        )
    nick_model.add_argument('-rho_sat', type=float, help='rho at saturation', required=True)
    nick_model.add_argument('-n', type=float, help='exponent', required=True)

    cmdline = parser.parse_args()
    return(cmdline)


def update_args(args, new_args):
    """ update args with new args from dict """
    for key, val in new_args.items():
        setattr(args, key, val)
    return(args)


def check_args(args):
    """ check consistency of args """
    if isinstance(args.fnames, str):
        args.fnames = [args.fnames]
    return(args)


def archie_model(rho, rho_sat, n):
    eff_sat = (rho_sat / rho) ** (1 / n)
    return(eff_sat)


def prepare_cov(mesh, cov, min_cov=0.05):
    """
    in bert the coverage is the log of the absolute values of the jacobian matrix column
    here we limit the values between a minimum and 1,
    this ensures a nice transition and shape, and compatibility with the plotting opacity argument
    """
    cov_np = mesh[cov]
    cov_np[cov_np < min_cov] = 0
    cov_np[cov_np > 1] = 1
    mesh[cov] = cov_np
    return(mesh)


def plot_remapped(vtk, field, cov, cmin, cmax, clipping='bounds', fpng='remapped.png'):
    pv.set_plot_theme("document")
    min_cov = 0.05
    bounds = [-1, 32.5, -1.9, 0.1, 0, 0]
    vtk = prepare_cov(vtk, 'cov', min_cov=min_cov)
    sba = dict(width=0.5, position_x=0.25, height=0.1, position_y=0.025)
    plotter = pv.Plotter(window_size=(1300, 600))
    # select region
    if clipping == 'cov':
        vtk = vtk.threshold(min_cov, cov, invert=False)
    elif clipping == 'bounds':
        vtk = vtk.clip_box(bounds, invert=False)
    # camera
    length = vtk.GetLength()
    bounds = vtk.GetBounds()
    print(bounds)
    xc = (bounds[1] + bounds[0]) / 2
    yc = (bounds[3] + bounds[2]) * 2.5 / 4
    cam = [[xc, yc, length], [xc, yc, 0], [0, 1, 0]]
    plotter.camera_position = cam
    # vtk and data
    _ = plotter.add_mesh(
        vtk,
        scalars=field,
        opacity=cov,
        show_edges=False,
        edge_color='k',
        show_scalar_bar=True,
        scalar_bar_args=sba,
        stitle='volumetric water content\n',
        cmap='RdBu',
        clim=(cmin, cmax),
        )
    # electrodes
    electrodes = np.zeros((64, 3))
    electrodes[:, 0] = np.arange(0, 32, 0.5)
    electrodes = pv.PolyData(electrodes)
    electrodes["elec_num"] = ["{}".format(i+1) for i in range(electrodes.n_points)]
    plotter.add_point_labels(electrodes, "elec_num", font_size=12, point_size=5, shape=None, point_color='k')
    # names
    cultivars_xyz = np.zeros((10, 3))
    cultivars_xyz[:, 0] = np.linspace(2.5, 28.6, 10, endpoint=True)
    cultivars_xyz[:, 0] -= 1
    cultivars_xyz[:, 1] = 0.5
    cultivars_names = [
        'Burchett', 'Chisholm', 'Warrior', 'Sturdy', 'Cossack',
        'Warrior', 'Burchett', 'Sturdy', 'Chisholm', 'Cossack'
        ]
    cultivars = pv.PolyData(cultivars_xyz)
    cultivars['names'] = cultivars_names
    plotter.add_point_labels(cultivars, "names", font_size=14, shape_color=[1, 1, 1])
    # bounds
    _ = plotter.show_bounds(
        mesh=vtk,
        grid=None,
        location='outer',
        ticks='outside',
        font_size=18,
        font_family='times',
        use_2d=True,
        padding=0,
        xlabel='m',
        ylabel='m',
        corner_factor=0,
        )
    plotter.show(screenshot=fpng, interactive=False, auto_close=True)


def __remap__(fn, args):
    vtk = pv.read(fn)
    field = vtk[args.sn]
    if args.model == 'archie':
        field_remapped = archie_model(field, args.rho_sat, args.n)
        print('remapped field\n', field_remapped)
        vtk['wcnt'] = field_remapped
        basename_vtk = os.path.basename(fn)
        basename_png = basename_vtk.replace('.vtk', '.png')
        fout = os.path.join(args.out_dir, basename_png)
        plot_remapped(vtk, field='wcnt', cov='cov', cmin=0.2, cmax=0.6, clipping='bounds', fpng=fout)
        fout = os.path.join(args.out_dir, basename_vtk)
        vtk.save(fout)


def remap(**kargs):
    """
    Takes and combines arguments from CLI and kargs,
    so that it can be called from python or CLI.
    Exposes the IO.
    """
    cmdline_args = get_cmdline()
    args = update_args(args=cmdline_args, new_args=kargs)
    args = check_args(args)

    fnames = args.fnames
    print('arguments are:\n', args)
    for fn in fnames:
        print('remapping ', fn)
        __remap__(fn, args)


if __name__ == '__main__':
    remap()
