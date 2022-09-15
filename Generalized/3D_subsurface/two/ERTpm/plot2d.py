import argparse
import numpy as np
import pyvista as pv
from IPython import embed


def get_cmd():
    """ get command line arguments for data processing """
    parse = argparse.ArgumentParser()
    main = parse.add_argument_group('main')
    option = parse.add_argument_group('option')
    output = parse.add_argument_group('output')
    # MAIN
    main.add_argument('-fName', type=str, help='data file to process', nargs='+')
    main.add_argument('-bounds', type=float, help='(xmin, xmax, ymin, ymax, zmin, zmax)', nargs='+')
    main.add_argument('-dName', type=str, help='data file for difference')
    main.add_argument('-rho', type=str, default='res', help='vtk resistivity name')
    main.add_argument('-cov', type=str, default='cov', help='coverage vtk name')
    # OPTIONS
    option.add_argument('-dHow', type=str, help='how to perform difference', default='pct')
    option.add_argument('-Cmin', type=float, help='min colorbar', default=1)
    option.add_argument('-Cmax', type=float, help='max colorbar', default=100)
    option.add_argument('-Cgrid', type=str, help='grid color', default=None)
    option.add_argument('-Cedge', type=str, help='edge color', default='darkgrey')
    option.add_argument('-Cmap', type=str, help='colormap name', default='turbo')
    option.add_argument('-clipping', help='clipping method', default='cov', choices=['cov', 'bounds', None])
    option.add_argument('-ticks', type=str, help='figure ticks', default='outside')
    option.add_argument('-notes', help='additional notes')
    option.add_argument('-Fsize', type=float, help='float size', default=18)
    option.add_argument('-min_cov', type=float, help='minimum coverage', default=0.05)

    # OUTPUT
    output.add_argument('-dir_vista', type=str, help='output directory', default='vista')
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
    return(args)


def prepare_rho(mesh, rho, dName, dHow):
    if dName is not None:
        mesh_diff = pv.read(dName)
        if dHow == 'val':
            mesh[rho] = mesh[rho] - mesh_diff[rho]
        if dHow == 'pct':
            mesh[rho] = mesh[rho] / mesh_diff[rho]
    return(mesh)


def prepare_cov(mesh, cov, min_cov=0.05):
    """ in bert the coverage is the log of the absolute values of the jacobian matrix column
    here we limit the values between a minimum and 1,
    this ensures a nice transition and shape, and compatibility with the plotting opacity argument """
    cov_np = mesh[cov]
    cov_np[cov_np < min_cov] = 0
    cov_np[cov_np > 1] = 1
    mesh[cov] = cov_np
    return(mesh)


def _plot_(f, args):
    mesh = pv.read(f)
    mesh = prepare_rho(mesh=mesh, rho=args.rho, dName=args.dName, dHow=args.dHow)
    mesh = prepare_cov(mesh=mesh, cov=args.cov, min_cov=0.05)
    sba = dict(title='resistivity ohm m\n', width=0.5, position_x=0.25, height=0.1, position_y=0.025)
    plotter = pv.Plotter(window_size=(1300, 600))
    # select region
    if args.clipping == 'cov':
        mesh = mesh.threshold(args.min_cov, args.cov, invert=False)
    elif args.clipping == 'bounds':
        # cell_centers = mesh.cell_centers().points
        # inside = np.where(cell_centers[:, 2] > -4)
        # mesh = mesh.extract_cells(inside)
        mesh = mesh.clip_box((-1, 32.5, -6, 1, 0, 0), invert=False)
    # camera
    length = mesh.GetLength()
    bounds = mesh.GetBounds()
    xc = (bounds[1] + bounds[0]) / 2
    yc = (bounds[3] + bounds[2]) * 2.5 / 4
    cam = [[xc, yc, length], [xc, yc, 0], [0, 1, 0]]
    plotter.camera_position = cam
    # mesh and data
    _ = plotter.add_mesh(
        mesh, scalars=args.rho, opacity=args.cov,
        show_edges=False, edge_color='k',
        show_scalar_bar=True, scalar_bar_args=sba,
        cmap=args.Cmap, clim=(args.Cmin, args.Cmax)
    )
    # electrodes
    electrodes = np.zeros((64, 3))
    electrodes[:, 0] = np.arange(0, 32, 0.5)
    electrodes = pv.PolyData(electrodes)
    electrodes_labels = ["{}".format(i + 1) for i in range(electrodes.n_points)]
    electrodes["elec_num"] = electrodes_labels
    plotter.add_point_labels(electrodes, "elec_num", font_size=14, point_size=5, shape=None, point_color='k')
    # add optional notes if present
    if args.notes is not None:
        for k, v in args.notes.items():
            coordinates = pv.PolyData(np.array(v))
            coordinates['name'] = [k]
            plotter.add_point_labels(
                coordinates, 'name', font_size=14, point_size=5, shape=None, point_color='k'
            )
    # bounds
    _ = plotter.show_bounds(mesh=mesh, grid=args.Cgrid,
                            location='outer', ticks=args.ticks, font_size=args.Fsize,
                            font_family='times', use_2d=True, padding=0,
                            xlabel='m', ylabel='m')
    fpng = f.replace('.vtk', '.png')
    plotter.show(screenshot=fpng)
    return(fpng)


def plot2d(**kargs):
    cmd_args = get_cmd()
    args = update_args(cmd_args, dict_args=kargs)
    args = check_args(args)
    pv.set_plot_theme("document")
    for f in args.fName:
        print('\n', '-' * 80, '\n', f)
        print(f)
        fpng = _plot_(f, args)
        yield(fpng)


if __name__ == '__main__':
    plot2d()
