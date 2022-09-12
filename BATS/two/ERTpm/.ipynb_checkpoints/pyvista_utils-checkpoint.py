import pyvista as pv
from IPython import embed


def prepare_cov(mesh, cov, min_cov=0.05):
    """ in bert the coverage is the log of the absolute values of the jacobian matrix column
    here we limit the values between a minimum and 1,
    this ensures a nice transition and shape, and compatibility with the plotting opacity argument """
    cov_np = mesh[cov]
    cov_np[cov_np < min_cov] = 0
    cov_np[cov_np > 1] = 1
    mesh[cov] = cov_np
    return(mesh)

def multiblock_gif(mb, s):
    # mb = mb.clip_box([0, 31.5, -2, 0.1, 0, 0], invert=False)
    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=(1300, 600))
    plotter.open_gif('mbgif.gif')
    plotter.open_movie('mbgif.mp4', framerate=1)
    # camera
    xc = (mb.bounds[1] + mb.bounds[0]) / 2
    yc = (mb.bounds[3] + mb.bounds[2]) * 2.5 / 4
    cam = [[xc, yc, mb.length], [xc, yc, 0], [0, 1, 0]]
    plotter.camera_position = cam
    # color bar args
    sba = dict(width=0.5, position_x=0.25, height=0.1, position_y=0.025)
    # data
    plotter.show(auto_close=False)
    for vtk in mb:
        vtk = prepare_cov(vtk, 'cov')
        plotter.add_mesh(
            vtk,
            scalars=s,
            opacity='cov',
            show_edges=False,
            edge_color='darkgrey',
            show_scalar_bar=True,
            scalar_bar_args=sba,
            stitle='volumetric water content',
            cmap='viridis',
            clim=(0.2, 0.6),
            )

        plotter.show_bounds(
            mesh=vtk,
            grid='k',
            location='outer',
            ticks=True,
            font_size=14,
            font_family='times',
            use_2d=True,
            padding=0,
            xlabel='m',
            ylabel='m',
            )
        plotter.write_frame()
        plotter.clear()

    plotter.close()
