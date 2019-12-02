import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cortex


def add_labels(ax, model, fontsize=12, color="k"):
    surfs = [cortex.polyutils.Surface(*d)
             for d in cortex.db.get_surf(model.subject.name, "flat")]

    for surf, hemi in zip(surfs, ["lh", "rh"]):
        mins = np.min(surf.pts, axis=0)
        maxs = np.max(surf.pts, axis=0)
        dx, dy, dz = maxs - mins

        label_pos = {}
        for label in model.labels_surf:
            if label.hemi == hemi:
                label_xmin, label_ymin = np.min(surf.pts[label.vertices], axis=0)[:2]
                label_xmax, label_ymax = np.max(surf.pts[label.vertices], axis=0)[:2]
                label_dx = label_xmax - label_xmin
                label_dy = label_ymax - label_ymin

                center_of_masses = []
                for ind, (delta, label_delta) in enumerate(zip([dx, dy], [label_dx, label_dy])):
                    coords = surf.pts[label.vertices][:, ind]
                    coord_mean = np.mean(coords)
                    if label_delta > 0.5 * delta:
                        threshold = (mins[ind] + maxs[ind]) / 2.0
                        if coord_mean > threshold:
                            coord_mean = np.mean(coords[coords > threshold])
                        else:
                            coord_mean = np.mean(coords[coords < threshold])
                    center_of_masses.append(coord_mean)

                label_pos[label.name] = center_of_masses

        for s in label_pos:
            if hemi == "lh":
                x = label_pos[s][0] + mins[0]
            else:
                x = label_pos[s][0] - mins[0]

            ax.text(x, label_pos[s][1], s[:-3], horizontalalignment='center',
                    verticalalignment='center', fontsize=fontsize, color=color)


def draw_roi_borders(vertex_map, model):
    surfs = [cortex.polyutils.Surface(*d)
             for d in cortex.db.get_surf(model.subject.name, "flat")]

    hemi_num_verts = np.array([surfs[0].pts.shape[0], surfs[1].pts.shape[0]])
    mask = np.zeros((hemi_num_verts.sum(),), dtype=bool)
    for num_verts, surf, hemi in zip(hemi_num_verts, surfs, ["lh", "rh"]):
        for label in model.labels_surf:
            if label.hemi == hemi:

                # Get the border of the region
                m = np.zeros((num_verts,), dtype=bool)
                m[label.vertices] = 1
                subsurf = surf.create_subsurface(m)
                m = subsurf.lift_subsurface_data(subsurf.boundary_vertices)
                if hemi == 'rh':
                    mask[hemi_num_verts[0]:] += m
                else:
                    mask[:hemi_num_verts[0]] += m

    vertex_map.data[mask] = np.nan


def _plot_mpl_stc(stc, subject=None, surface='inflated', hemi='lh',
                  colormap='auto', time_label='auto', smoothing_steps=10,
                  subjects_dir=None, views='lat', clim='auto', figure=None,
                  initial_time=None, time_unit='s', background='black',
                  spacing='oct6', time_viewer=False, colorbar=True,
                  transparent=True):
    """Plot source estimate using mpl."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.widgets import Slider
    import nibabel as nib
    from scipy import stats
    from ..morph import _get_subject_sphere_tris
    if hemi not in ['lh', 'rh']:
        raise ValueError("hemi must be 'lh' or 'rh' when using matplotlib. "
                         "Got %s." % hemi)
    lh_kwargs = {'lat': {'elev': 0, 'azim': 180},
                 'med': {'elev': 0, 'azim': 0},
                 'ros': {'elev': 0, 'azim': 90},
                 'cau': {'elev': 0, 'azim': -90},
                 'dor': {'elev': 90, 'azim': -90},
                 'ven': {'elev': -90, 'azim': -90},
                 'fro': {'elev': 0, 'azim': 106.739},
                 'par': {'elev': 30, 'azim': -120}}
    rh_kwargs = {'lat': {'elev': 0, 'azim': 0},
                 'med': {'elev': 0, 'azim': 180},
                 'ros': {'elev': 0, 'azim': 90},
                 'cau': {'elev': 0, 'azim': -90},
                 'dor': {'elev': 90, 'azim': -90},
                 'ven': {'elev': -90, 'azim': -90},
                 'fro': {'elev': 16.739, 'azim': 60},
                 'par': {'elev': 30, 'azim': -60}}
    kwargs = dict(lh=lh_kwargs, rh=rh_kwargs)
    _check_option('views', views, sorted(lh_kwargs.keys()))
    colormap, scale_pts, _, _, _ = _limits_to_control_points(
        clim, stc.data, colormap, transparent, linearize=True)
    del transparent

    time_label, times = _handle_time(time_label, time_unit, stc.times)
    fig = plt.figure(figsize=(6, 6)) if figure is None else figure
    ax = Axes3D(fig)
    hemi_idx = 0 if hemi == 'lh' else 1
    surf = op.join(subjects_dir, subject, 'surf', '%s.%s' % (hemi, surface))
    if spacing == 'all':
        coords, faces = nib.freesurfer.read_geometry(surf)
        inuse = slice(None)
    else:
        stype, sval, ico_surf, src_type_str = _check_spacing(spacing)
        surf = _create_surf_spacing(surf, hemi, subject, stype, ico_surf,
                                    subjects_dir)
        inuse = surf['vertno']
        faces = surf['use_tris']
        coords = surf['rr'][inuse]
        shape = faces.shape
        faces = stats.rankdata(faces, 'dense').reshape(shape) - 1
        faces = np.round(faces).astype(int)  # should really be int-like anyway
    del surf
    vertices = stc.vertices[hemi_idx]
    n_verts = len(vertices)
    tris = _get_subject_sphere_tris(subject, subjects_dir)[hemi_idx]
    e = mesh_edges(tris)
    e.data[e.data == 2] = 1
    n_vertices = e.shape[0]
    maps = sparse.identity(n_vertices).tocsr()
    e = e + sparse.eye(n_vertices, n_vertices)
    cmap = cm.get_cmap(colormap)
    greymap = cm.get_cmap('Greys')

    curv = nib.freesurfer.read_morph_data(
        op.join(subjects_dir, subject, 'surf', '%s.curv' % hemi))[inuse]
    curv = np.clip(np.array(curv > 0, np.int), 0.33, 0.66)
    params = dict(ax=ax, stc=stc, coords=coords, faces=faces,
                  hemi_idx=hemi_idx, vertices=vertices, e=e,
                  smoothing_steps=smoothing_steps, n_verts=n_verts,
                  inuse=inuse, maps=maps, cmap=cmap, curv=curv,
                  scale_pts=scale_pts, greymap=greymap, time_label=time_label,
                  time_unit=time_unit)
    _smooth_plot(initial_time, params)

    ax.view_init(**kwargs[hemi][views])

    try:
        ax.set_facecolor(background)
    except AttributeError:
        ax.set_axis_bgcolor(background)

    fig.subplots_adjust(left=0., bottom=0., right=1., top=1.)

    # add colorbar
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(scale_pts[0], scale_pts[2]))
    cax = inset_axes(ax, width="80%", height="5%", loc=8, borderpad=3.)
    plt.setp(plt.getp(cax, 'xticklabels'), color='w')
    sm.set_array(np.linspace(scale_pts[0], scale_pts[2], 256))
    if colorbar:
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb_yticks = plt.getp(cax, 'yticklabels')
        plt.setp(cb_yticks, color='w')
        cax.tick_params(labelsize=16)
        cb.patch.set_facecolor('0.5')
        cax.set(xlim=(scale_pts[0], scale_pts[2]))
    plt.show()
    return fig









def plot_vertex_map(vertex_map, ax, model, fontsize=12, cbar_orient="horizontal", label_color="b",
                    draw_borders=True):

    if draw_borders:
        draw_roi_borders(vertex_map, model)
    fig = cortex.quickflat.make_figure(vertex_map, fig=ax, with_colorbar=True) #,
                                       #vmin=np.min(vertex_map.data), vmax=np.max(vertex_map.data))

    cb_axes = fig.axes[-1]
    cb_axes.set_visible(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.2)
    plt.colorbar(mappable=cb_axes.collections[0], cax=cax, orientation=cbar_orient)

    add_labels(ax, model, fontsize=fontsize, color=label_color)


def plot_cond_diff_flat(model, ax, mean_diffs, ic_widths, **kwargs):
    vertex_map = cortex.db.get_surfinfo(model.subject.name, type="curvature")

    if "sources_all" in mean_diffs:
        data = (mean_diffs["sources_all"] / ic_widths["sources_all"]).mean("times").data
        vertex_map.data = sources_to_values(data, model, type_="all")

    elif "sources" in mean_diffs:
        vertex_map.data[:] = np.nan
        for no, label in enumerate(model.labels_surf):
            val = float((mean_diffs["sources"] / ic_widths["sources"]).mean("times").sel(signals=label.name))
            if label.hemi == "rh":
                vertex_map.data[label.vertices + model.surface_src[0]["np"]] = val
            else:
                vertex_map.data[label.vertices] = val

    plot_vertex_map(vertex_map, ax, model, **kwargs)


def surf_to_values(data, model):
    """
     Data is an array of values for each vertex.
    """
    surface_src = model.surface_src
    vertno = np.concatenate([surface_src[0]['vertno'],
                             surface_src[1]['vertno'] + surface_src[0]["np"]])
    nearests = np.concatenate([surface_src[0]['nearest'],
                               surface_src[1]['nearest'] + surface_src[0]["np"]])
    ind_map = {no: ind for ind, no in enumerate(vertno)}
    inds = [ind_map[n] for n in nearests]
    return data[inds]


def labels_to_values(data, model):
    """
     Data is a dictionary of {label_name:values}
    """

    labels_parc = model.labels_surf
    surface_src = model.surface_src
    nearests = np.concatenate([surface_src[0]['nearest'],
                               surface_src[1]['nearest'] + surface_src[0]["np"]])
    rh_offset = surface_src[0]["np"]
    label_vect_dict = {}
    for label in labels_parc:
        if label.hemi == "lh":
            label_vect_dict.update({v: label.name for v in label.vertices})
        else:
            label_vect_dict.update({v: label.name for v in label.vertices + rh_offset})

    return np.array([data[label_vect_dict[n]] if n in label_vect_dict else np.nan for n in nearests])


def sources_to_values(data, model, type_):
    if type_ == "all":
        return surf_to_values(data, model)
    if type_ == "labels":
        return labels_to_values(data, model)
    raise ValueError


def plot_sources_flat(model, recording, type_="all", figsize=(10, 10), **kwargs):

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    vertex_map = cortex.db.get_surfinfo(model.subject.name, type="curvature")
    data = model.get_sources(recording=recording, grouping="mean", type_=type_).to_array().mean("times").squeeze().load()
    if type_ == "all":
        data = data.data
    elif type_ == "labels":
        #data = model.get_sources(recording=recording, grouping="mean", type_=type_).to_array().mean("times").squeeze()
        data = {label: value for label, value in zip(data.coords["labels"].data, data.data)}

    vertex_map.data = sources_to_values(data, model, type_=type_)

    plot_vertex_map(vertex_map, ax, model, **kwargs)
