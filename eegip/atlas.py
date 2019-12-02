import pandas as pd
import os
import trimesh
import numpy as np
from skimage import measure 
import mne
from mne import setup_volume_source_space, setup_source_space, \
    make_bem_model, make_bem_solution

from .config import eegip_config


center_of_masses = pd.read_csv(os.path.join(eegip_config['paths']['atlas_dir'], "center_of_masses.csv"), index_col=0)


def build_bem(fs_subject="", subjects_dir=None, surface="pial",
              spacing='oct6', add_dist=False, ico=None, pos=0.5, labels_vol=None,
              fname_aseg=None, fname_src=None, fname_bem=None, add_interpolator=True):

    if subjects_dir is None:
        atlas_dir = eegip_config['paths']['atlas_dir']
    else:
        atlas_dir = subjects_dir
    if fname_aseg is None:
        fname_aseg = os.path.join(atlas_dir, fs_subject, 'mri', 'aseg.mgz')        
    if fname_src is None:
        fname_src = os.path.join(atlas_dir, fs_subject, 'infant_atlas-src.fif')
    if fname_bem is None:
        fname_bem = os.path.join(atlas_dir, fs_subject, 'infant_atlas-bem')
    if labels_vol is None:
        # List of sub structures we are interested in. We select only the
        # sub structures we want to include in the source space
        labels_vol = ['Left-Amygdala',
                      'Left-Thalamus-Proper',
                      'Left-Cerebellum-Cortex',
                      'Brain-Stem',
                      'Right-Amygdala',
                      'Right-Thalamus-Proper',
                      'Right-Cerebellum-Cortex']
        
    src = setup_source_space(fs_subject, subjects_dir=atlas_dir, surface=surface,
                             spacing=spacing, add_dist=add_dist)

    model = make_bem_model(subject=fs_subject, ico=ico, subjects_dir=atlas_dir)
    solution = make_bem_solution(model)  # does the potential computations
    # Setup a volume source space
    # set pos=10.0 for speed, not very accurate; we recommend something smaller
    # like 5.0 in actual analyses:
    vol_src = setup_volume_source_space(pos=pos, bem=solution, add_interpolator=add_interpolator,
                                        mri=fname_aseg, volume_label=labels_vol, subjects_dir=atlas_dir)
    # Generate the mixed source space
    src += vol_src

    src.save(fname_src)
    mne.write_bem_solution(fname_bem, solution)


def mesh_atlas_pacels(labels_parc, src, atlas_dir):
    # src should contains the two hemisphere first and then the volume sources
    src_hemi = src[:2]
    vol_src = src[2:]
    vertices_lst = []
    simplices_lst = []
    labels = []

    for label in labels_parc:
        print(label.name)

        if label.hemi == "lh":
            src_obj = src[0]
        elif label.hemi == "rh":
            src_obj = src[1]
        else:
            raise ValueError

        all_vertices_no = src_obj["vertno"]
        label_vertices_no = label.get_vertices_used(all_vertices_no)

        all_used_tris = src_obj["use_tris"]  # INDEXED STARTING AT 0
        label_tris = label.get_tris(all_used_tris, all_vertices_no)
        label_tris_reindexed = np.array([label_vertices_no.tolist().index(a) 
                                         for a in label_tris.flatten()]).reshape(label_tris.shape)

        vertices_lst.append(src_obj["rr"][label_vertices_no])
        simplices_lst.append(label_tris_reindexed)
        labels.append(label.name)           

    for label, src_obj in zip(["left_hemi", "right_hemi"], src_hemi):
        print(label)
        all_vertices_no = src_obj["vertno"]
        all_used_tris = src_obj["use_tris"] 
        tris_reindexed = np.array([all_vertices_no.tolist().index(a)                                          
                                   for a in all_used_tris.flatten()]).reshape(all_used_tris.shape)

        vertices_lst.append(src_obj["rr"][all_vertices_no])
        simplices_lst.append(tris_reindexed)
        labels.append(label)           

    for src_obj in vol_src:
        roi_str = src_obj["seg_name"]
        if 'left' in roi_str.lower():
            label = roi_str.replace('Left-', '') + '-lh'
        elif 'right' in roi_str.lower():
            label = roi_str.replace('Right-', '') + '-rh'   
        else:
            label = roi_str
        print(label)        

        points = src_obj["rr"][src_obj["vertno"]]         
        points_4d = np.hstack((points, np.ones((points.shape[0], 1))))
        trans_points = np.linalg.inv(src_obj['src_mri_t']["trans"])  @  points_4d.T
        volumetric_point_data = np.zeros(src_obj["shape"])
        for ix, iy, iz in np.round(trans_points[:3, :]).T.astype(int):
            volumetric_point_data[ix, iy, iz] = 1

        vertices, simplices = measure.marching_cubes_lewiner(volumetric_point_data, spacing=(1, 1, 1),
                                                             allow_degenerate=False)[:2]    

        vertices = src_obj['src_mri_t']["trans"] @ np.hstack((vertices, np.ones((vertices.shape[0], 1)))).T

        vertices_lst.append(vertices[:3, :].T)
        simplices_lst.append(simplices)
        labels.append(label)          

    for label, vertices, simplices in zip(labels, vertices_lst, simplices_lst):
        mesh = trimesh.Trimesh(vertices=vertices, faces=simplices)
        trimesh.repair.fix_normals(mesh, multibody=False)
        with open(os.path.join(atlas_dir, "meshes", label + ".obj"), 'w') as file_obj:
             file_obj.write(trimesh.exchange.obj.export_obj(mesh))     

    
def compute_pacels_centers_of_masse(src, labels_parc, atlas_dir, fs_subject=""):
    center_of_masses_dict = {}
    for src_obj in src[2:]:
        roi_str = src_obj["seg_name"]
        if 'left' in roi_str.lower():
            roi_str = roi_str.replace('Left-', '') + '-lh'
        elif 'right' in roi_str.lower():
            roi_str = roi_str.replace('Right-', '') + '-rh'    

        center_of_masses_dict[roi_str] = np.average(src_obj['rr'][src_obj["vertno"]], axis=0)

    for label in labels_parc:
        ind_com = np.where(label.vertices == label.center_of_mass(subject=fs_subject, subjects_dir=atlas_dir))[0]
        if len(label.pos[ind_com, :]):
            center_of_masses_dict[label.name] = label.pos[ind_com, :][0]

    center_of_masses_df = pd.DataFrame(center_of_masses_dict).T
    center_of_masses_df.columns = ["x", "y", "z"]
    center_of_masses_df.to_csv(os.path.join(atlas_dir, "center_of_masses.csv"))
