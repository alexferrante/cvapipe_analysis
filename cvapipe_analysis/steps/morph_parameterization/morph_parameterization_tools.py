import argparse
import concurrent
import pandas as pd
import numpy as np
from pathlib import Path
from aicscytoparam import cytoparam

from cvapipe_analysis.tools import io, general, controller

class MorphParameterizer(io.DataProducer):
    def __init__(self, config):
        super().__init__(config)
        self.subfolder = 'morphparameterization/morph-representations'
        
    def workflow(self):
        self.pint = self.read_parameterized_intensity(self.row['CellId'])
        self.cell_mean_mesh = self.read_mean_shape_mesh(self.control.get_outer_most_alias_to_parameterize())
        self.nuc_mean_mesh = self.read_mean_shape_mesh(self.control.get_inner_most_alias_to_parameterize())
        self.morph_on_mean_shape()

    def get_voxelized_mesh(self):

        avg_mesh_file = Path(self.subfolder) / 'avg_shape_voxelized.tif'
        if not avg_mesh_file.is_file():
            # imagedata, shape, rmin = get_mesh_params(self.mean_mesh)
            # voxelized_mean_mesh = cytoparam.voxelize_mesh(imagedata, shape, self.mean_mesh, rmin)
            # self.write_ome_tif(avg_mesh_file, voxelized_mean_mesh)
            voxelized_mean_mesh_domain, voxelized_mean_mesh_origin = cytoparam.voxelize_meshes([self.cell_mean_mesh, self.nuc_mean_mesh])
            # self.write_ome_tif(avg_mesh_file, voxelized_mean_mesh)
        else:
            # imagedata, shape, rmin = get_mesh_params(self.mean_mesh)
            # voxelized_mean_mesh = cytoparam.voxelize_mesh(imagedata, shape, self.mean_mesh, rmin)
            # voxelized_mean_mesh_domain, voxelized_mean_mesh_origin = cytoparam.voxelize_meshes([self.cell_mean_mesh, self.nuc_mean_mesh])
            # img_obj = AICSImage(avg_mesh_file)
            # voxelized_mean_mesh = img_obj.data.squeeze()
            voxelized_mean_mesh_domain, voxelized_mean_mesh_origin = cytoparam.voxelize_meshes([self.cell_mean_mesh, self.nuc_mean_mesh])
        return voxelized_mean_mesh_domain, voxelized_mean_mesh_origin

    def morph_on_mean_shape(self):
        n = self.control.get_number_of_interpolating_points()
        mean_mesh_img_domain, mean_mesh_img_origin = self.get_voxelized_mesh()
        coords_param, _ = cytoparam.parameterize_image_coordinates(
            seg_mem=(mean_mesh_img_domain>0).astype(np.uint8),
            seg_nuc=(mean_mesh_img_domain>1).astype(np.uint8),
            lmax=self.control.get_lmax(), 
            nisos=[n, n]
        )
        self.morphed = cytoparam.morph_representation_on_shape(
            img=mean_mesh_img_domain,
            param_img_coords=coords_param,
            representation=self.pint
        )
        self.morphed = np.stack([mean_mesh_img_domain, self.morphed])
        return

    def get_output_file_name(self):
        return f"{self.row.CellId}.tif"

    def save(self):
        save_as = self.get_output_file_path()
        img = self.morphed
        self.write_ome_tif(
            save_as, img, channel_names=['domain', save_as.stem]
        )
        return save_as


if __name__ == "__main__":

    config = general.load_config_file()
    control = controller.Controller(config)

    parser = argparse.ArgumentParser(description='Batch parameterization via morphed representation.')
    parser.add_argument('--csv', help='Path to the dataframe.', required=True)
    args = vars(parser.parse_args())

    df = pd.read_csv(args['csv'])
    print(f"Processing dataframe of shape {df.shape}")

    morph_parameterizer = MorphParameterizer(control)
    with concurrent.futures.ProcessPoolExecutor(control.get_ncores()) as executor:
        executor.map(morph_parameterizer.execute, [row for _,row in df.iterrows()])