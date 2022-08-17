#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from datastep import Step, log_run_params
from typing import Dict, List, Optional, Union

import concurrent
from cvapipe_analysis.tools import io, general, cluster
from .morph_parameterization_tools import MorphParameterizer

log = logging.getLogger(__name__)

class MorphParameterization(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(self, distribute: Optional[bool]=False, **kwargs):

        with general.configuration(self.step_local_staging_dir) as control:

            device = io.LocalStagingIO(control)
            df = device.load_step_manifest("preprocessing")
            log.info(f"Manifest: {df.shape}")

            save_dir = self.step_local_staging_dir/"morph-representations"
            save_dir.mkdir(parents=True, exist_ok=True)

            if distribute:

                distributor = cluster.MorphParameterizationDistributor(control)
                distributor.set_data(df)
                distributor.distribute()
                log.info(f"Multiple jobs have been launched. Please come back when the calculation is complete.")
                
                return None

            morph_parameterizer = MorphParameterizer(control)
            with concurrent.futures.ProcessPoolExecutor(control.get_ncores()) as executor:
                executor.map(morph_parameterizer.execute, [row for _,row in df.iterrows()])
