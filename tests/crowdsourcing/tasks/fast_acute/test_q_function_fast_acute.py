#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
End-to-end testing for the Fast ACUTE crowdsourcing task.
"""

import json
import os
import shutil
import tempfile
import unittest

import pytest


try:

    from parlai.crowdsourcing.tasks.fast_acute.run_q_function import (
        QLearningFastAcuteExecutor,
        BLUEPRINT_TYPE,
    )
    from parlai.crowdsourcing.tasks.fast_acute.util import AbstractFastAcuteTest

    class TestQFunctionFastAcute(AbstractFastAcuteTest):
        """
        Test the Q-function Fast ACUTE crowdsourcing task.
        """

        @pytest.fixture(scope="module")
        def setup_teardown(self):
            """
            Call code to set up and tear down tests.

            Run this only once because we'll be running all Fast ACUTE code before
            checking any results.
            """

            self._setup()

            # Set up common temp directory
            root_dir = tempfile.mkdtemp()

            # Params
            config_path = os.path.join(root_dir, 'config.json')

            # Copy over expected self-chat files
            shutil.copytree(
                os.path.join(self.FAST_ACUTE_TASK_DIRECTORY, 'self_chats'),
                os.path.join(root_dir, 'self_chats'),
            )

            # Define output structure
            outputs = {}

            # # Run Q-function Fast ACUTEs and analysis on the base task

            # Set up config
            assert len(self.MODELS) == 2
            q_function_overrides = [
                f'+mephisto.blueprint.config_path={config_path}',
                '+mephisto.blueprint.models=""',
                f'+mephisto.blueprint.model_pairs={self.MODELS[0]}:{self.MODELS[1]}',
            ]
            # TODO: clean this up when Hydra has support for recursive defaults
            self._set_up_config(
                blueprint_type=BLUEPRINT_TYPE,
                task_directory=self.ACUTE_EVAL_TASK_DIRECTORY,
                overrides=self._get_common_overrides(root_dir) + q_function_overrides,
            )
            self.config.mephisto.blueprint.models = None
            # TODO: hack to manually set mephisto.blueprint.models to None. Remove when
            #  Hydra releases support for recursive defaults

            # Save the config file
            config = {}
            for model in self.MODELS:
                config[model] = {
                    'log_path': QLearningFastAcuteExecutor.get_relative_selfchat_log_path(
                        root_dir=self.config.mephisto.blueprint.root_dir,
                        model=model,
                        task=self.config.mephisto.blueprint.task,
                    ),
                    'is_selfchat': True,
                }
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Run Fast ACUTEs
            runner = QLearningFastAcuteExecutor(self.config)
            runner.set_up_acute_eval()
            self.config.mephisto.blueprint = runner.fast_acute_args
            self._set_up_server()
            outputs['state'] = self._get_agent_state(task_data=self.TASK_DATA)

            # Run analysis
            runner.analyze_results(args=f'--mephisto-root {self.database_path}')
            outputs['results_folder'] = runner.results_path

            yield outputs
            # All code after this will be run upon teardown

            self._teardown()

            # Tear down temp file
            shutil.rmtree(root_dir)


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
