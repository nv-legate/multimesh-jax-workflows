import numpy as np
import unittest
from unittest.mock import patch
import dataclasses
from pathlib import Path
from multimesh_jax_workflows import get_maxtext_results


@dataclasses.dataclass
class DummyRunResult:
    stdout: str
    stderr: str
    returncode: int


RUN_CMD_RESULT = DummyRunResult(stdout="", stderr="", returncode=0)

FAKE_JAX_OUT = """
completed step: 0, seconds: 1.285, TFLOP/s/device: 3466.475, Tokens/s/device: 12750.214, loss: 2659.990
To see full metrics 'tensorboard --logdir=/workspace/cwd/0/logs/my_name/tensorboard/'
completed step: 1, seconds: 2.828, TFLOP/s/device: 1575.376, Tokens/s/device: 5794.467, loss: 2659.990
completed step: 2, seconds: 12.768, TFLOP/s/device: 348.873, Tokens/s/device: 1283.208, loss: 557.604
completed step: 3, seconds: 9.163, TFLOP/s/device: 486.111, Tokens/s/device: 1787.989, loss: 462.564
completed step: 4, seconds: 9.128, TFLOP/s/device: 488.021, Tokens/s/device: 1795.015, loss: 443.733
completed step: 5, seconds: 9.141, TFLOP/s/device: 487.309, Tokens/s/device: 1792.397, loss: 440.809
completed step: 6, seconds: 9.105, TFLOP/s/device: 489.210, Tokens/s/device: 1799.389, loss: 431.128
completed step: 7, seconds: 9.088, TFLOP/s/device: 490.117, Tokens/s/device: 1802.724, loss: 426.007
completed step: 8, seconds: 9.076, TFLOP/s/device: 490.789, Tokens/s/device: 1805.195, loss: 399.825
completed step: 9, seconds: 9.118, TFLOP/s/device: 488.513, Tokens/s/device: 1796.825, loss: 386.619
completed step: 10, seconds: 9.089, TFLOP/s/device: 490.063, Tokens/s/device: 1802.523, loss: 378.499
completed step: 11, seconds: 9.100, TFLOP/s/device: 489.497, Tokens/s/device: 1800.442, loss: 373.285
"""  # noqa

FAKE_JAX_LOG = """
[1 - 7ffff7c93740]  251.706088 {1}{multimesh.jax}: CreateStore: device=0, shape=ShardedShape(type=3,Sharding(devices[0...16),dims={ShardingDim(size=16384,shard=8,perm=2),ShardingDim(size=12288,shard=1,perm=0),ShardingDim(size=1,shard=2,perm=1),})), name=get-tuple-element.23856
[1 - 7ffff7c93740]  251.706118 {1}{multimesh.jax}: CreateStore: device=0, shape=ShardedShape(type=3,Sharding(devices[0...16),dims={ShardingDim(size=12288,shard=1,perm=2),ShardingDim(size=3,shard=1,perm=3),ShardingDim(size=96,shard=8,perm=4),ShardingDim(size=128,shard=1,perm=0),ShardingDim(size=1,shard=2,perm=1),})), name=get-tuple-element.23857
[1 - 7ffff7c93740]  251.706144 {1}{multimesh.jax}: CreateStore: device=0, shape=ShardedShape(type=3,Sharding(devices[0...16),dims={ShardingDim(size=3,shard=1,perm=2),ShardingDim(size=96,shard=8,perm=3),ShardingDim(size=128,shard=1,perm=0),ShardingDim(size=1,shard=2,perm=1),})), name=get-tuple-element.23858
[1 - 7ffff7c93740]  251.706169 {1}{multimesh.jax}: CreateStore: device=0, shape=ShardedShape(type=3,Sharding(devices[0...16),dims={ShardingDim(size=96,shard=8,perm=3),ShardingDim(size=128,shard=1,perm=0),ShardingDim(size=12288,shard=1,perm=1),ShardingDim(size=1,shard=2,perm=2),})), name=get-tuple-element.23859
[1 - 7ffff7c93740]  251.706200 {1}{multimesh.jax}: CreateStore: device=0, shape=ShardedShape(type=3,Sharding(devices[0...16),dims={ShardingDim(size=12288,shard=8,perm=1),ShardingDim(size=1,shard=2,perm=0),})), name=get-tuple-element.23860
[1 - 7ffff7c93740]  251.706225 {1}{multimesh.jax}: CreateStore: device=0, shape=ShardedShape(type=3,Sharding(devices[0...16),dims={ShardingDim(size=12288,shard=8,perm=1),ShardingDim(size=1,shard=2,perm=0),})), name=get-tuple-element.23861
[1 - 7ffff7c93740]  251.706247 {1}{multimesh.jax}: CreateStore: device=0, shape=ShardedShape(type=3,Sharding(devices[0...16),dims={ShardingDim(size=12288,shard=8,perm=1),ShardingDim(size=1,shard=2,perm=0),})), name=get-tuple-element.23862
[1 - 7ffff7c93740]  251.706275 {1}{multimesh.jax}: CreateStore: device=0, shape=ShardedShape(type=3,Sharding(devices[0...16),dims={ShardingDim(size=49152,shard=8,perm=2),ShardingDim(size=12288,shard=1,perm=0),ShardingDim(size=1,shard=2,perm=1),})), name=get-tuple-element.23863
[1 - 7ffff7c93740]  251.706303 {1}{multimesh.jax}: CreateStore: device=0, shape=ShardedShape(type=3,Sharding(devices[0...16),dims={ShardingDim(size=12288,shard=8,perm=1),ShardingDim(size=1,shard=2,perm=0),})), name=get-tuple-element.23864
[0 - 7fe83c272740]  237.324541 {2}{multimesh.jax}: pjit_autoshard_train_fn finished in 9.14151s
[0 - 7fe83c272740]  246.423823 {2}{multimesh.jax}: pjit_autoshard_train_fn finished in 9.09944s
[0 - 7fe83c272740]  255.510948 {2}{multimesh.jax}: pjit_autoshard_train_fn finished in 9.08728s
[0 - 7fe83c272740]  264.588168 {2}{multimesh.jax}: pjit_autoshard_train_fn finished in 9.07738s
[0 - 7fe83c272740]  273.705849 {2}{multimesh.jax}: pjit_autoshard_train_fn finished in 9.11784s
[0 - 7fe83c272740]  282.794389 {2}{multimesh.jax}: pjit_autoshard_train_fn finished in 9.08869s
"""  # noqa
FAKE_JAX_MEMORY_TOTAL = 704686080


def mock_get_file_text(path: str | Path):
    if ".log" in str(path):
        return FAKE_JAX_LOG
    if ".out" in str(path):
        return FAKE_JAX_OUT
    return ""


class MaxtextTest(unittest.TestCase):
    @patch("eos_workflows.slurm.run_cmdline", return_value=RUN_CMD_RESULT)
    @patch("multimesh_jax_workflows.parse_logs.get_file_text", mock_get_file_text)
    def test_get_maxtext_results(self, mock_run_cmdline):
        timings, losses, memory, mfus, _, rc = get_maxtext_results(
            image="test", job_folder="test", model="gpt3-175b", gpus=8
        )
        mock_run_cmdline.assert_called_once()

        self.assertEqual(rc, 0)
        self.assertGreater(len(timings), 0)
        self.assertGreater(np.mean(timings), 9.0)
        self.assertGreater(np.mean(mfus[4:]), 480)
        self.assertEqual(memory, FAKE_JAX_MEMORY_TOTAL)


if __name__ == "__main__":
    unittest.main()