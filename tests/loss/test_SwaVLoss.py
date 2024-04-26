import unittest

import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist

from lightly.loss import SwaVLoss


class TestNTXentLoss:
    def test__sinkhorn_gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        SwaVLoss(sinkhorn_gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__sinkhorn_gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            SwaVLoss(sinkhorn_gather_distributed=True)
        mock_is_available.assert_called_once()


class TestSwaVLossUnitTest(unittest.TestCase):
# Updated code snippet:
# - Convert the existing test method from unittest style to pytest style.
# - Add the test method to the TestSwavLoss class.
# - Use pytest.mark.parametrize to iterate over different values for testing.

import pytest
import torch
from tests.loss.test_SwaVLoss import TestSwavLoss
from your_module import SwaVLoss

class TestSwavLoss:
    
    @pytest.mark.parametrize("n_low_res, sinkhorn_iterations", [(0, 0), (1, 1), (2, 2)])
    def test_forward_pass(self, n_low_res, sinkhorn_iterations):
        n = 32
        n_high_res = 2
        high_res = [torch.eye(32, 32) for i in range(n_high_res)]
        criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
        low_res = [torch.eye(n, n) for i in range(n_low_res)]

                with self.subTest(
                    msg=f"n_low_res={n_low_res}, sinkhorn_iterations={sinkhorn_iterations}"
                ):
                    loss = criterion(high_res, low_res)
                    # loss should be almost zero for unit matrix
                    self.assertGreater(0.5, loss.cpu().numpy())

    def test_forward_pass_queue(self):
        n = 32
        n_high_res = 2
        high_res = [torch.eye(32, 32) for i in range(n_high_res)]
        queue_length = 128
        queue = [torch.eye(128, 32) for i in range(n_high_res)]

        for n_low_res in range(6):
            for sinkhorn_iterations in range(3):
                criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
                low_res = [torch.eye(n, n) for i in range(n_low_res)]

                with self.subTest(
                    msg=f"n_low_res={n_low_res}, sinkhorn_iterations={sinkhorn_iterations}"
                ):
                    loss = criterion(high_res, low_res, queue)
                    # loss should be almost zero for unit matrix
                    self.assertGreater(0.5, loss.cpu().numpy())

    def test_forward_pass_bsz_1(self):
        n = 32
        n_high_res = 2
        high_res = [torch.eye(1, n) for i in range(n_high_res)]

        for n_low_res in range(6):
            for sinkhorn_iterations in range(3):
                criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
                low_res = [torch.eye(1, n) for i in range(n_low_res)]

                with self.subTest(
                    msg=f"n_low_res={n_low_res}, sinkhorn_iterations={sinkhorn_iterations}"
                ):
                    loss = criterion(high_res, low_res)

    def test_forward_pass_1d(self):
        n = 32
        n_high_res = 2
        high_res = [torch.eye(n, 1) for i in range(n_high_res)]

        for n_low_res in range(6):
            for sinkhorn_iterations in range(3):
                criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
                low_res = [torch.eye(n, 1) for i in range(n_low_res)]

                with self.subTest(
                    msg=f"n_low_res={n_low_res}, sinkhorn_iterations={sinkhorn_iterations}"
                ):
                    loss = criterion(high_res, low_res)
                    # loss should be almost zero for unit matrix
                    self.assertGreater(0.5, loss.cpu().numpy())

    @unittest.skipUnless(torch.cuda.is_available(), "skip")
    def test_forward_pass_cuda(self):
        n = 32
        n_high_res = 2
        high_res = [torch.eye(n, n).cuda() for i in range(n_high_res)]

        for n_low_res in range(6):
            for sinkhorn_iterations in range(3):
                criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
                low_res = [torch.eye(n, n).cuda() for i in range(n_low_res)]

                with self.subTest(
                    msg=f"n_low_res={n_low_res}, sinkhorn_iterations={sinkhorn_iterations}"
                ):
                    loss = criterion(high_res, low_res)
                    # loss should be almost zero for unit matrix
                    self.assertGreater(0.5, loss.cpu().numpy())
