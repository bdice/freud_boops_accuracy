from calculate_boops import read_dat_snapshot, compute_qls_and_neighbors
import gsd.hoomd
import numpy as np
import numpy.testing as npt
import pytest


def read_gsd_snapshot(filename):
    with gsd.hoomd.open(filename) as traj:
        gsd_frame = traj[-1]
    return gsd_frame


class TestInputData:
    def test_dat_gsd_box_and_positions_equal(self):
        """Make sure the dat file and gsd file have equivalent box/positions."""
        gsd_frame = read_gsd_snapshot("Test_Configuration.gsd")
        dat_frame = read_dat_snapshot("Test_Configuration.dat")

        dat_frame.particles.position -= np.array(dat_frame.configuration.box[:3]) / 2

        npt.assert_allclose(dat_frame.configuration.box, gsd_frame.configuration.box)
        npt.assert_allclose(
            dat_frame.particles.position, gsd_frame.particles.position, atol=2e-6
        )

    @pytest.mark.parametrize(
        "average,weighted", [(False, False), (True, False), (False, True), (True, True)]
    )
    def test_dat_gsd_steinhardts_equal(self, average, weighted):
        """Make sure the dat file and gsd file have equivalent Steinhardt values from freud."""
        gsd_frame = read_gsd_snapshot("Test_Configuration.gsd")
        dat_frame = read_dat_snapshot("Test_Configuration.dat")

        dat_frame.particles.position -= np.array(dat_frame.configuration.box[:3]) / 2

        dat_qls = compute_qls_and_neighbors(dat_frame, average, weighted)
        gsd_qls = compute_qls_and_neighbors(gsd_frame, average, weighted)
        names = ["q4", "q6", "w4", "w6"]
        for i, name in enumerate(names):
            npt.assert_allclose(
                dat_qls[:, i], gsd_qls[:, i], atol=2e-5, err_msg=f"{name} failed"
            )


class TestSteinhardtValues:
    def test_gc_radius(self):
        gc_data = np.genfromtxt("GC_rc1.4_q4q6w4w6.txt")
        gsd_frame = read_gsd_snapshot("Test_Configuration.gsd")
        freud_data = compute_qls_and_neighbors(gsd_frame, average=False, weighted=False)
        names = ["q4", "q6", "w4", "w6"]
        for i, name in enumerate(names):
            npt.assert_allclose(
                gc_data[:, i], freud_data[:, i], atol=2e-5, err_msg=f"{name} failed"
            )

    def test_gc_radius_ave(self):
        gc_data = np.genfromtxt("GC_rc1.4_avq4avq6avw4avw6.txt")
        gsd_frame = read_gsd_snapshot("Test_Configuration.gsd")
        freud_data = compute_qls_and_neighbors(gsd_frame, average=True, weighted=False)
        names = ["ave. q4", "ave. q6", "ave. w4", "ave. w6"]
        for i, name in enumerate(names):
            npt.assert_allclose(
                gc_data[:, i], freud_data[:, i], atol=2e-5, err_msg=f"{name} failed"
            )

    def test_rvd_msm(self):
        rvd_data = np.genfromtxt("RvD_MSM_q4q6w4w6.txt")
        gsd_frame = read_gsd_snapshot("Test_Configuration.gsd")
        freud_data = compute_qls_and_neighbors(gsd_frame, average=True, weighted=False)
        names = ["q'4", "q'6", "w'4", "w'6"]
        for i, name in enumerate(names):
            npt.assert_allclose(
                rvd_data[:, i], freud_data[:, i], atol=2e-5, err_msg=f"{name} failed"
            )

    def test_rvd_msm_ave(self):
        rvd_data = np.genfromtxt("RvD_MSM_avq4avq6avw4avw6.txt")
        gsd_frame = read_gsd_snapshot("Test_Configuration.gsd")
        freud_data = compute_qls_and_neighbors(gsd_frame, average=True, weighted=False)
        names = ["ave. q'4", "ave. q'6", "ave. w'4", "ave. w'6"]
        for i, name in enumerate(names):
            npt.assert_allclose(
                rvd_data[:, i], freud_data[:, i], atol=2e-5, err_msg=f"{name} failed"
            )
