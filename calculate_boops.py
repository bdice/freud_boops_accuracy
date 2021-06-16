import freud
import gsd
import gsd.hoomd
import numpy as np
import os
import sys


def read_dat_snapshot(filename):
    with open(filename, "r") as file:
        N = int(file.readline())
        s = file.readline().split()
        if len(s) == 3:
            boxMatrix = [float(s[0]), 0, 0, 0, float(s[1]), 0, 0, 0, float(s[2])]
        elif len(s) == 9:
            boxMatrix = np.array(
                [
                    [float(s[0]), float(s[1]), float(s[2])],
                    [float(s[3]), float(s[4]), float(s[5])],
                    [float(s[6]), float(s[7]), float(s[8])],
                ]
            )
        else:
            print(
                "Box information string has neither 3 nor 9 elements, likely error. Exiting."
            )
            sys.quit()
        positions = []
        for i in range(N):
            line = file.readline().split()
            posString = line[0:3]
            positions.append(
                [float(posString[0]), float(posString[1]), float(posString[2])]
            )

    # boxMatrix contains an arbitrarily oriented right-handed box matrix.
    v = [[], [], []]
    v[0] = boxMatrix[:, 0]
    v[1] = boxMatrix[:, 1]
    v[2] = boxMatrix[:, 2]
    Lx = np.sqrt(np.dot(v[0], v[0]))
    a2x = np.dot(v[0], v[1]) / Lx
    Ly = np.sqrt(np.dot(v[1], v[1]) - a2x * a2x)
    xy = a2x / Ly
    v0xv1 = np.cross(v[0], v[1])
    v0xv1mag = np.sqrt(np.dot(v0xv1, v0xv1))
    Lz = np.dot(v[2], v0xv1) / v0xv1mag
    a3x = np.dot(v[0], v[2]) / Lx
    xz = a3x / Lz
    yz = (np.dot(v[1], v[2]) - a2x * a3x) / (Ly * Lz)

    snap = gsd.hoomd.Snapshot()
    snap.particles.N = N
    snap.configuration.box = [Lx, Ly, Lz, xy, xz, yz]
    snap.particles.position = positions
    return snap


def compute_steinhardts(system, steinhardt_params, neighbors):
    qls = []
    for params in steinhardt_params:
        op = freud.order.Steinhardt(**params)
        op.compute(system, neighbors=neighbors)
        qls.append(op.particle_order)
    return np.array(qls).T


def compute_qls(system, neighbors, average=False, weighted=False):
    return compute_steinhardts(
        system,
        [
            dict(average=average, weighted=weighted, l=4),
            dict(average=average, weighted=weighted, l=6),
            dict(average=average, weighted=weighted, l=4, wl=True, wl_normalize=True),
            dict(average=average, weighted=weighted, l=6, wl=True, wl_normalize=True),
        ],
        neighbors,
    )


def compute_qls_and_neighbors(system, average=False, weighted=False):
    if weighted:
        # Use Voronoi neighbors
        voro = freud.locality.Voronoi().compute(system=system)
        nlist = voro.nlist
    else:
        # Use neighbors within radius
        nbQueryDict = dict(mode="ball", r_max=1.4, exclude_ii=True)
        nq = freud.locality.AABBQuery.from_system(system)
        nlist = (
            nq.from_system(system)
            .query(system.particles.position, nbQueryDict)
            .toNeighborList()
        )
    return compute_qls(system, nlist, average, weighted)


def compute_msms(system, lmax, average=False, wl=False):
    """Returns Minkowski Structure Metrics up to a maximum l value."""
    # Use Voronoi neighbors
    voro = freud.locality.Voronoi().compute(system=system)
    return compute_steinhardts(
        system,
        [
            dict(average=average, weighted=True, l=i, wl=wl, wl_normalize=wl)
            for i in range(lmax + 1)
        ],
        voro.nlist,
    )


def write_file(fname, qls):
    with open(fname, "w") as file:
        for row in qls:
            for el in row:
                file.write("%f " % el)
            file.write("\n")
    print(f"Saved to {fname}")


if __name__ == "__main__":
    print(f"Using freud {freud.__version__}")

    dir = os.path.dirname(os.path.abspath(sys.argv[1]))
    snap = read_dat_snapshot(sys.argv[1])

    # First calculate the non-averaged Steinhardt order parameters with cutoff neighbours
    qls = compute_qls_and_neighbors(snap, average=False, weighted=False)
    write_file(dir + "/Freud_rc1.4_q4q6w4w6.txt", qls)

    # Next the averaged ql/wl with cutoff neighbours
    qls = compute_qls_and_neighbors(snap, average=True, weighted=False)
    write_file(dir + "/Freud_rc1.4_avq4avq6avw4avw6.txt", qls)

    # Next, the non-averaged structure metrics
    qls = compute_qls_and_neighbors(snap, average=False, weighted=True)
    write_file(dir + "/Freud_MSM_q4q6w4w6.txt", qls)

    # And finally averaged structure metrics
    qls = compute_qls_and_neighbors(snap, average=True, weighted=True)
    write_file(dir + "/Freud_MSM_avq4avq6avw4avw6.txt", qls)
