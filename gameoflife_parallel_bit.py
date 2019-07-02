# Python code to implement Conway's Game Of Life
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import cuda


# setting up the values for the grid
ON = 1
OFF = 0
vals = [ON, OFF]


def randomGrid(N):
    """returns a grid of NxN random values"""
    return np.random.choice(vals, N * N, p=[0.1, 0.9]).reshape(N, N)


def addGlider(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0, 0, 1],
                       [1, 0, 1],
                       [0, 1, 1]])
    grid[i:i + 3, j:j + 3] = glider


def addGosperGliderGun(i, j, grid):
    """adds a Gosper Glider Gun with top left
       cell at (i, j)"""
    gun = np.zeros(11 * 38).reshape(11, 38)

    gun[5][1] = gun[5][2] = 1
    gun[6][1] = gun[6][2] = 1

    gun[3][13] = gun[3][14] = 1
    gun[4][12] = gun[4][16] = 1
    gun[5][11] = gun[5][17] = 1
    gun[6][11] = gun[6][15] = gun[6][17] = gun[6][18] = 1
    gun[7][11] = gun[7][17] = 1
    gun[8][12] = gun[8][16] = 1
    gun[9][13] = gun[9][14] = 1

    gun[1][25] = 1
    gun[2][23] = gun[2][25] = 1
    gun[3][21] = gun[3][22] = 1
    gun[4][21] = gun[4][22] = 1
    gun[5][21] = gun[5][22] = 1
    gun[6][23] = gun[6][25] = 1
    gun[7][25] = 1

    gun[3][35] = gun[3][36] = 1
    gun[4][35] = gun[4][36] = 1

    grid[i:i + 11, j:j + 38] = gun

TPB = 128


@cuda.jit(device=True)
def update_bit_parallel_helper(d_grid, m, n, i, j):
    """
    Computes and returns the value of the function g(x0)
    """
    live = 0
    for p in range(i - 1, i + 2):
        for q in range(j - 1, j + 2):
            live += d_grid[p % m, q % n] & 1
    if live == 3 or live == d_grid[i, j] + 3:
        d_grid[i, j] += 2


@cuda.jit()
def Kernel_update(d_grid, N):

    i, j = cuda.grid(2)
    if i < N and j < N:
        update_bit_parallel_helper(d_grid, N, N, i, j)

@cuda.jit()
def Kernel_shift(d_grid, N):

    i, j = cuda.grid(2)
    if i < N and j < N:
        update_shift(d_grid, i, j)

@cuda.jit(device=True)
def update_shift(d_grid, i, j):
    d_grid[i, j] >>= 1


def update_bit_parallel(frame, img, grid, N):
    d_grid = cuda.to_device(grid)
    NX, NY = N, N  # Array size
    TPBX, TPBY = 4, 4

    threads = (TPBX, TPBY)
    blocks = ((NX + TPBX - 1) // TPBX, (NY + TPBY - 1) // TPBY)
    Kernel_update[blocks, threads](d_grid, N)
    Kernel_shift[blocks, threads](d_grid, N)

    new_grid = d_grid.copy_to_host()
    grid[:] = new_grid[:]
    img.set_data(grid)
    return img,




# main() function
def main():
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life simulation.")

    # add arguments
    parser.add_argument('--grid-size', dest='N', required=False)
    parser.add_argument('--mov-file', dest='movfile', required=False)
    parser.add_argument('--interval', dest='interval', required=False)
    parser.add_argument('--glider', action='store_true', required=False)
    parser.add_argument('--gosper', action='store_true', required=False)
    args = parser.parse_args()

    # set grid size
    N = 500
    if args.N and int(args.N) > 8:
        N = int(args.N)

        # set animation update interval
    updateInterval = 50
    if args.interval:
        updateInterval = int(args.interval)

    # declare grid
    grid = np.array([])

    # check if "glider" demo flag is specified
    if args.glider:
        grid = np.zeros(N * N).reshape(N, N)
        addGlider(1, 1, grid)
    elif args.gosper:
        grid = np.zeros(N * N).reshape(N, N)
        addGosperGliderGun(10, 10, grid)

    else:  # populate grid with random on/off -
        # more off than on
        grid = randomGrid(N)

        # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, update_bit_parallel, fargs=(img, grid, N,),
                                  frames=10,
                                  interval=updateInterval,
                                  save_count=50)

    # # of frames?
    # set output file
    if args.movfile:
        ani.save(args.movfile, fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


# call main
if __name__ == '__main__':
    main()