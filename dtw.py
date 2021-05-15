import numpy as np 

def computed_dtwMat(A, B):
    d = lambda x, y: np.abs(x - y)
    dtw_mat = np.zeros((len(B), len(A)))

    for i in range(len(B)):
        for j in range(len(A)):
            if i == 0 and j == 0:
                dtw_mat[i, j] = d(B[i], A[j])
            else:
                if i == 0 and j > 0:
                    choice = dtw_mat[i, j-1]
                elif i > 0 and j == 0:
                    choice = dtw_mat[i-1, j]
                else:
                    choice = [dtw_mat[i-1, j], dtw_mat[i, j-1], dtw_mat[i-1, j-1]]
                
                dtw_mat[i, j] = d(B[i], A[j]) + np.min(choice)

    return dtw_mat

def get_warpingPath(A, B):
    dtw_mat = computed_dtwMat(A, B)
    path = [[len(B)-1, len(A)-1]]
    while(True):
        i, j = path[-1][0], path[-1][1]
        if i == 0 and j == 0:
            break
        elif i == 0 and j > 0:
            path.append([i, j-1])
        elif i > 0 and j == 0:
            path.append([i-1, j])
        else:
            choice = [dtw_mat[i-1, j], dtw_mat[i, j-1], dtw_mat[i-1, j-1]]
            ind = [[i-1, j], [i, j-1], [i-1, j-1]]
            k = np.argmin(choice)
            path.append(ind[k])

    warp = np.zeros((len(B), len(A)))
    for p in path:
        warp[p[0], p[1]] = 1 

    return path, warp, dtw_mat

def normalized_dist(A, B):
    path, warp, dtw_mat = get_warpingPath(A, B)
    D = np.sum(warp * dtw_mat)/len(path)
    return D