import numpy as np

def grid2pos(occurence_matrix, size):
	dimx, dimy = occurence_matrix.shape[0], occurence_matrix.shape[1]
	assert(dimx == dimy)
	assert((dimx-2)*size==2)
	midx, midy = dimx//2, dimy//2
	positions = []
	for i in range(dimx):
		for j in range(dimy):
			if occurence_matrix[i, j]:
				m, n = j, i
				posx, posy = ((m - midx)*size + size/2), ((midy-n)*size - size/2)
				posx = float(format(posx, '.2f'))
				posy = float(format(posy, '.2f'))
				positions.append([posx, posy])

	return positions

def getnumberofwall(occurence_matrix):
	return (np.sum(occurence_matrix))

grid1 = np.array([
	[0,0,0,1,0,0,1,0,0,0],
	[0,0,0,1,0,0,1,0,0,0],
	[0,0,0,1,0,0,1,0,0,0],
	[1,1,1,1,0,0,1,1,1,1],
	[0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0],
	[1,1,1,1,0,0,1,1,1,1],
	[0,0,0,1,0,0,1,0,0,0],
	[0,0,0,1,0,0,1,0,0,0],
	[0,0,0,1,0,0,1,0,0,0]
])

# from pprint import pprint
# pprint(grid2pos(grid1, 0.2))

# print(getnumberofwall(grid1))