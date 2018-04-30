import os
import argparse
import pickle
import numpy as np
from sktensor import tucker
from sktensor.dtensor import dtensor

def main(args):
	name = os.path.basename(args.tensor)
	name = name.split('.')[0]
	with open(args.tensor, 'rb') as f:
		data = pickle.load(f).T
		filters, channels, cols, rows = data.shape
		channel_data = []
		col_data = []
		row_data = []
		for d in data:
			core, U = tucker.hooi(dtensor(d.T), [1, 1, 1], init='nvecs')
			core = np.squeeze(core)
			channel_data.append((core * U[2]).reshape(1, 1, channels)) # channels
			col_data.append(U[1].reshape(1, cols)) # cols
			row_data.append(U[0].reshape(rows, 1)) # rows

		channel_params = np.stack(channel_data, axis=-1)
		col_params = np.expand_dims(np.stack(col_data, axis=-1), axis=-1)
		row_params = np.expand_dims(np.stack(row_data, axis=-1), axis=-1)

	path = os.path.join(args.dest_dir, name + '_d.params')
	with open(path, 'w+') as f:
		pickle.dump(channel_params, f)

	path = os.path.join(args.dest_dir, name + '_h.params')
	with open(path, 'w+') as f:
		pickle.dump(col_params, f)

	path = os.path.join(args.dest_dir, name + '_v.params')
	with open(path, 'w+') as f:
		pickle.dump(row_params, f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
        '--tensor',
        type=str,
        help='Loss results')  
	parser.add_argument(
        '--dest_dir',
        type=str,
        help='Error results')
	args = parser.parse_args()
	main(args)