import pickle
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.style as style
# style.use('bmh')

def read_data(file):
	with open(file, 'r') as f:
		return pickle.load(f)

def main(args):
	loss = read_data(args.loss)
	error = read_data(args.error)
	epoch = read_data(args.epoch)

	fig, ax1 = plt.subplots()
	ax1.plot(epoch, loss)
	ax1.set_xlabel('epoch')
	# Make the y-axis label, ticks and tick labels match the line color.
	ax1.set_ylabel('loss', color='#345995')
	ax1.tick_params('y', colors='#345995')

	ax2 = ax1.twinx()
	ax2.plot(epoch, error, 'r')
	ax2.set_ylabel('error', color='#FB4D3D')
	ax2.tick_params('y', colors='#FB4D3D')

	plt.title(args.title)
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
        '--loss',
        type=str,
        help='Loss results')  
	parser.add_argument(
        '--error',
        type=str,
        help='Error results')
	parser.add_argument(
        '--epoch',
        type=str,
        help='Epoch results')
	parser.add_argument(
        '--title',
        type=str,
        help='Title')
	args = parser.parse_args()
	main(args)