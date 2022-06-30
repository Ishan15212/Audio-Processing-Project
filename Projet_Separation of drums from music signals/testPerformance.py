import librosa as lb
from librosa.display import specshow
import numpy as np
from Separator import separate
import matplotlib.pyplot as plt 
import sys

def test_different_gammas(filename, y_array = np.linspace(0, 1, 5)):

	plt.figure(figsize=(12, 10))

	i = 1

	audioOG, srOG = lb.load(filename, sr=None)
	sum_squares_OG = np.sum(audioOG**2)
	sum_squares_OG_minus_H_array = np.array([])
	sum_squares_OG_minus_P_array = np.array([])

	for y in y_array:
		separate(filename, y);
		audioH, srH = lb.load('H.wav', sr=None)
		audioP, srP = lb.load('P.wav', sr=None)

		DH = lb.amplitude_to_db(np.abs(lb.stft(audioH)), ref=np.max)
		DP = lb.amplitude_to_db(np.abs(lb.stft(audioP)), ref=np.max)

		sum_squares_OG_minus_H = np.sum((audioOG - audioH)**2)
		sum_squares_OG_minus_P = np.sum((audioOG - audioP)**2)
		sum_squares_OG_minus_H_array = np.append(sum_squares_OG_minus_H_array,
												 sum_squares_OG_minus_H)
		sum_squares_OG_minus_P_array = np.append(sum_squares_OG_minus_P_array,
												 sum_squares_OG_minus_P)

		plt.subplot(5, 2, i)
		specshow(DH, y_axis='linear')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Harmonic power spectrogram with gamma = ' + str(y))

		plt.subplot(5, 2, i+1)
		specshow(DP, y_axis='linear')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Percussive power spectrogram with gamma = ' + str(y))

		i += 2

	#plot differnt gamma values
	plt.tight_layout()
	plt.show()

	signal_to_noise_OG_minus_H = 10*np.log10(sum_squares_OG/sum_squares_OG_minus_H_array)
	signal_to_noise_OG_minus_P = 10*np.log10(sum_squares_OG/sum_squares_OG_minus_P_array)
	plt.plot(y_array, signal_to_noise_OG_minus_H, label='Harmonic')
	plt.plot(y_array, signal_to_noise_OG_minus_P, label='Percussive')
	plt.xlabel('Gamma')
	plt.ylabel('SNR')
	plt.legend()
	plt.title('Signal-to-noise ratio of harmonic and percussive components from file '
			+ filename + ' with different gammas')
	plt.show()

def test_diffent_iterations_num(filename, k_array = [5, 10, 20, 60, 100]):

	plt.figure(figsize=(12, 10))

	i = 1

	audioOG, srOG = lb.load(filename, sr=None)
	sum_squares_OG = np.sum(audioOG**2)
	sum_squares_OG_minus_H_array = np.array([])
	sum_squares_OG_minus_P_array = np.array([])

	for k in k_array:
		separate(filename, k_max = k);
		audioH, srH = lb.load('H.wav', sr=None)
		audioP, srP = lb.load('P.wav', sr=None)

		DH = lb.amplitude_to_db(np.abs(lb.stft(audioH)), ref=np.max)
		DP = lb.amplitude_to_db(np.abs(lb.stft(audioP)), ref=np.max)

		sum_squares_OG_minus_H = np.sum((audioOG - audioH)**2)
		sum_squares_OG_minus_P = np.sum((audioOG - audioP)**2)
		sum_squares_OG_minus_H_array = np.append(sum_squares_OG_minus_H_array,
												 sum_squares_OG_minus_H)
		sum_squares_OG_minus_P_array = np.append(sum_squares_OG_minus_P_array,
												 sum_squares_OG_minus_P)

		plt.subplot(5, 2, i)
		specshow(DH, y_axis='linear')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Harmonic power spectrogram with ' + str(k) + ' iterations')

		plt.subplot(5, 2, i+1)
		specshow(DP, y_axis='linear')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Percussive power spectrogram with ' + str(k) + ' iterations')

		i += 2

	#plot the different iteration values
	plt.tight_layout()
	plt.show()

	#Using the formula

	signal_to_noise_OG_minus_H = 10*np.log10(sum_squares_OG/sum_squares_OG_minus_H_array)
	signal_to_noise_OG_minus_P = 10*np.log10(sum_squares_OG/sum_squares_OG_minus_P_array)
	plt.plot(k_array, signal_to_noise_OG_minus_H, label='Harmonic')
	plt.plot(k_array, signal_to_noise_OG_minus_P, label='Percussive')
	plt.xlabel('No. iterations')
	plt.ylabel('SNR')
	plt.legend()
	plt.title('Signal-to-noise ratio of harmonic and percussive components from file '
			+ filename + ' with different number of iterations')
	plt.show()

if len(sys.argv) == 1:
    print("Need to enter the name of audio file after .py ")
    exit(0)
else:
    filename = sys.argv[1]
#Calling the function
test_different_gammas(filename)
test_diffent_iterations_num(filename)