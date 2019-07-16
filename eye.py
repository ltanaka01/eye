import os, glob, fnmatch, re
import numpy as np
import pandas as pd
from pathlib import Path

# subdivide a single .asc into a sequence of trials, there should be 85 per file
def get_trials(filename, ppd, xpixels, ypixels, sampling_rate):

	lines = open(filename).readlines()

	trials = []
	trial = []
	idat = []
	idats = []

	trial_started = False
	trial_ended = True

	for line in lines:

		if 'TRIALID' in line and trial_started == False and trial_ended == True:
			trial_started = True
			trial_ended = False

		elif 'TRIAL_RESULT' in line and trial_started == True and trial_ended == False:
			trial_started = False
			trial_ended = True
			trials.append(trial)
			
			# fix units of eye data
			# convert time to seconds
			# convert pixels to degrees
			if np.any(idat):
				idat = np.array(idat) # make array
				idat[:,0] = idat[:,0] - idat[0,0] # make first timepoint 0
				idat[:,0] = idat[:,0] / sampling_rate # convert to secondsd
				idat[:,1] = (idat[:,1] - xpixels/2) / ppd
				idat[:,2] = (idat[:,2] - ypixels/2) / ppd
			idats.append(np.array(idat))
			idat = []
			trial = []

		elif 'TRIALID' not in line and trial_started == True and trial_ended == False:
			if not line.split("\t")[0].isnumeric():
				trial.append(line)
			else:
				try:
					eye_t = np.int(line.split('\t')[0])
					eye_x = np.float(line.split('\t')[1])
					eye_y = np.float(line.split('\t')[2])
					idat.append([eye_t, eye_x, eye_y])
				except:
					pass

	return trials, idats

# hit/miss/abort and there are trials marked 'a'
def get_outcome(trial):

	# for each event in the trial
	for event in trial:

		# find the response and grab the type
		if 'TRIAL_VAR Response' in event:
			return event.split('TRIAL_VAR Response')[1].strip()

# time between go-signal and intial saccade onset
# go-signal for MGS (fixation offset) VGS/GAP (target presentation)
def get_latency(trial, task):

	for event in trial:

		# get target onset
		if task == 'MGS':
			if re.search(' FixationOff', event) or re.search(' FixationOff_T', event):
				target_offset = np.int(event.split('MSG')[1].split(' ')[0].strip())
				break
		elif task == 'VGS' or task == 'GAP':
			if re.search(' Target$', event):
				target_offset = np.int(event.split('MSG')[1].split(' ')[0].strip())
				break

	# get onset of the initial saccade
	saccade_onset = np.int(get_initial_saccade(trial).split("\t")[1])

	return saccade_onset-target_offset


# get define the intial saccade as the one with the largest amplitude.
# we could fit a double logistic function to the eye timeseries data and use the model parameter estimates instead.
def get_initial_saccade(trial):

	amps = []
	events = []
	for event in trial:
		if 'ESACC' in event:
			amps.append(np.float(event.split('\t')[-2].strip()))
			events.append(event)

	return events[np.argmax(amps)]

# euclidiean distance b/w the target and the endpoint of the initial saccade
def get_accuracy(trial, ppd, xpixels, ypixels):

	for event in trial:

		# get target location
		if re.search(' targetlocation ', event):
			target_x = (np.int(event.split(' targetlocation [')[1].split(',')[0]) - xpixels) / ppd
			target_y = (np.int(event.split(' targetlocation [')[1].split(',')[1].split(']')[0]) - ypixels) / ppd
			break

	# get location of the endpoint of the initial saccade
	initial_x = (np.float(get_initial_saccade(trial).split('\t')[5].strip()) - xpixels) / ppd
	initial_y = (np.float(get_initial_saccade(trial).split('\t')[6].strip()) - ypixels) / ppd

	# euclidean distance
	error = np.sqrt((target_x-initial_x)**2 + (target_y-initial_y)**2)

	return error

def get_velocity(trial):
	
	# get duration of the initial saccade in seconds
	duration = np.int(get_initial_saccade(trial).split('\t')[2])/1000

	# get the amplitude of the intial saccade (in degrees already?)
	amplitude = np.float(get_initial_saccade(trial).split('\t')[-2])

	return amplitude/duration


def find_files(directory, pattern):
	names = []
	for root, _, files in os.walk(directory):
			for basename in files:
					if fnmatch.fnmatch(basename, pattern):
							filename = os.path.join(root, basename)
							names.append(Path(filename))
	return names

def run_analysis(task, ppd, xpixels, ypixels, sampling_rate, base_path):
	
	path = Path(base_path) / task
	
	# find all the files to run the analysis on
	files = find_files(path, '*.asc')

	# intialize output
	columns = ['task', 'group', 'subject', 'outcome', 'latency', 'velocity', 'accuracy']
	dfs = []
	bad_trials = []

	for file in files:
		print(file)

		# get the task, group, and subject
		group = file.as_posix().split('/')[1]
		subject = file.as_posix().split('/')[2]

		# get the trials
		trials, idats = get_trials(file, ppd, xpixels, ypixels, sampling_rate)

		# loop over trials and extract the data
		for trial in trials:

			try:
				outcome = get_outcome(trial)

				if outcome == 'Hit' or outcome == 'Miss' or outcome == 'Abort' or outcome == 'a':
					latency = get_latency(trial, task)
					velocity = get_velocity(trial)
					accuracy = get_accuracy(trial, ppd, xpixels, ypixels)
					data = [[task, group, subject, outcome, latency, velocity, accuracy]]

					# create output
					df = pd.DataFrame(data = data, columns=columns)

					# store it
					dfs.append(df)
			except Exception as e:
				bad_trials.append([task,group,subject,trial[0], str(e)])

	return pd.concat(dfs), bad_trials


# API
if __name__ == '__main__':

	# degrees of visual angle 
	# and other settings
	viewing_distance = 70 # cm
	xpixels = 1920 # pixels
	ypixels = 1080 # pixels
	pixels_per_cm = 37.79 # pixels/cm
	screen_width = xpixels/pixels_per_cm # cm
	ppd = np.pi*xpixels/np.arctan(screen_width/viewing_distance/2.0)/360.0 # pixels per degree
	sampling_rate = 500 # Hz
	base_path = Path('./')

	# analyze the VGS
	trials, bad_trials = run_analysis('VGS', ppd, xpixels, ypixels, sampling_rate, base_path)
	trials.to_csv('VGS_analyzed.csv', index=False)
	f = open('VGS_bad_trials.txt', 'w')
	for item in bad_trials:
		f.write(str(item) + "\n")
	f.close()


	# analyze the MGS
	trials, bad_trials = run_analysis('MGS', ppd, xpixels, ypixels, sampling_rate, base_path)
	trials.to_csv('MGS_analyzed.csv', index=False)
	f = open('MGS_bad_trials.txt', 'w')
	for item in bad_trials:
		f.write(str(item) + "\n")
	f.close()


	# analyze the GAP
	trials, bad_trials = run_analysis('GAP', ppd, xpixels, ypixels, sampling_rate, base_path)
	trials.to_csv('GAP_analyzed.csv', index=False)
	f = open('GAP_bad_trials.txt', 'w')
	for item in bad_trials:
		f.write(str(item) + "\n")
	f.close()




