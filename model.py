import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.utils.visualize_util import plot
from keras.models import load_model
import tensorflow as tf
import random

import matplotlib.pyplot as plt
import json
import signal

# Signal handler to catch SIGINT to be able to pause the training
pause = False
def signal_handler(sig, frame):
        print("")
        global pause
        pause = True
signal.signal(signal.SIGINT, signal_handler)

def calc_correction_factor(measurement, left, correction_dist=40.0):
    """ Calculates the measurement for the left and right images.

    The corrected measurement is calculated by using a triangle with the sides 1 and correction_dist
    and the angle measurement between both. With these the corresponding angle for the side image is
    calculated.

    Args:
        measurement:    the original measurement for the center image
        left:           bool if the measurement is for the left or the right image
    Returns:
        the corrected measurement
    """
    b = 1.0
    c = correction_dist
    if left:
        alpha = 90+measurement*25.0
    else:
        alpha = 90-measurement*25.0
    a = np.sqrt(b**2.0+c**2.0-2.0*b*c*np.cos(np.deg2rad(alpha)))
    gamma = np.rad2deg(np.arcsin((c*np.sin(np.deg2rad(alpha)))/a))
    if left:
        return (90-gamma)/25.0
    else:
        return (gamma-90)/25.0

def loadData(paths, reduce_zero_measurement=1, side_image_prob=1, correction_dist=40.0, gauss_noise=0.01,
                measurement_threshhold=0):
    """ Loads the cvs files from all given paths.

    Args:
        paths:                      list of paths which contain the training data
        reduce_zero_measurement:    probability for including measurements which are zeroself.
                                    Used to reduce the high amount of zero measurements
        side_image_prob:            probability for including each side image
        correction_dist:            factor to calculate the corecction for the side images
        gauss_noise:                std of gaussian noise which is added to the measurements
        measurement_threshhold:     threshhold to include only measurements which are greater then this threshhold
    Returns:
        images:                     the paths of the images
        measurements:               the corresponding measurements for the images
    """
    image_paths = list(map(lambda path: path+"IMG/", paths))
    cvs_paths = list(map(lambda path: path+"driving_log.csv", paths))

    images = []
    measurements = []
    normal_image_count = 0
    zero_measurement_count = 0
    side_image_count = 0

    for image_path, cvs_path in zip(image_paths, cvs_paths):
        with open(cvs_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                source_path = line[0]
                source_path_left = line[1]
                source_path_right = line[2]
                filename = source_path.split('/')[-1]
                filename_left = source_path_left.split('/')[-1]
                filename_right = source_path_right.split('/')[-1]

                try:
                    measurement = float(line[3])
                except ValueError:
                    continue
                if np.abs(measurement) < measurement_threshhold:
                    continue
                if measurement==0 and random.random()<reduce_zero_measurement:
                    images.append(image_path + filename)
                    measurements.append(measurement+random.gauss(0.0, gauss_noise))
                    zero_measurement_count += 1
                    if random.random()<side_image_prob:
                        images.append(image_path + filename_left)
                        measurements.append(calc_correction_factor(measurement, True, correction_dist=correction_dist)+random.gauss(0.0, gauss_noise))
                        side_image_count += 1
                    if random.random()<side_image_prob:
                        images.append(image_path + filename_right)
                        measurements.append(calc_correction_factor(measurement, False, correction_dist=correction_dist)+random.gauss(0.0, gauss_noise))
                        side_image_count += 1
                if measurement!=0:
                    images.append(image_path + filename)
                    measurements.append(measurement+random.gauss(0.0, gauss_noise))
                    normal_image_count += 1

                    if random.random()<side_image_prob:
                        images.append(image_path + filename_left)
                        measurements.append(calc_correction_factor(measurement, True, correction_dist=correction_dist)+random.gauss(0.0, gauss_noise))
                        side_image_count += 1
                    if random.random()<side_image_prob:
                        images.append(image_path + filename_right)
                        measurements.append(calc_correction_factor(measurement, False, correction_dist=correction_dist)+random.gauss(0.0, gauss_noise))
                        side_image_count += 1

    #print("#Normal measurement samples: "+str(normal_image_count*augmentation_factor))
    #print("#Zero measurement samples: "+str(zero_measurement_count*augmentation_factor))
    #print("#Side image samples: "+str(side_image_count*augmentation_factor))
    return images, measurements

def getData(data, batch_size=16):
    """ Generator which loads the images from the disk and yields them

    Args:
        data:       list of tupels of images and measurements
        batch_size: the size of the batches which are generated
    Returns:
        yields a list of training data which labels
    """
    batch_size = batch_size//augmentation_factor
    nb_data = len(data)

    while 1:
        shuffle(data)
        for offset in range(0, nb_data, batch_size):
            batch_data = data[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_data_line in batch_data:
                measurement = batch_data_line[1]
                current_path = batch_data_line[0]

                image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2HSV)
                image = cv2.resize(image[60:135, :, :], (200,66))

                images.append(image)
                measurements.append(measurement)
                images.append(cv2.flip(image, 1))
                measurements.append(measurement*-1.0)

            X_train = np.array(images, dtype=np.uint8)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

def cap_data(images, measurements, max_bin_size=50, bin_step=0.0005):
    """ Sorts the give images and measurements in bins and the limits the size of this bins

    Args:
        images:       list of images
        measurements: meas corresponding to the images
        max_bin_size: maximal amount of data which can be in each bin
        bin_step:     the step size of the bins
    Returns:
        capped_images:       the images which respect to the maximal bin size
        capped_measurements: the corresponding measurements
    """
    data = shuffle(list(zip(images, measurements)))
    images, measurements = zip(*data)
    images = list(images)
    measurements = list(measurements)
    measurements = np.array(measurements)
    images = np.array(images)
    index_range = [x+int(1.0/bin_step) for x in range(-int(1.0/bin_step), int(1.0/bin_step)+1,1)]
    bins = [x/(1.0/bin_step) for x in range(-int(1.0/bin_step), int(1.0/bin_step),1)]

    bins_index = np.digitize(measurements, bins)
    data = []
    for bin in index_range:
        i = images[bins_index==bin]
        i = i[:min(len(i),max_bin_size)]
        m = measurements[bins_index==bin]
        m = m[:min(len(m),max_bin_size)]
        data.append((i,m))
    capped_measurements = []
    capped_images = []
    for bin in data:
        capped_images.extend(bin[0])
        capped_measurements.extend(bin[1])
    return capped_images, capped_measurements

def getModel():
    """ Generates the model. The model is nearly the same as in the given nvidia paper

    Returns:
        modle: the generated keras model
    """
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
    from keras.layers.convolutional import Convolution2D, Cropping2D
    from keras.layers.pooling import AveragePooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.noise import GaussianNoise

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5 , input_shape=(66,200,3)))
    model.add(Convolution2D(3,1,1))
    model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model

def show_images_with_angle(train_data):
    for imagepath, m in train_data:
        image = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2RGB)
        shape = image.shape
        cv2.line(image,(shape[1]//2,shape[0]*19//20),(shape[1]//2+int((shape[0]*19//20)*m),0),(255,0,0),5)
        plt.imshow(image)
        plt.show()

def plot_histo(measurements, bin_step=0.0005):
    plt.rcdefaults()
    measurements_list = list(measurements)
    measurements_list.extend([measurement * -1.0 for measurement in measurements])
    fig, ax1 = plt.subplots(1,1)
    bins = [x/(1.0/bin_step) for x in range(-int(1.0/bin_step), int(1.0/bin_step),1)]
    ax1.hist(measurements_list, bins=bins)
    ax1.set_title('Number of training pictures per label')
    ax1.set_ylabel("Amount")
    ax1.set_xlabel("Label as Number")
    plt.show()

def pause_callback():
    """ Callback which only waits for an input """
    global pause
    if pause:
        wait = input("Press return to continue.")
        pause = False

def load_saved_model():
    """ Used to load a model """
    filepath = ""
    initial_epoch = 0
    model = load_model(filepath)
    return model, initial_epoch


# Fixed parameters which not affect the training ot data
SHOW_PLOTS = False
DEBUG = False
LOAD_MODEL = False
augmentation_factor = 2

# Paths of the training data
paths = []
paths_track1 = ["./data/",
         "./recorded_data/track1-fastest/",
         "./recorded_data/track1-forward-fastest/",
         "./recorded_data/track1-forward-fast/",
         "./recorded_data/track1-forward-simple/",
         "./recorded_data/track1-forward-beautiful/",
         "./recorded_data/track1-backward-fastest/",
         "./recorded_data/track1-backward-fast/",
         "./recorded_data/track1-backward-simple/",
         "./recorded_data/track1-backward-beautiful/"
         ];
paths.append(paths_track1)
paths_track2 = [
         "./recorded_data/track2-forward-fastest/",
         "./recorded_data/track2-forward-simple/"
         ];
paths.append(paths_track2)
""" Track 1
correction_dist=40.0
side_image_prob=0.4
gauss_noise=0.005
epochs = 10
batch_size = 32


reduce_zero_measurement=0.1
max_bin_size=30
bin_step=0.0005
"""

# Parameters which will affect the training data
correction_dist=30.0
side_image_prob=0.3
gauss_noise=0.005
reduce_zero_measurement = 0.2
measurement_threshhold = 0.0
max_bin_size=5
bin_step=0.0005

# Paramets for training
epochs = 100
batch_size = 32


# Load training data
images_track1, measurements_track1 = loadData(paths_track1, reduce_zero_measurement=reduce_zero_measurement,
                                              side_image_prob=side_image_prob,
                                              correction_dist=correction_dist,
                                              gauss_noise=gauss_noise,
                                              measurement_threshhold=measurement_threshhold)
images_track2, measurements_track2 = loadData(paths_track2, reduce_zero_measurement=reduce_zero_measurement,
                                            side_image_prob=side_image_prob,
                                            correction_dist=correction_dist,
                                            gauss_noise=gauss_noise,
                                            measurement_threshhold=measurement_threshhold)
if SHOW_PLOTS:
    plot_histo(measurements_track1 + measurements_track2, bin_step=bin_step)
print("#Samples track 1: "+str(len(images_track1)))
print("#Samples track 2: "+str(len(images_track2)))

# Sort data in bin and limit the bin size
images_track1, measurements_track1 = cap_data(images_track1, measurements_track1, max_bin_size=max_bin_size, bin_step=bin_step)
images_track2, measurements_track2 = cap_data(images_track2, measurements_track2, max_bin_size=max_bin_size, bin_step=bin_step)
if SHOW_PLOTS:
    plot_histo(measurements_track1 + measurements_track2, bin_step=bin_step)
print("#Samples track 1 after capping: "+str(len(images_track1)))
print("#Samples track 2 after capping: "+str(len(images_track2)))

images = []
images.extend(images_track1)
images.extend(images_track2)
measurements = []
measurements.extend(measurements_track1)
measurements.extend(measurements_track2)

# Generate train and validation set
data = list(zip(images, measurements))
train_data, validation_data = train_test_split(data, test_size=0.2, shuffle=True)
print("#Train samples: "+str(len(train_data)*augmentation_factor))
print("#Validation samples: "+str(len(validation_data)*augmentation_factor))

# Get generators for data
train_generator = getData(train_data, batch_size=batch_size)
validation_generator = getData(validation_data, batch_size=batch_size)

# Define a name for the model
model_name = ("model_track2_correction_"+str(correction_dist) +
                  ".side_prob_"+str(side_image_prob) +
                  ".reduce_zero_measurement_"+str(reduce_zero_measurement) +
                  ".gauss_noise_"+str(gauss_noise) +
                  ".max_bin_size"+str(max_bin_size)
              )
# Get Callbacks for saving and pause
model_checkpoint = ModelCheckpoint('./models/'+model_name+'.{epoch:02d}-{val_loss:.4f}.h5')
pause_checkpoint = LambdaCallback(on_epoch_begin=lambda epoch, logs: pause_callback(),
                                  on_epoch_end=lambda epoch, logs: pause_callback(),
                                  on_batch_begin=lambda epoch, logs: pause_callback(),
                                  on_batch_end=lambda epoch, logs: pause_callback()
                                  )

# Load model or complie new one
if not LOAD_MODEL:
    model = getModel()
    initial_epoch = 0
    model.compile(loss='mse', optimizer='adam')
else:
    model, initial_epoch = load_saved_model()
model.summary()
plot(model, to_file='models/' + model_name + '.png', show_shapes=True, show_layer_names=False)

# Dict to save the parameters
dict = {
        'correction_dist':correction_dist,
        'gauss_noise':gauss_noise,
        'reduce_zero_measurement':reduce_zero_measurement,
        'side_image_prob':side_image_prob,
        'bin_step':bin_step,
        'max_bin_size':max_bin_size,
        'batch_size':batch_size,
        'data':paths,
        'model':json.loads(model.to_json())
        }
with open('./models/' + model_name + '.json','w') as file:
    json.dump(dict, file, indent=4)

# Train model
model.fit_generator(train_generator,
    samples_per_epoch=len(train_data)*augmentation_factor,
    validation_data=validation_generator,
    nb_val_samples=len(validation_data)*augmentation_factor,
    nb_epoch=epochs,
    initial_epoch=initial_epoch,
    callbacks=[model_checkpoint,pause_checkpoint])
