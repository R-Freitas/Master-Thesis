# Plot images
from keras.datasets import mnist
from matplotlib import pyplot
import scipy.misc
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def generate_images():
    count_files = 0
    count_cells = 0
	imag_count=0
    dir = os.getcwd()
    dirs = []
    dirs.append(dir)

    for dir in dirs:
        print("DIR: ", dir)
        cells = []

        i = 0  # Used to transverse the cells list when printing the images
        for roots, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('.mat'):
                    path = os.path.realpath(os.path.join(roots, file))
                    print("PATH: ", path)
                    data = (sio.loadmat(path, struct_as_record=True))['storage']

                    for case in data:
                        count_cells += 1
                        if (set_class(case['CellCycle'][0][0]) < 3):
                            cells.append(Cell_Info(np.matrix(case['Mask'][0]), case['CellCycle'][0][0]))

                    count_files += 1

        print(count_files, "file(s) found")
        print(count_cells, "cell(s) found,", len(cells), "cell(s) used")

    # With all the cells cells in a list, and an input size chosen it is
    # time to create the input for the neural network itself
    treated_cells = []

    for cell in cells:
        S_mask = np.zeros((FRAME_SIZE, FRAME_SIZE))

        y_diff = cell.y_max - cell.y_min
        x_diff = cell.x_max - cell.x_min

        if (y_diff > FRAME_SIZE or x_diff > FRAME_SIZE):
            print("Impossible to fit cell, please increase frame size")
        else:
            y_offset = int((FRAME_SIZE - y_diff) / 2)
            x_offset = int((FRAME_SIZE - x_diff) / 2)

            S_mask[y_offset:y_diff + y_offset + 1, x_offset:x_diff + x_offset +
                   1] = cell.Matrix[cell.y_min: cell.y_max + 1, cell.x_min:cell.x_max + 1]
            treated_cells.append(Cell_Info(S_mask.astype(float), cell.CellCycle))

    del cells
    #data = np.array([(cell.Matrix) for cell in treated_cells])
    #labels = to_categorical(np.array([(int(cell.Class)) for cell in treated_cells]),num_classes=3)
    data_G1 = np.array([(cell.Matrix) for cell in treated_cells if cell.Class == 0])
    data_S = np.array([(cell.Matrix) for cell in treated_cells if cell.Class == 1])
    data_G2 = np.array([(cell.Matrix) for cell in treated_cells if cell.Class == 2])


    for ima in data_G1:
        filename = "image_" + str(imag_count) + ".jpg"
        scipy.misc.imsave(.. / Treated_Data / Test_Images / G1 / filename, ima)



# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# create a grid of 3x3 images
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))
# show the plot
# pyplot.show()
scipy.misc.imsave('outfile.jpg', X_train[0])

img = load_img('outfile.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
print(x.shape)
