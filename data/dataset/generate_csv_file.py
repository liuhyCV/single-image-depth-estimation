import os
from sklearn.model_selection import train_test_split
import re



def create_nyu_raw_train_test_csv(dataset_dir):
    # read all image files under dataset_dir

    trains_test = []

    '''
    for root, dirs, files in os.walk(dataset_dir):
        #print files
        for d in files:
            trains_test.append( os.path.join(root, d) )
            #print d

    '''

    folder_rgbs = []
    folder_depths = []
    folder_depthmeters = []

    for lists in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, lists)

        if os.path.isdir(path) & (re.search('processed', path) != None):
            print path

            for sub_lists in os.listdir(path):
                sub_path = os.path.join(path, sub_lists)
                print sub_path
                for sub_sub_lists in os.listdir(sub_path):
                    image_path = os.path.join(sub_path, sub_sub_lists)
                    print image_path
                    if (re.search('Rgb', image_path) != None):
                        folder_rgbs.append(image_path)
                    elif (re.search('DepthGray', image_path) != None):
                        folder_depths.append(image_path)
                    elif (re.search('DepthMeters', image_path) != None):
                        folder_depthmeters.append(image_path)
    folder_rgbs.sort()
    folder_depths.sort()
    folder_depthmeters.sort()

    for img_number in range(0, len(folder_rgbs) - 1):
        print img_number
        trains_test.append((folder_rgbs[img_number], folder_depths[img_number], folder_depthmeters[img_number]))

    with open('train_test.csv', 'w') as output:
        for (path_rgbs, path_depths, path_depthmeters) in trains_test:
            output.write("%s,%s,%s" % (path_rgbs, path_depths, path_depthmeters))
            output.write("\n")

    X_train, X_test = train_test_split(trains_test, test_size=0.1, random_state=1)

    with open('train.csv', 'w') as output:
        for (path_rgbs, path_depths, path_depthmeters) in X_train:
            output.write("%s,%s,%s" % (path_rgbs, path_depths, path_depthmeters))
            output.write("\n")
    with open('test.csv', 'w') as output:
        for (path_rgbs, path_depths, path_depthmeters) in X_test:
            output.write("%s,%s,%s" % (path_rgbs, path_depths, path_depthmeters))
            output.write("\n")


def create_vkitti_train_test_csv(vkitti_dataset_dir):
    trains_test = []

    folder_rgbs = []
    folder_depths = []
    folder_scenes = []

    for lists in os.listdir(vkitti_dataset_dir):
        path = os.path.join(vkitti_dataset_dir, lists)

        print path

        if (re.search('rgb', path) != None):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".png"):
                        folder_rgbs.append(os.path.join(root, file))
        elif (re.search('depth', path) != None):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".png"):
                        folder_depths.append(os.path.join(root, file))
        elif (re.search('scene', path) != None):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".png"):
                        folder_scenes.append(os.path.join(root, file))

    folder_rgbs.sort()
    folder_depths.sort()
    folder_scenes.sort()

    print len(folder_rgbs)
    print len(folder_depths)
    print len(folder_scenes)

    for img_number in range(0, len(folder_rgbs) - 1):
        trains_test.append((folder_rgbs[img_number], folder_depths[img_number], folder_scenes[img_number]))

    with open('vkitti_train_test.csv', 'w') as output:
        for (path_rgbs, path_depths, path_scenes) in trains_test:
            output.write("%s,%s,%s" % (path_rgbs, path_depths, path_scenes))
            output.write("\n")

    X_train, X_test = train_test_split(trains_test, test_size=0.1, random_state=1)

    with open('vkitti_train.csv', 'w') as output:
        for (path_rgbs, path_depths, path_scenes) in X_train:
            output.write("%s,%s,%s" % (path_rgbs, path_depths, path_scenes))
            output.write("\n")
    with open('vkitti_test.csv', 'w') as output:
        for (path_rgbs, path_depths, path_scenes) in X_test:
            output.write("%s,%s,%s" % (path_rgbs, path_depths, path_scenes))
            output.write("\n")

def create_nyu_dataset_train_set_csv(dataset_dir, mode = 'depthmeters'):
    trains_test = []
    trains_test_depthmeters = []

    folder_rgbs = []
    folder_depths = []
    folder_depths_meters = []

    for lists in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, lists)

        if (re.search('img.png', path) != None):
            folder_rgbs.append(path)
        elif (re.search('dep.png', path) != None):
            folder_depths.append(path)
        elif (re.search('dep_meters.png', path) != None):
            folder_depths_meters.append(path)

    folder_rgbs.sort()
    folder_depths.sort()
    folder_depths_meters.sort()

    for img_number in range(0, len(folder_rgbs) - 1):
        print img_number
        trains_test.append((folder_rgbs[img_number], folder_depths[img_number]))

    for img_number in range(0, len(folder_rgbs) - 1):
        print img_number
        trains_test_depthmeters.append((folder_rgbs[img_number], folder_depths_meters[img_number]))

    if(mode == 'depthmeters'):
        X_train_val, X_test = train_test_split(trains_test_depthmeters, test_size=0.2, random_state=1)
        X_train, X_val = train_test_split(X_train_val, test_size=0.25, random_state=1)
    elif(mode == 'depth'):
        X_train_val, X_test = train_test_split(trains_test, test_size=0.2, random_state=1)
        X_train, X_val = train_test_split(X_train_val, test_size=0.25, random_state=1)


    with open('train.csv', 'w') as output:
        for (path_rgbs, path_depths) in X_train:
            output.write("%s,%s" % (path_rgbs, path_depths))
            output.write("\n")

    with open('test.csv', 'w') as output:
        for (path_rgbs, path_depths) in X_test:
            output.write("%s,%s" % (path_rgbs, path_depths))
            output.write("\n")

    with open('eval.csv', 'w') as output:
        for (path_rgbs, path_depths) in X_val:
            output.write("%s,%s" % (path_rgbs, path_depths))
            output.write("\n")

dataset_dir = '/home/linze/liuhy/code/cnn_depth/data/dataset/NYU_dataset'

create_nyu_dataset_train_set_csv(dataset_dir, mode = 'depthmeters')
