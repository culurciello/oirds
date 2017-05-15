#!/usr/bin/env python3

# Eugenio Culurciello, May 2017
# OIRDS data loader and data prepare
# data: https://sourceforge.net/projects/oirds/?source=typ_redirect
# updated code from: https://github.com/cogman2/oirds/blob/master/tools/prep.py
# creates train/ and val/ folders with car/ no_car/ for pytorch/torch training

# =================================================================
# 1. Load the metadata from .xls files.
# 2. Correct for ground sample distance variation.
# 3. Remove vehicles with shadows.
# 4. Rotate the images in 15 degree increments.
# * - We are currently missing Chul's centroid and polygon corrections.
# 6. Select patches of a given size around vehicles in the images.
# 7. Crop 'no vehicle' patches from images with only one vehicle.
# 8. Create the corresponding training and validation set .txt files.
#
#  This approach puts all the images in the same folders, and depends
#  on the .txt files to keep them separate (though filesnames also 
#  distinguish crop sizes and measures of rotation).
# 
# Example usage: sudo ./prep.py <crop size> <rotate>
# 
# crop size .......... dimensions in pixels          (int)
# rotate ............. should the images be rotated? (bool)
# =================================================================


import os
from PIL import Image
import pandas as pd
import numpy as np
import sys
import re

# Create a csv from the xls files.
def load_xl(data_dir):

    dataframe = pd.DataFrame()
    for i in range(20):
        fname = data_dir + '/DataSet_' + str(i + 1) + '/DataSet' + str(i + 1) + '.xls'
        book = pd.read_excel(io=fname, sheetname=0, parse_cols=[1, 2, 3, 7, 8, 9, 15, 28, 47])
        dataframe = dataframe.append(book)

    dataframe = dataframe[['Image Path', 'Image Name', 'Target Number',
                           'Intersection Polygon', 'Average Target Centroid',
                           'Mode of Target Type', 'Average Target Orientation',
                           'Average Target Shadow %', ' Average GSD']]
    return dataframe


# rm -rf a folder
def rmrf(folder):
    if os.path.isdir(folder):
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
            os.rmdir(root)


# Ground sample distance normalization
# - Zooms in to match the minimum value
def gsdnorm(df):
    # Average GSD
    val = np.unique(df.iloc[:, 8].values)
    base = val.min()
    rmrf('data/oirds/zoom')
    os.mkdir('data/oirds/zoom')
    for i, gsd in enumerate(df.iloc[:, 8]):
        multiplier = gsd / base
        original = df.iloc[i, 1][:-3] + 'png'
        infile = 'data/oirds/png/' + original
        outfile = 'data/oirds/zoom/' + original
        im = Image.open(infile)
        x_size = int(im.size[0] * multiplier)
        y_size = int(im.size[1] * multiplier)
        im = im.resize((x_size, y_size), Image.ANTIALIAS)
        im.save(outfile)

        # The polygon
        matrix = df.iloc[i, 3][1:-1].split(';')
        matrix = [x.split(' ') for x in matrix]
        matrix = pd.DataFrame(matrix)
        matrix = matrix.convert_objects(convert_numeric=True)
        matrix = matrix.multiply(multiplier).astype('int')
        matrix = matrix.to_string(header=False, index=False)
        df.iloc[i, 3] = re.sub(r' +', ' ', matrix).replace('\n ', ';')

        # The centroid
        center = pd.to_numeric(df.iloc[1, 4][1:-1].split())        
        df.iloc[1, 4] = str(center * multiplier) # EC added str() here
        return df


def main():
    # global total, total
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    data = 'data/oirds'
    try:
        chip_size = int(sys.argv[1])
        half_chip = chip_size / 2

        rmrf(data+'/train')
        os.mkdir(data + '/train')
        rmrf(data+'/val')
        os.mkdir(data + '/val')

        rmrf(data+'/train/no_car')
        os.mkdir(data + '/train/no_car')
        rmrf(data+'/train/car')
        os.mkdir(data + '/train/car')

        rmrf(data+'/val/no_car')
        os.mkdir(data + '/val/no_car')
        rmrf(data+'/val/car')
        os.mkdir(data + '/val/car')

        crop = 1

    except ValueError:
        crop = 0
        chip_size = ''

    # Extract data from the spreadsheets.
    total = load_xl(data)
    # total = pd.read_csv(data+'/datasets.csv', index_col=0)

    # Correct for ground sample distance variation.
    total = gsdnorm(total)

    # Exclude vehicles with shadow values greater than or equal to 0.1.
    total = total[total['Average Target Shadow %'] < 0.1]

    # Find the names of images with a second target.
    multiples = total[total.iloc[:, 2] == 2].iloc[:, 1]
    # Limit the number of no-vehicle chips to the number of vehicle chips.
    train_test_proportion = 5  # train/test ratio
    limit = len(total.iloc[:, 2]) / train_test_proportion
    counter = 0

    rotate = sys.argv[2] in ['True', 'true', 'y', 'Y', 'Yes', 'yes']

    # Loop over the vehicles.
    for i, ctr in enumerate(total.iloc[:, 4]):  # Centroid coordinates
        original = total.iloc[i, 1][:-3] + 'png'
        print('processing image:', data + '/png/' + original)
        im1 = Image.open(data + '/png/' + original)
        uid = total.iloc[i, 1][:-4] + str(total.iloc[i, 2]) + '_' + str(chip_size)

        # Rotate the image.
        if rotate:
            copyfile(original, data + '/rotate' + original)
            for j in range(15, 180, 15):
                img = im1.rotate(j)
                fname = data + '/rotate/' + uid + '_' + str(j) + '.png'
                img.save(fname)


        if crop != 0:
            # Crop the image around the vehicles. ctr = string with center coordinates!
            txt = ctr.replace(']', '').replace('[', '')

            # EC: THIS BELOW DID NOT WORK!!! SO WAS REPLACED
            # io_txt = StringIO(txt)
            # print(io_txt)
            # ctr2 = pd.DataFrame.from_csv(io_txt,
            #                              sep=" ",
            #                              parse_dates=False,
            #                              index_col=None
            #                              ).apply(int)

            ctr2 = np.fromstring(txt, dtype=float, count=2, sep=' ') # get x,y from string
            w, h = im1.size
            # ctr_x, ctr_y = ctr2.iloc[0], ctr2.iloc[1]
            ctr_x, ctr_y = int(ctr2[0]), int(ctr2[1])

            # The distance from the top-left corner to the 
            #   left edge, top edge, right edge and bottom edge.
            l, u, r, low = ctr_x - half_chip, ctr_y - half_chip, ctr_x + half_chip, ctr_y + half_chip
            if ctr_x < half_chip:
                l, r = 0, chip_size

            if ctr_x > w - half_chip:
                l, r = w - chip_size, w

            if ctr_y < half_chip:
                u, low = 0, chip_size

            if ctr_y > h - half_chip:
                u, low = h - chip_size, h

            im2 = im1.crop((l, u, r, low))
            uid = total.iloc[i, 1][:-4] + str(total.iloc[i, 2]) + '_' + str(chip_size)

            if i % train_test_proportion == 0:
                fname = data + '/val/car/' + uid + '_0.png'
                im2.save(fname)
            else:
                fname = data + '/train/car/' + uid + '_0.png'
                im2.save(fname)

            # Tile the single-object images with "no car" chips.
            if total.iloc[i, 1] not in multiples:
                # Tile to the right.
                # x
                for j in range( int( (w - ctr_x - half_chip) / chip_size) ):
                    # y
                    for k in range( int(h / chip_size) ):
                        right = w - j * chip_size
                        upper = k * chip_size
                        left = right - chip_size
                        lower = upper + chip_size
                        im3 = im1.crop((left, upper, right, lower))
                        code = j * h / chip_size + k
                        uid = total.iloc[i, 1][:-4] + str(code) + '_' + str(chip_size)

                        if counter < limit:
                            fname = data + '/val/no_car/' + uid + 'R.png'
                            im3.save(fname)
                        else:
                            fname = data + '/train/no_car/' + uid + 'R.png'
                            im3.save(fname)
                        counter += 1
                # Tile to the left.
                # x
                for m in range( int( (ctr_x - half_chip) / chip_size) ):
                    # y
                    for n in range( int(h / chip_size) ):
                        left = m * chip_size
                        right = left + chip_size
                        upper = n * chip_size
                        lower = upper + chip_size
                        im3 = im1.crop((left, upper, right, lower))
                        code = m * h / chip_size + n
                        uid = total.iloc[i, 1][:-4] + str(code) + '_' + str(chip_size)
                        if counter < limit:
                            fname = data + '/val/no_car/' + uid + 'L.png'
                            im3.save(fname)
                        else:
                            fname = data + '/train/no_car/' + uid + 'L.png'
                            im3.save(fname)
                        counter += 1

    #end:
    rmrf('data/oirds/zoom')

if __name__ == "__main__":
    main()


