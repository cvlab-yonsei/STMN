#!/usr/bin/env python2
#-*-encoding:utf-8-*-
import os,sys
import re
os.system('mkdir -p %s'%('./LSVID_database'))

sum_img = 0
data_dir = '/path/to/LS-VID/'

## train split
train_data_dir = data_dir + 'tracklet_train/'
file = open('./LSVID_database/train_path.txt','w')
fielnum = 0
list = os.listdir(train_data_dir)
list.sort()
for line in list:
	sub_dir = train_data_dir+'/'+line
	sub = os.listdir(sub_dir)
	sub.sort()
	for subline in sub:
		label = int(line[0:4])
		full_name = train_data_dir +''+line +'/' + subline +'\n'
		file.write(full_name)
		fielnum = fielnum + 1   	
print('all train img num is '+ str(fielnum))
file.close()
sum_img = sum_img + fielnum

## test split
test_data_dir = data_dir + 'tracklet_test/'
file = open('./LSVID_database/test_path.txt','w')
fielnum = 0
list = os.listdir(test_data_dir)
list.sort()
for line in list:
	sub_dir = test_data_dir+'/'+line
	sub = os.listdir(sub_dir)
	sub.sort()
	for subline in sub:
		label = int(line[0:4])
		full_name = test_data_dir +''+line +'/' + subline +'\n'
		file.write(full_name)
		fielnum = fielnum + 1
print('all test img num is '+ str(fielnum))
file.close()
sum_img = sum_img + fielnum

## validation split
val_data_dir = data_dir + 'tracklet_val/'
file = open('./LSVID_database/val_path.txt','w')
fielnum = 0
list = os.listdir(val_data_dir)
list.sort()
for line in list:
	sub_dir = val_data_dir+'/'+line
	sub = os.listdir(sub_dir)
	sub.sort()
	for subline in sub:
		label = int(line[0:4])
		full_name = val_data_dir +''+line +'/' + subline +'\n'
		file.write(full_name)
		fielnum = fielnum + 1
sum_img = sum_img + fielnum
print('all val img num is '+ str(fielnum))
file.close()

print('all img num is '+ str(sum_img))