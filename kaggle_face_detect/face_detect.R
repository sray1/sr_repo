# create variables to store the path to the files
setwd("C:/python_repo/sr_repo/kaggle_face_detect")
data.dir   <- '~/'
train.file <- paste0('Data/', 'training.csv')
test.file  <- paste0('Data/', 'test.csv')

# read in csv files (takes 1 min)
d.train <- read.csv(train.file, stringsAsFactors=F)
# Compactly display structure
str(d.train)

# help on read.csv function
?read.csv

# first rows displayed
head(d.train)

# remove rightmost column from dataframe and save in seperate variable
im.train      <- d.train$Image
d.train$Image <- NULL
head(d.train)

# Each image (each row) contains long string of numbers, 
# where each number represents the intensity of a pixel in the image. 
# Lets look at the first value in the column
im.train[1]

# convert these strings to integers by splitting them and converting the result to integer
# strsplit splits the string, unlist simplifies its output
# to vector of strings and as.integer converts it to a vector of integers.
t_im <- as.integer(unlist(strsplit(im.train[1], " ")))

# windows alternative to 'doMC' library to do parallel computing
install.packages('doSNOW ')
install.packages('foreach')
library('doSNOW')
library('foreach')
cl <- makeCluster(2)
registerDoSNOW(cl)

# implement the parallelization (7 min)
# foreach loop will evaluate the inner command for each row in im.train, 
# and combine the results with rbind (combine by rows). 
# %dopar% instructs R to do all evaluations in parallel.
im.train <- foreach(im = im.train, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}

# im.train is now a matrix with 7049 rows (one for each image) and 9216 columns 
# (one for each pixel):
str(im.train)

# Repeat the process for test.csv (1 min), as we are going to need it later. 
# Notice in the test file, we don't have the first 30 columns with the keypoint locations.
d.test  <- read.csv(test.file, stringsAsFactors=F)
im.test <- foreach(im = d.test$Image, .combine=rbind) %dopar% {
  as.integer(unlist(strsplit(im, " ")))
}
d.test$Image <- NULL

# save all four variables into the data.Rd file: (3 min)
save(d.train, im.train, d.test, im.test, file='data.Rd')

# can reload them at any time with the following command
load('data.Rd')

##############################################################################

# To visualize each image, we need to first convert these 9216 integers into 96x96 matrix
# im.train[1,] returns first row of im.train, which corresponds to the first training 
# image. rev reverse the resulting vector to match interpretation of R's image function
# (which expects origin to be in lower left corner).
im <- matrix(data=rev(im.train[1,]), nrow=96, ncol=96)

# display image
image(1:96, 1:96, im, col=gray((0:255)/255))

# color the coordinates for the eyes and nose:
points(96-d.train$nose_tip_x[1],         96-d.train$nose_tip_y[1],         col="red")
points(96-d.train$left_eye_center_x[1],  96-d.train$left_eye_center_y[1],  col="blue")
points(96-d.train$right_eye_center_x[1], 96-d.train$right_eye_center_y[1], col="green")

# where are the centers of each nose in the 7049 images? 
for(i in 1:nrow(d.train)) {
  points(96-d.train$nose_tip_x[i], 96-d.train$nose_tip_y[i], col="red")
}

# Looking at one extreme example we get this:
idx <- which.max(d.train$nose_tip_x)
im  <- matrix(data=rev(im.train[idx,]), nrow=96, ncol=96)
image(1:96, 1:96, im, col=gray((0:255)/255))
points(96-d.train$nose_tip_x[idx], 96-d.train$nose_tip_y[idx], col="red")

# One of the simplest things to try is to compute the mean of the coordinates of each 
# keypoint in the training set and use that as a prediction for all images. 
# This is a very simplistic algorithm, as it completely ignores the images, but we can use it
# a starting point to build a first submission
# Computing mean for each column is straightforward with colMeans na.rm=T ignores missing values.
colMeans(d.train, na.rm=T)

# apply these computed coordinates to the test instances:
p <- matrix(data=colMeans(d.train, na.rm=T), nrow=nrow(d.test), ncol=ncol(d.train), byrow=T)
colnames(p) <- names(d.train)
predictions <- data.frame(ImageId = 1:nrow(d.test), p)
head(predictions)

# expected submission format has one one keypoint per row, but we can easily get that 
# with the helpinstall.packages('reshape2')
library(reshape2)
submission <- melt(predictions, id.vars="ImageId", variable.name="FeatureName", value.name="Location")
head(submission)

# join this with the sample submission file to preserve the same order of entries and
# save the result
s <- paste0('Data/', 'SampleSubmission.csv')
example.submission <- read.csv(s)
sub.col.names      <- names(example.submission)
example.submission$Location <- NULL
submission <- merge(example.submission, submission, all.x=T, sort=F)
submission <- submission[, sub.col.names]
write.csv(submission, file="submission_means.csv", quote=F, row.names=F)

# extract patch around this keypoint in each image, and average the result. 
# This average_patch can then be used as a mask to search for the keypoint in test images

# parameters
coord      <- "left_eye_center"
patch_size <- 10

# coord is the keypoint we are working on, and patch_size is the number of pixels we are going
# to extract in each direction around the center of the keypoint. So 10 means we will have 
# a square of 21x21 pixels (10+1+10). This will become more clear with an example:

coord_x <- paste(coord, "x", sep="_")
coord_y <- paste(coord, "y", sep="_")
patches <- foreach (i = 1:nrow(d.train), .combine=rbind) %do% {
  im  <- matrix(data = im.train[i,], nrow=96, ncol=96)
  x   <- d.train[i, coord_x]
  y   <- d.train[i, coord_y]
  x1  <- (x-patch_size)
  x2  <- (x+patch_size)
  y1  <- (y-patch_size)
  y2  <- (y+patch_size)
  if ( (!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96) )
  {
    as.vector(im[x1:x2, y1:y2])
  }
  else
  {
    NULL
  }
}
mean.patch <- matrix(data = colMeans(patches), nrow=2*patch_size+1, ncol=2*patch_size+1)

# foreach loop will get each image and:
# extract the coordinates of the keypoint: x and y
# compute the coordinates of the patch: x1, y1, x2 and y2
# check if the coordinates are available (is.na) and are inside the image
# if yes, return the image patch as a vector; if no, return NULL
# All the non-NULL vectors will then be combined with rbind, which concatenates them as rows.
# The result patches will be a matrix where each row is a patch of an image. 
# We then compute mean of all images with colMeans, put back in matrix format and store in mean.patch

# visualize the result with image
image(1:21, 1:21, mean.patch[21:1,21:1], col=gray((0:255)/255))

# define parameter
search_size <- 2

# search_size indicates how many pixels we are going to move in each direction when searching 
# for the keypoint. We will center the search on the average keypoint location, and go
# search_size pixels in each direction
mean_x <- mean(d.train[, coord_x], na.rm=T)
mean_y <- mean(d.train[, coord_y], na.rm=T)
x1     <- as.integer(mean_x)-search_size
x2     <- as.integer(mean_x)+search_size
y1     <- as.integer(mean_y)-search_size
y2     <- as.integer(mean_y)+search_size

# In this particular case the search will be from (64,35) to (68,39). 
# We can use expand.grid to build a data frame with all combinations of x's and y's:
params <- expand.grid(x = x1:x2, y = y1:y2)
params

# Given a test image we need to try all these combinations, and see which one best matches 
# the average_patch. We will do that by taking patches of the test images around these points 
# and measuring their correlation with the average_patch. Take the first test image as an example:
im <- matrix(data = im.test[1,], nrow=96, ncol=96)

r  <- foreach(j = 1:nrow(params), .combine=rbind) %dopar% {
  x     <- params$x[j]
  y     <- params$y[j]
  p     <- im[(x-patch_size):(x+patch_size), (y-patch_size):(y+patch_size)]
  score <- cor(as.vector(p), as.vector(mean.patch))
  score <- ifelse(is.na(score), 0, score)
  data.frame(x, y, score)
}

# Inside the for loop, given a coordinate we extract an image patch p and compare it 
# to the average_patch with cor. The ifelse is necessary for the cases where all the image
# patch pixels have the same intensity, as in this case cor returns NA. The result will 
# look like this:
r

#Now all we need to do is return the coordinate with the highest score:
best <- r[which.max(r$score), c("x", "y")]
best


