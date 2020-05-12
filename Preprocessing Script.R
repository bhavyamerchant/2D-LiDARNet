#Converting cross sections into subsections

#Create a vector with all of the input file names
inputCrossSections <- list.files ('./Cross Sections', full.names = TRUE)

#Read all of the cross section csv files and put them into a list
CrossSectionList <- list()

for (i in 1:length (inputCrossSections))
{
  CrossSectionList[[i]] <- read.delim (inputCrossSections[i], header = F, sep = ",")
}

#Create PNG images from the csv files
for (i in 1:length (inputCrossSections))
{
  attach(CrossSectionList[[i]])
  png (paste ("./PNG Cross Sections/slice_", i, ".png", sep = ""))
  
  par (mar = c(0,0,0,0))
  plot (V1,V3, type = "p", asp = 1, pch = ".", cex = 1, axes = 0, xaxs = "i", yaxs = "i")
  
  dev.off ()
  detach(CrossSectionList[[i]])
}

# Load libraries
library (EBImage)

#Reading images
inputImages <- list.files ('./PNG Cross Sections', full.names = TRUE)

#Convert grayscale images to matricies
imageMatricies <- imageData (channel (readImage (inputImages), "gray"))
 
#Function to split a matrix into r by c matricies
matrixSplitter <- function (M, r, c)
{
  simplify2array
  (
    lapply
    (
      split
      (
        M, interaction ((row (M) - 1) %/% r + 1,(col (M) - 1) %/% c + 1)
      ),
      
      function(x)
      {
        dim(x) <- c(r,c); x;
      }
    )
  )
}

splitMatricies <- list()

for (i in 1:length(inputImages))
{
  splitMatricies[[i]] <- matrixSplitter(imageMatricies[,,i], 48, 48)
}

#Initializing For loop Variables
blackCounter = 0 #To see how many black pixels are in each subsection
outputImages <- vector()

for (i in 1:length(inputImages))
{
  for (j in 1:100)
  {
    for (k in 1:48)
    {
      for (l in 1:48)
      {
        if (splitMatricies[[i]][j][[1]][k,l] == 0)
        {
          blackCounter = blackCounter + 1
        }
      }
    }
    
    if (blackCounter > 20)
    {
      outputImages[i] <- paste("./Subsections/subsection_", i, "_", j, ".png", sep = "")
      writeImage(splitMatricies[[i]][j][[1]], outputImages[i])
    }
    
    blackCounter = 0
  }
}
