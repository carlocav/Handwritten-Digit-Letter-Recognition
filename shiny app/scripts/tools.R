library(spatstat)
library(EBImage)

tool.htmlImageToBlob <- function(img) {
  blob = gsub("data:image/png;base64,", "", img, fixed=TRUE)
  blob <- base64enc::base64decode(blob)
  return(blob)
}

tool.writeBlob <- function(file, data) {
  to.write <- file(file, "wb")
  writeBin(data, to.write)
  close(to.write)
  invisible(NULL)
}

# 384/128 = 3 i.e. our resolution
tool.readPNG <- function(file, resolution = 3) {
  m = png::readPNG(file)
  if (dim(m)[3] == 4) {
    m = (m[,,4]*m[,,1]+m[,,4]*m[,,2]+m[,,4]*m[,,3])/3
  } else {
    m = (m[,,1]+m[,,2]+m[,,3])/3
  }
  # m is now a matrix of dimension 384 * 384

  mStart = array(numeric(128*128), dim=c(128,128))
  for (i in 1:128) {
    for (j in 1:128) {
      mStart[j,i] = mean(m[(1+resolution*(i-1)):(1+resolution*(i-1)+(resolution-1)),(1+resolution*(j-1)):(1+resolution*(j-1)+(resolution-1))])
    }
  }
  # gaussian blur
  m <- as.matrix(blur(as.im(mStart), sigma = 1))
  m <- m[c(-1,-2,-127,-128), c(-1,-2,-127,-128)]  

  # ROI extraction
  rmean <- apply(m, 1, mean)<0.99
  cmean <- apply(m, 2, mean)<0.99
  m <- m[rmean, cmean]

  # center the frame
  if(dim(m)[1] < dim(m)[2]) {
    halfdiff = (dim(m)[2] - dim(m)[1])/2 
    if (halfdiff%%1 == 0) {
      mat <- matrix(1,halfdiff,dim(m)[2])
      m <- rbind(mat,m,mat)
    }
    else {
      mat1 <- matrix(1,round(halfdiff+0.01),dim(m)[2])
      mat2 <- matrix(1,round(halfdiff+0.01)-1,dim(m)[2])
      m <- rbind(mat1,m,mat2)
    }
  } else if (dim(m)[1] > dim(m)[2]) {
    halfdiff = (dim(m)[1] - dim(m)[2])/2
    if (halfdiff%%1 == 0) {
      mat <- matrix(1,dim(m)[1],halfdiff)
      m <- cbind(mat,m,mat)
    }
    else {
      mat1 <- matrix(1, dim(m)[1], round(halfdiff+0.01))
      mat2 <- matrix(1, dim(m)[1], round(halfdiff+0.01) - 1)
      m <- cbind(mat1, m, mat2)
    }
  }
  
  # resized
  mSmall <- EBImage::resize(m, w = 26, h = 26)
  
  # add 2 pixel padding
  mat <- rep(1,nrow(mSmall))
  mSmall <- cbind(mat, mSmall, mat)
  mat <- rep(1,ncol(mSmall))
  mSmall <- rbind(mat, mSmall, mat)
  
  return(1 - mSmall)
}