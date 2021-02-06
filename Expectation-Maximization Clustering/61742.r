library(MASS)
library(mixtools)
set.seed(421)

class_means <- matrix(c(+2.5, +2.5,
                        -2.5, +2.5,
                        -2.5, -2.5,
                        +2.5, -2.5,
                         0.0,  0.0), 2, 5)

class_covariances <- array(c(+0.8, -0.6, -0.6, +0.8,
                             +0.8, +0.6, +0.6, +0.8,
                             +0.8, -0.6, -0.6, +0.8,
                             +0.8, +0.6, +0.6, +0.8,
                             +1.6,  0.0,  0.0, +1.6), c(2, 2, 5))

class_sizes <- c(50,50,50,50,100)

points1 <- mvrnorm(n = class_sizes[1], mu = class_means[,1], Sigma = class_covariances[,,1])
points2 <- mvrnorm(n = class_sizes[2], mu = class_means[,2], Sigma = class_covariances[,,2])
points3 <- mvrnorm(n = class_sizes[3], mu = class_means[,3], Sigma = class_covariances[,,3])
points4 <- mvrnorm(n = class_sizes[4], mu = class_means[,4], Sigma = class_covariances[,,4])
points5 <- mvrnorm(n = class_sizes[5], mu = class_means[,5], Sigma = class_covariances[,,5])
X <- rbind(points1, points2, points3, points4, points5)

plot(X[,1], X[,2], type = "p", pch = 19, col = "black", las = 1, xlim = c(-6, 6), ylim = c(-6, 6), xlab = "x1", ylab = "x2")

centroids <- X[sample(1:300, 5),]

for(i in 1:2){
  distances <- as.matrix(dist(rbind(centroids, X), method = "euclidean"))
  distances <- distances[1:nrow(centroids), (nrow(centroids) + 1):(nrow(centroids) + nrow(X))]
  assignments <- sapply(1:ncol(distances), function(i) {which.min(distances[,i])})
  
  for (k in 1:5) {
    centroids[k,] <- colMeans(X[assignments == k,])
  }
}

density <- function(x_row, k){
  priors[k]*(det(covariances[,,k])**(-1/2)) * exp((-1/2)*sc(x_row - centroids[k,], covariances[,,k]))
}
sc <- function(X, cov){X %*% solve(cov) %*% cbind(X)}


H <- matrix(sapply(assignments, function(component){ (1:5) == component }), 300, 5, byrow = TRUE)

i <- 0

while(i<100){
  covariances <- sapply(X = 1:5, FUN = function(k) {   
  (t(X) - matrix(centroids[k,], 2, 300)) %*% diag(H[,k]) %*% t(t(X) - matrix(centroids[k,], 2, 300))/ sum(H[,k]) }, simplify = "array")

  priors <- colMeans(H)
  
  H <- t(sapply(1:300, function(n){
    row <- sapply(1:5, function(k){density(X[n,], k)})
    return(row / sum(row))
  }))
  
  centroids <- (t(H) %*% X ) / matrix(colSums(H), 5, 2)
  i <- i + 1
}

print(centroids)

means <- centroids

D <- as.matrix(dist(rbind(means, X), method = "euclidean"))
D <- D[1:nrow(means), (nrow(means) + 1):(nrow(means) + nrow(X))]
assignments <<- sapply(1:ncol(D), function(i) {which.min(D[,i])})

plot(X[assignments == 1, 1], X[assignments == 1, 2], type = "p", pch = 19, col = "blue", las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")
points(X[assignments == 2, 1], X[assignments == 2, 2], type = "p", pch = 19, col = "purple")
points(X[assignments == 3, 1], X[assignments == 3, 2], type = "p", pch = 19, col = "green")
points(X[assignments == 4, 1], X[assignments == 4, 2], type = "p", pch = 19, col = "orange")
points(X[assignments == 5, 1], X[assignments == 5, 2], type = "p", pch = 19, col = "red")

for(c in 1:5){
  ellipse(class_means[,c], class_covariances[,,c], alpha = .05, npoints = class_sizes[c], newplot = FALSE, draw = TRUE, lty=2, lwd=2)
  ellipse(means[c,], matrix(covariances[,,c], 2,2), alpha = .05, npoints = class_sizes[c], newplot = FALSE, draw = TRUE, lwd=2)
}




