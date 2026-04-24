library(this.path)
library(Matrix)

setwd(this.dir())
matrix <- readMM("../data/matrix.mtx")

# -------------------------------
# ---  Matrix visualization   ---
# -------------------------------

# Dimensions
dim(matrix)

# Total counts per cell
lib_size <- colSums(matrix)

# Summary per cell
summary(lib_size)

# % of zeros in gene expression
mean(matrix == 0) * 100

# Distribution total counts per cell
lib_size |> hist(main = "Counts per cell", breaks = 100)

# Now with log to visualize better
hist(log10(lib_size), breaks = 100, 
     main = "Total counts per cell",
     xlab = "log10(total counts)")

# Distribution total counts per gene
#rowSums(matrix) |> hist(main = "Counts per gene", breaks = 100)

