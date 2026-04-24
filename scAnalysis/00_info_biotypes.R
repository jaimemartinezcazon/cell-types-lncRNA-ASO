library(this.path)
setwd(this.dir())
data <- read.table("../data/gene_biotypes.tsv", header = TRUE, sep = "\t")

# column gene_biotype
biotypes <- data$gene_biotype

# unique values
biotypes_unique <- unique(biotypes)

# total nº of biotypes
num_biotypes <- length(biotypes_unique)

# show
cat("Nº of biotypes:", num_biotypes, "\n")
cat("List of biotypes:\n")
print(biotypes_unique)
