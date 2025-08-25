library(ggpubr)

data <- read.table("result/selected_with_length_fixed.txt", header = TRUE, sep = "\t")
data$CATH_C <- factor(data$CATH_C)

p <- gghistogram(data, x = "DOMAIN_LENGTH", fill = "CATH_C", facet.by = "CATH_C",
           color = "gray",  xlab = "Domain", ylab = "Count", add = "median",
           title = "Domain Length distribution", bins=30, palette = "npg")

ggsave("result/histogram.pdf", plot = p, width = 8, height = 6)
ggsave("result/histogram.png", plot = p, width = 8, height = 6, type = "cairo")

