# NLP Project based on GutenbergR

###################################################################
# 1. Find a text and start analysing it
###################################################################

library(gutenbergr)
library(tidytext)
library(dplyr)
library(stringr)

my_mirror <- "http://mirrors.xmission.com/gutenberg/"

df <- gutenberg_metadata

unique(df$author)[startsWith(unique(df$author), "Por")]

gutenberg_works(author == "Porter, Jane")

ScotChi <- gutenberg_download(6086, mirror = my_mirror)


###################################################################
# 2. Make a word and bi-gram count
###################################################################

words_ScotChi <- unnest_tokens(ScotChi, words, text)

count_unigrams <- count(words_ScotChi, words, sort=TRUE)

bigrams_ScotChi <- unnest_tokens(ScotChi, words, text, token = "ngrams", n=2)

count_bigrams <- count(bigrams_ScotChi, words, sort=TRUE)

count_bigrams <- count_bigrams[!is.na(count_bigrams$words), ]


###################################################################
# 3. Test for the dependence of the words of the bi-gram with a chi
# square test and illustrate the result with a mosaic plot
###################################################################

count_bigrams[startsWith(count_bigrams$words, "going to"),]
count_bigrams[startsWith(count_bigrams$words, "going "),]
count_bigrams[endsWith(count_bigrams$words, " to"),]

g.t <- count_bigrams[startsWith(count_bigrams$words, "going to"),]$n
g.nott <- sum(count_bigrams[startsWith(count_bigrams$words, "going "),]$n) - g.t
notg.t <- sum(count_bigrams[endsWith(count_bigrams$words, " to"),]$n) - g.t
notg.nott <- sum(count_bigrams$n) - g.nott -notg.t - g.t

freq <- matrix(c(g.t, g.nott, notg.t, notg.nott), ncol = 2, byrow = TRUE)
mosaicplot(freq)
chisq.test(freq)

contingency_table <- matrix(
  c(g.t, g.nott, notg.t, notg.nott),
  ncol = 2,
  byrow = TRUE,
  dimnames = list(c("w2", "not w2"), c("w1", "not w1"))
)

contingency_table
mosaicplot(contingency_table, main="Mosaic Plot of Bigram Dependence", color = TRUE)

###################################################################
# 4. Hypothesis test & chi-square test - Explanation
###################################################################


###################################################################
# 5. Compute the entropies of each 1k word string
###################################################################

# Start with empty vector
entropy <- c()

# Divide text into parts of 1k words
for(i in 0:285) # Text contains 286k+ words and we do not want to get out of bounds
{
  entr <- words_ScotChi[(i*1000 + 1):(i*1000+1000), 2]
  # Extract characters (words letter by letter)
  char <- unnest_tokens(entr, token, words, token = "characters")
  df.char <- as.data.frame(count(char, token, sort = TRUE))
  df.char$relfreq <- df.char$n/sum(df.char$n)
  df.char$sent <- df.char$relfreq*log2(df.char$relfreq)
  entropy <- c(entropy, - sum(df.char$sent))
}

entropy
plot(entropy)


###################################################################
# 6. Explanation - Entropy of random variable
###################################################################


###################################################################
# 7. 95% Confidence Interval for Entropies - T-Test Statistic
###################################################################

# Calculate the mean and standard deviation of entropy
mean_entropy <- mean(entropy)
sd_entropy <- sd(entropy)
n <- length(entropy)

# Compute the 95% confidence interval using the t-distribution
alpha <- 0.05
t_crit <- qt(1 - alpha/2, df = n - 1)

margin_of_error <- t_crit * (sd_entropy / sqrt(n))
lower_bound <- mean_entropy - margin_of_error
upper_bound <- mean_entropy + margin_of_error

print("95% Confidence Interval for Entropy: [", lower_bound, ",", upper_bound, "]\n")


###################################################################
# 8. Explanation - Confidence Interval
###################################################################


###################################################################
# 9. Naive Bayes Classification of a Sentence Across four Sections
###################################################################

# Extract the example sentence (~10 words)
example_sentence <- ScotChi$text[147] # Line 147
example_sentence

# Split into sentences to get one whole sentence starting in this line
example_sentence <- str_split(example_sentence, "\\.\\s+", simplify = TRUE)[1]
example_sentence

# Adjust Naive Bayes classification to use this sentence
# Divide text into 4 sections
n <- nrow(words_ScotChi)
section_size <- ceiling(n / 4)
sections <- list(
  section1 = words_ScotChi[1:section_size, ],
  section2 = words_ScotChi[(section_size + 1):(2 * section_size), ],
  section3 = words_ScotChi[(2 * section_size + 1):(3 * section_size), ],
  section4 = words_ScotChi[(3 * section_size + 1):n, ]
)

# Calculate word probabilities for each section
word_probs <- lapply(sections, function(section) {
  section %>% count(words) %>% mutate(prob = n / sum(n))
})

# Compute the Naive Bayes probability for the sentence in each section
naive_bayes_probs <- sapply(word_probs, function(wp) {
  probs <- wp %>% filter(words %in% sentence) %>% pull(prob)
  prod(probs, na.rm = TRUE) # Multiply probabilities
})

# Normalize to get relative probabilities
naive_bayes_probs <- naive_bayes_probs / sum(naive_bayes_probs)

# Output classification
print("Sentence classification probabilities:\n")
print(naive_bayes_probs)
print("Classified as section:", which.max(naive_bayes_probs), "\n")
