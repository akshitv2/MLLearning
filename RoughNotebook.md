
1. Central Limit Theorem
	- As sample size increases, sampling distribution of the mean tends to become normally distributed regardless of shape of population dist.
	- When sample is large (usually n>=30)
	- Sampling distribution should approximate normality to be valid

2. Standard Error
	$$\text{SE} = \frac{s}{\sqrt{n}}$$
	- Standard deviation of sampling dist
	- Measures how much sample mean is expected to vary from sample to sample.
	- Decreases as n increases

3. Law of large numbers
	- As number of observations increase sample mean converges with true population mean

4. Sample size determination
	- For Population Mean
			$$n = \left( \frac{E}{Z \cdot \sigma} \right)^2$$
		- 𝑛 = required sample size
		- 𝑍 = Z-score corresponding to the confidence level (📌e.g., 1.96 for 95%)
		- 𝜎 = population standard deviation (estimate if unknown)
		- 𝐸 = desired margin of error
	- For Population Proportion
		$$n = \frac{Z^2 \cdot p \cdot (1 - p)}{E^2}$$
		- Z = z-score corresponding to the desired confidence level
		- 𝑝 = estimated population proportion
		- 𝐸 = margin of error (in decimal form, 📌e.g., 0.05 for 5%)

5. Margin of Error (E)
	- Range within which true population parameter is expected to lie with level of confidence.

2. Efficient Softmax Variants

When the vocabulary is huge (millions), we don’t want to compute the full softmax denominator (summing over all tokens). Alternatives:

Sampled Softmax / Negative Sampling
Only compute logits for the true word + a random subset of "negative" words. Used in word2vec, RNN LMs, etc.

Hierarchical Softmax
Represent vocab as a binary tree (or Huffman tree). Computing probability is 
𝑂
(
log
⁡
∣
𝑉
∣
)
O(log∣V∣) instead of 
𝑂
(
∣
𝑉
∣
)
O(∣V∣).
Example: output layer decides "is it in the left subtree or right subtree?" recursively.

Adaptive Softmax (used in large LMs like fairseq)
Frequent words get their own dense cluster (fast), while rare words are grouped into classes, so you don’t compute the full distribution each time.

Mixture of Softmaxes (MoS)
Improves expressivity while still working with large vocab outputs.