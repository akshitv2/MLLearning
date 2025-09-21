# Statistics

## Central Tendency Measures
1. **Mean**: The arithmetic average of a dataset, calculated by summing all values and dividing by the number of observations. For a dataset \( x_1, x_2, \ldots, x_n \), the mean is:
   \[
   \bar{x} = \frac{\sum_{i=1}^n x_i}{n}
   \]
2. **Mode**: The value that appears most frequently in a dataset. A dataset may have one mode (unimodal), multiple modes (multimodal), or no mode if all values occur equally often.
3. **Median**: The middle value when a dataset is ordered. For an odd number of observations, it is the middle value; for an even number, it is the average of the two middle values.

## Spread Measures
1. **Variance**: Measures the average squared deviation of each data point from the mean. For a population:
   \[
   \sigma^2 = \frac{\sum_{i=1}^N (x_i - \mu)^2}{N}
   \]
   For a sample:
   \[
   s^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n-1}
   \]
2. **Standard Deviation**: The square root of the variance, providing a measure of spread in the same units as the data. For a population: \(\sigma = \sqrt{\sigma^2}\); for a sample: \(s = \sqrt{s^2}\).
3. **Range**: The difference between the maximum and minimum values in a dataset.
4. **Interquartile Range (IQR)**: The range between the first quartile (Q1, 25th percentile) and the third quartile (Q3, 75th percentile), calculated as:
   \[
   \text{IQR} = Q3 - Q1
   \]
5. **Outlier Detection using IQR**: Outliers are values that lie below \( Q1 - 1.5 \cdot \text{IQR} \) or above \( Q3 + 1.5 \cdot \text{IQR} \).

## Sampling
1. **Probability-Based Sampling**:
   - **Simple Random Sampling**: Each unit in the population has an equal chance of being selected.
   - **Stratified Sampling**: The population is divided into strata, and random samples are taken from each stratum.
   - **Cluster Sampling**: The population is divided into clusters, and entire clusters are randomly selected.
   - **Systematic Sampling**: Elements are selected at regular intervals from an ordered list.
2. **Non-Probability-Based Sampling**:
   - **Convenience Sampling**: Selecting units that are easily accessible.
   - **Judgmental Sampling**: Selecting units based on expert judgment.
   - **Quota Sampling**: Selecting a fixed number of units from specific subgroups.
   - **Snowball Sampling**: Existing subjects recruit future subjects, often used for rare populations.

## Important Distributions
1. **Normal Distribution**:
   - A symmetric, bell-shaped distribution characterized by its mean (\(\mu\)) and standard deviation (\(\sigma\)).
   - **Z-Score**: Measures how many standard deviations a data point is from the mean:
     \[
     z = \frac{x - \mu}{\sigma}
     \]
   - **Empirical Rule**: For a normal distribution:
     - Approximately 68% of data lies within \(\mu \pm 1\sigma\).
     - Approximately 95% of data lies within \(\mu \pm 2\sigma\).
     - Approximately 99.7% of data lies within \(\mu \pm 3\sigma\).
2. **Bernoulli Distribution**: Models a single trial with two outcomes (success or failure), with probability of success \(p\). Mean: \(p\), Variance: \(p(1-p)\).
3. **Poisson Distribution**: Models the number of events occurring in a fixed interval, with mean \(\lambda\). Probability mass function:
   \[
   P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots
   \]
   Mean and variance: \(\lambda\).
4. **Binomial Distribution**: Models the number of successes in \(n\) independent Bernoulli trials, each with success probability \(p\). Probability mass function:
   \[
   P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
   \]
   Mean: \(np\), Variance: \(np(1-p)\).

## Confidence Interval
- A range where a population parameter is likely to lie, with a specified confidence level (e.g., 95%).
- For large samples (\(n > 30\)), use the z-score:
  \[
  \text{CI} = \bar{x} \pm z \cdot \frac{s}{\sqrt{n}}
  \]
- For small samples (\(n \leq 30\)), use the t-score, which accounts for greater variability with fatter tails. The t-distribution varies with degrees of freedom (df = \(n-1\)).

## Expected Value
- The long-run average of a random variable, calculated as:
  \[
  E(X) = \sum x_i \cdot P(x_i) \quad \text{(discrete case)}
  \]
  \[
  E(X) = \int x \cdot f(x) \, dx \quad \text{(continuous case)}
  \]

## Standard Error
- Measures the precision of a sample mean as an estimate of the population mean:
  \[
  \text{SE} = \frac{s}{\sqrt{n}}
  \]

## Probability
- **Probability of an Event**: \( P(A) = \frac{\text{Number of favorable outcomes for A}}{\text{Total number of outcomes in sample space}} \).
- **Conditional Probability**: The probability of event A given event B:
  \[
  P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
  \]
- **Bayes' Theorem**: Relates conditional probabilities:
  \[
  P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
  \]
- **Probability Rules**:
  - **Union**: \( P(A \cup B) = P(A) + P(B) - P(A \cap B) \)
  - **Intersection**: \( P(A \cap B) = P(A) \cdot P(B) \) (if A and B are independent)
- **Probability Independence**: Events A and B are independent if:
  \[
  P(A \cap B) = P(A) \cdot P(B)
  \]

## Permutations
- The number of ways to arrange \(n\) items taken \(r\) at a time:
  \[
  P(n, r) = \frac{n!}{(n-r)!}
  \]

## Combinations
- The number of ways to choose \(r\) items from \(n\) without regard to order:
  \[
  C(n, r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}
  \]

## Random Variable
- A variable whose value depends on the outcome of a random experiment.
- **Discrete Random Variable**: Takes countable values (e.g., number of heads in coin flips).
- **Continuous Random Variable**: Takes values in a continuous range (e.g., height of a person).

## Probability Mass Function (PMF)
- For a discrete random variable, gives the probability of each possible value:
  \[
  P(X = x)
  \]
  Must satisfy: \(\sum P(X = x) = 1\).

## Probability Density Function (PDF)
- For a continuous random variable, describes the likelihood of values within an interval:
  \[
  P(a \leq X \leq b) = \int_a^b f(x) \, dx
  \]
  The total area under the PDF curve equals 1.

## Cumulative Distribution Function (CDF)
- Gives the probability that a random variable \(X\) is less than or equal to a value \(x\):
  \[
  F(x) = P(X \leq x)
  \]
  For discrete variables: \( F(x) = \sum_{k \leq x} P(X = k) \).
  For continuous variables: \( F(x) = \int_{-\infty}^x f(t) \, dt \).

## Additional Topics
1. **Skewness and Kurtosis**:
   - **Skewness**: Measures the asymmetry of a distribution. Positive skew indicates a longer right tail; negative skew indicates a longer left tail.
   - **Kurtosis**: Measures the "tailedness" of a distribution. High kurtosis indicates heavy tails; low kurtosis indicates light tails.
2. **Hypothesis Testing**:
   - Involves testing a null hypothesis (\(H_0\)) against an alternative hypothesis (\(H_a\)).
   - Key components: test statistic, p-value, significance level (\(\alpha\)).
   - Common tests: z-test, t-test, chi-square test, ANOVA.
3. **Correlation and Covariance**:
   - **Covariance**: Measures how two variables change together:
     \[
     \text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]
     \]
   - **Correlation**: Normalized covariance, ranging from -1 to 1:
     \[
     \rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
     \]
4. **Central Limit Theorem**:
   - States that the distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the population distribution, provided the population has finite mean and variance.
5. **Law of Large Numbers**:
   - As the sample size increases, the sample mean converges to the population mean.
6. **Regression Analysis**:
   - **Linear Regression**: Models the relationship between a dependent variable and one or more independent variables:
     \[
     y = \beta_0 + \beta_1 x + \epsilon
     \]
   - **Multiple Regression**: Extends linear regression to multiple predictors.
7. **Chi-Square Distribution**:
   - Used in hypothesis testing and goodness-of-fit tests, especially for categorical data.
8. **Exponential Distribution**:
   - Models the time between events in a Poisson process. PDF:
     \[
     f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
     \]
     Mean: \(1/\lambda\), Variance: \(1/\lambda^2\).