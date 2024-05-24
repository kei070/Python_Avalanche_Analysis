#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for running a Monte-Carlo style p-value test. Generated with the help of ChatUiT.
"""

# imports
import numpy as np

# test statistic --> here we use the difference in means
def test_statistic(sample_a, sample_b):
    return np.mean(sample_a) - np.mean(sample_b)
# end def

# the Monte-Carlo style function
def monte_carlo_p(sample_ai, sample_bi, num_permutations=500000, switch=True):

    """
    This function "tests" the hypothesis that it is only by chance that sample a is larger than sample b. That is, if
    the resulting p-value is < 0.05 one may accept on th 95% percent level that sample a is larger than sample b.
    IMPORTANT: If the switch parameter is set to True the function will check if the average of sample a is larger than
               that of sample b and if not, it will switch the samples.

    Explanation:
       --> the observed statisitic is mean(sample_a) - mean(sample_b)
       --> if mean(sample_a) > mean(sample_b) this is positive
       --> we then shuffle the data around and extract two new (now random) samples and calculate their difference
       --> we then check if the difference between this random sample is LARGER than our observed statistic
       --> only if it is larger, we increase the counter
       --> we repeat this a larger number of times (e.g., 500000) = number of permuations
       --> if the statistic based on the random samples is OFTEN larger than our observed statistic, our count variable
           will also be quite large and vice versa
       --> we then calculate the p-value as the count variable divided by the number of perumtations
       --> if this p-value is smaller than 0.05 we may say that our result is significant on the 95% level
    """

    diff = np.mean(sample_ai) - np.mean(sample_bi)

    # switch the samples if requested
    if switch:
        if diff < 0:
            sample_a = sample_bi
            sample_b = sample_ai
            print("\nSwitching samples...")
        else:
            sample_a = sample_ai
            sample_b = sample_bi
        # end if else
    else:
        sample_a = sample_ai
        sample_b = sample_bi
    # end if


    # calculate the test statistic for the original samples
    observed_statistic = test_statistic(sample_a, sample_b)

    # Permutation test
    count = 0

    combined_samples = np.concatenate((sample_a, sample_b))
    n1 = len(sample_a)


    for _ in range(num_permutations):
        np.random.shuffle(combined_samples)
        new_sample1 = combined_samples[:n1]
        new_sample2 = combined_samples[n1:]
        new_statistic = test_statistic(new_sample1, new_sample2)

        # Check if the permuted test statistic is at least as extreme as the observed
        if new_statistic >= observed_statistic:
            count += 1
        # end if
    # end for _

    # Calculate the p-value
    p_value = count / num_permutations

    # Print the result
    # print(f"Observed statistic: {observed_statistic}")
    print(f"P-value: {p_value}")

    return p_value

# end def