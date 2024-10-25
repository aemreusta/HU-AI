# Optimizing Earthquake Aid Distribution

This program solves the Optimizing Earthquake Aid Distribution using Dynamic Programming approach.

## Problem Statement

Given a set of humanitarian aid items of varying weights and a package box that can carry at most K kilograms, place as many items as possible into the box such that the maximum possible weight of humanitarian aid is distributed given the package box capacity. You can assume there is just one copy of each item, which can either be packed as a whole or not (i.e., items cannot be split into smaller parts).

A greedy strategy that can come to mind is to continue taking the heaviest item that still fits into the remaining capacity of the box until no more items can be placed inside. However, that strategy will not always result in the most optimum solution. A Dynamic Programming approach is used to optimally solve this problem.

## Approach

Dynamic Programming approach organizes computations to avoid recomputing values that are already known, which can often save a great deal of time and space.

Instead of solving the original problem, the problem is reduced to essentially the same problem with smaller number of items and possibly smaller box capacity. First, it is checked if it is possible to fully pack the box of capacity K with the items of weights w0, ..., w(n-1). If it is possible to fully pack the box, then there exists a set of items S ⊆ {w0, ..., w(n-1)} of total weight K. Then, there are two possible cases:

> If w(n-1) ∉ S, then a box of capacity K can be fully packed using the first n-1 itmes.
>
> If w(n-1) ∈ S, then we can remove the item of weight wn-1 from the box, and the remaining items will have a total weight of (K - w(n-1)) kg. So, a box of capacity (K - w(n-1)) kg can be fully packed with the first n-1 items.

The problem can be reduced to smaller subproblems until there are no more items to consider, in which case, depending on the remaining capacity of the box, it can be concluded that it is either possible to pack all items if the remaining capacity is 0, or not, otherwise. To sum up the approach of reducing the given problem to subproblems and using them to construct the final solution, let P(w, i) be a Boolean function that evaluates to True if it is possible to fully pack a box of capacity w using the first i items, and False, otherwise. Then,

```text
P(w, i) = P(w, i-1) OR P(w - w(n-1), i-1) if w(n-1) <= w 
P(w, i-1) otherwise
```

To get the maximum weight of a subset of the given items that fits into a box of capacity K, the maximum w such that P(w, n) is True is found. I.e., the maximum weight that can be packed when all n items are considered.

Memoization is used to keep the solutions of subproblems instead of using a recursive approach, which would solve the problem too slowly since it would recompute the same values over and over again.

## Input Format

The first line of the input contains an integer K (capacity of the box in kg) and the number of items n (like food, water, medical supplies, and blankets). The second line contains n integers w0, ..., w(n-1) separated by spaces, which represent the items' weights in kg.

```text
K n
w0 w1 ... w(n-1)
```

## Output Format

On the matrices are printed with one space between each element, and each row is printed on a separate line.

Sample Input

```text
20 6
10 4 6 9 10 7
```

Sample Output

```text
20
TTTTTTT
FFFFFFF
FFFFFFF
FFFFFFF
FFTTTTT
FFFFFFF
FFFTTTT
FFFFFFT
FFFFFFF
FFFFTTT
FTTTTTT
FFFFFFT
FFFFFFF
FFFFTTT
FFTTTTT
FFFFTTT
FFFTTTT
FFFFFFT
FFFFFFF
FFFFTTT
FFFTTTT
```
