"""
BAP — Thought Space Population (Programmatic, No LLM)
======================================================
Domains: fibonacci_reasoning, arithmetic, multihop_facts

All primitives are hand-curated.
No LLM, no API key, no review step needed.
Quality is guaranteed by construction.

Usage:
    python populate_thought_space.py                        # builds all domains
    python populate_thought_space.py --domain arithmetic
    python populate_thought_space.py --domain fibonacci_reasoning --verify
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer
from shsrs.engine import SHSRSEngine

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

EMBED_MODEL   = "all-MiniLM-L6-v2"
GLOBAL_BUDGET = 1.0

HARM_TERMS = [
    "violence", "illegal", "harmful", "dangerous", "weapon",
    "abuse", "exploit", "toxic", "fraud", "deception"
]

# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThoughtNode:
    id:       int
    text:     str
    type:     str         # fact | rule | function | constraint
    embed:    List[float]
    strength: float = 0.5 # fixed at population — no bias
    age:      float = 0.0
    harm:     float = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# PRIMITIVES — HAND CURATED
# Each entry: (text, type)
# Types: fact | rule | function | constraint
# ──────────────────────────────────────────────────────────────────────────────

DOMAINS: Dict[str, List[Tuple[str, str]]] = {

    "fibonacci_reasoning": [
        # ── FACTS — core sequence values (F(0) to F(12)) ─────────────────────
        ("The Fibonacci sequence begins: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34.", "fact"),
        ("F(0) = 0.", "fact"),
        ("F(1) = 1.", "fact"),
        ("F(2) = 1.", "fact"),
        ("F(3) = 2.", "fact"),
        ("F(4) = 3.", "fact"),
        ("F(5) = 5.", "fact"),
        ("F(6) = 8.", "fact"),
        ("F(7) = 13.", "fact"),
        ("F(8) = 21.", "fact"),
        ("F(9) = 34.", "fact"),
        ("F(10) = 55.", "fact"),
        ("F(11) = 89.", "fact"),
        ("F(12) = 144.", "fact"),
        # ── FACTS — golden ratio (phi, ASCII-safe) ────────────────────────────
        ("The golden ratio phi ~= 1.6180339887.", "fact"),
        ("phi = (1 + sqrt(5)) / 2.", "fact"),
        ("Consecutive Fibonacci ratios F(n+1)/F(n) converge to phi as n increases.", "fact"),
        ("F(13)/F(12) = 233/144 ~= 1.618.", "fact"),
        # ── FACTS — classic properties ────────────────────────────────────────
        ("Every third Fibonacci number is even.", "fact"),
        ("The sum of the first n Fibonacci numbers equals F(n+2) - 1.", "fact"),
        ("Cassini identity: F(n-1)*F(n+1) - F(n)^2 = (-1)^n.", "fact"),
        ("The GCD of F(m) and F(n) equals F(GCD(m,n)).", "fact"),
        ("Fibonacci numbers grow approximately as phi^n / sqrt(5).", "fact"),
        ("Pascal's triangle shallow diagonals sum to Fibonacci numbers.", "fact"),
        # ── FACTS — real-world appearances ────────────────────────────────────
        ("Sunflower seed spirals follow Fibonacci counts, typically 34 and 55.", "fact"),
        ("Pinecone spirals count 8 and 13 — both Fibonacci numbers.", "fact"),
        ("Phyllotaxis describes leaf arrangements that follow Fibonacci patterns.", "fact"),
        ("The Fibonacci sequence is named after Leonardo of Pisa (Fibonacci).", "fact"),
        # ── FACTS — extended sequence values (F(13) to F(24)) ────────────────
        ("F(13) = 233.", "fact"),
        ("F(14) = 377.", "fact"),
        ("F(15) = 610.", "fact"),
        ("F(16) = 987.", "fact"),
        ("F(17) = 1597.", "fact"),
        ("F(18) = 2584.", "fact"),
        ("F(19) = 4181.", "fact"),
        ("F(20) = 6765.", "fact"),
        ("F(21) = 10946.", "fact"),
        ("F(22) = 17711.", "fact"),
        ("F(23) = 28657.", "fact"),
        ("F(24) = 46368.", "fact"),
        # ── FACTS — Lucas numbers ─────────────────────────────────────────────
        ("The Lucas sequence uses L(n) = L(n-1) + L(n-2) but starts L(0)=2, L(1)=1.", "fact"),
        ("L(0) = 2.", "fact"),
        ("L(1) = 1.", "fact"),
        ("L(2) = 3.", "fact"),
        ("L(3) = 4.", "fact"),
        ("L(4) = 7.", "fact"),
        ("L(5) = 11.", "fact"),
        ("L(6) = 18.", "fact"),
        ("L(7) = 29.", "fact"),
        ("L(8) = 47.", "fact"),
        # ── FACTS — Tribonacci sequence ───────────────────────────────────────
        ("The Tribonacci sequence: T(n) = T(n-1) + T(n-2) + T(n-3), T(0)=0, T(1)=0, T(2)=1.", "fact"),
        ("T(3)=1, T(4)=2, T(5)=4, T(6)=7, T(7)=13, T(8)=24, T(9)=44.", "fact"),
        ("The Tribonacci constant is approximately 1.3247, the limit of T(n+1)/T(n).", "fact"),
        ("T(10) = 81, T(11) = 149, T(12) = 274.", "fact"),
        ("The Tribonacci sequence requires three seed values to be fully determined.", "fact"),
        # ── FACTS — golden ratio deep properties ──────────────────────────────
        ("phi^2 = phi + 1, so phi satisfies x^2 - x - 1 = 0.", "fact"),
        ("1/phi = phi - 1 ~= 0.618.", "fact"),
        ("phi is irrational and is the unique positive root of x^2 - x - 1 = 0.", "fact"),
        ("The continued fraction of phi is [1; 1, 1, 1, ...] with all coefficients equal to 1.", "fact"),
        ("phi appears in the regular pentagon: the diagonal-to-side ratio equals phi.", "fact"),
        ("phi * psi = -1, where psi = (1 - sqrt(5)) / 2 ~= -0.618.", "fact"),
        ("psi = (1 - sqrt(5)) / 2 ~= -0.6180339887.", "fact"),
        ("phi is algebraic of degree 2 over the rationals.", "fact"),
        # ── FACTS — Fibonacci in CS, nature, history ──────────────────────────
        ("The Fibonacci heap uses Fibonacci numbers to bound the maximum degree of heap nodes.", "fact"),
        ("Fibonacci search uses Fibonacci numbers as division points for efficient sorted-array lookup.", "fact"),
        ("Every positive integer has a unique Zeckendorf representation as a sum of non-consecutive Fibonacci numbers.", "fact"),
        ("Flowers often have a Fibonacci number of petals: 3, 5, 8, 13, 21, or 34.", "fact"),
        ("The Fibonacci spiral closely approximates the golden spiral in nautilus shells.", "fact"),
        ("Leonardo of Pisa introduced the sequence in Liber Abaci (1202) via a rabbit breeding problem.", "fact"),
        ("L(n) = F(n-1) + F(n+1) relates each Lucas number to two Fibonacci numbers.", "fact"),
        # ── FACTS — Pisano period ─────────────────────────────────────────────
        ("The Pisano period pi(m) is the period of the Fibonacci sequence modulo m.", "fact"),
        ("pi(2) = 3: Fibonacci mod 2 cycles 0, 1, 1, 0, 1, 1, ...", "fact"),
        ("pi(5) = 20: Fibonacci mod 5 has period 20.", "fact"),
        ("pi(10) = 60: the last decimal digit of F(n) repeats every 60 terms.", "fact"),
        ("The Pisano period pi(m) is always even for m > 2.", "fact"),
        # ── FACTS — number theory ─────────────────────────────────────────────
        ("Consecutive Fibonacci numbers F(n) and F(n+1) are always coprime.", "fact"),
        ("F(6) = 8 is the only Fibonacci number known to be a perfect cube.", "fact"),
        ("F(n) is a Fibonacci prime only if n itself is prime (necessary, not sufficient).", "fact"),
        ("F(n) is always odd unless n is divisible by 3.", "fact"),
        ("The ratio L(n)/F(n) approaches sqrt(5) as n grows large.", "fact"),

        # ── RULES — Fibonacci core ────────────────────────────────────────────
        ("If F(n) and F(n-1) are known, then F(n+1) = F(n) + F(n-1).", "rule"),
        ("If a sequence has each term equal to the sum of the two prior terms, it is Fibonacci-like.", "rule"),
        ("If F(n) is given and n >= 2, then F(n-1) = F(n) - F(n-2).", "rule"),
        ("If F(n) is even, then n is a multiple of 3.", "rule"),
        ("If two consecutive Fibonacci terms are known, all subsequent terms are determined.", "rule"),
        ("If F(n)/F(n-1) is computed for large n, the result approaches phi.", "rule"),
        ("If a number equals the sum of its two predecessors, apply the Fibonacci recurrence.", "rule"),
        ("If n is a multiple of 3, F(n) is even.", "rule"),
        # ── RULES — Lucas sequence ────────────────────────────────────────────
        ("L(n) = L(n-1) + L(n-2) for all n >= 2, with L(0)=2, L(1)=1.", "rule"),
        ("If L(n-1) and L(n-2) are known, then L(n) = L(n-1) + L(n-2).", "rule"),
        ("L(n) = F(n-1) + F(n+1) for all n >= 1.", "rule"),
        ("If the ratio L(n+1)/L(n) is computed for large n, it also approaches phi.", "rule"),
        ("L(2n) = L(n)^2 - 2 * (-1)^n.", "rule"),
        ("L(n)^2 - 5 * F(n)^2 = 4 * (-1)^n.", "rule"),
        # ── RULES — identities ────────────────────────────────────────────────
        ("F(2n) = F(n) * L(n).", "rule"),
        ("F(2n+1) = F(n+1)^2 + F(n)^2.", "rule"),
        ("F(2n-1) = F(n)^2 + F(n-1)^2.", "rule"),
        ("F(n+m) = F(m) * F(n+1) + F(m-1) * F(n) (Fibonacci addition formula).", "rule"),
        ("F(n) * F(n+2) - F(n+1)^2 = (-1)^(n+1).", "rule"),
        ("phi^n = phi * F(n) + F(n-1) for n >= 1.", "rule"),
        # ── RULES — divisibility ──────────────────────────────────────────────
        ("If m divides n, then F(m) divides F(n).", "rule"),
        ("If gcd(m, n) = d, then gcd(F(m), F(n)) = F(d).", "rule"),
        ("F(n) and F(n+1) are always coprime — they share no common prime factor.", "rule"),
        ("Every fourth Fibonacci number is divisible by 3.", "rule"),
        ("Every fifth Fibonacci number is divisible by 5.", "rule"),
        ("Every sixth Fibonacci number is divisible by 8.", "rule"),
        ("F(n) is odd if and only if n is not a multiple of 3.", "rule"),
        # ── RULES — parity and growth ─────────────────────────────────────────
        ("F(n) < phi^n for all n >= 1.", "rule"),
        ("F(n) < 2^n for all n >= 1.", "rule"),
        ("F(n) grows as phi^n / sqrt(5); the number of digits is approximately n * log10(phi).", "rule"),
        ("phi - 1 = 1/phi (golden ratio reciprocal identity).", "rule"),
        ("phi = 1 + 1/phi (golden ratio as continued fraction base case).", "rule"),
        # ── RULES — Tribonacci ────────────────────────────────────────────────
        ("T(n) = T(n-1) + T(n-2) + T(n-3) for all n >= 3, with T(0)=0, T(1)=0, T(2)=1.", "rule"),
        ("If the tribonacci ratio T(n+1)/T(n) is computed for large n, it approaches 1.3247.", "rule"),
        # ── RULES — Pisano and modular ────────────────────────────────────────
        ("If the Pisano period pi(m) = k, then F(n+k) mod m = F(n) mod m for all n.", "rule"),
        ("F(n) mod 10 repeats with period 60 (the Pisano period of 10).", "rule"),
        ("pi(p^k) = p^(k-1) * pi(p) for prime p and k >= 1.", "rule"),
        # ── RULES — Zeckendorf ────────────────────────────────────────────────
        ("Every positive integer has a Zeckendorf representation using non-consecutive Fibonacci numbers.", "rule"),
        ("If the Zeckendorf representation of k is found, each Fibonacci summand must be non-consecutive.", "rule"),
        # ── RULES — sums ─────────────────────────────────────────────────────
        ("Sum of odd-indexed Fibonacci numbers: F(1)+F(3)+...+F(2n-1) = F(2n).", "rule"),
        ("Sum of even-indexed Fibonacci numbers: F(2)+F(4)+...+F(2n) = F(2n+1) - 1.", "rule"),
        ("Sum of squares: F(1)^2 + F(2)^2 + ... + F(n)^2 = F(n) * F(n+1).", "rule"),
        ("F(n)^2 - F(n-2)*F(n+2) = (-1)^n * F(2) (generalized Cassini for gap 2).", "rule"),
        ("If F(a) and F(b) are both even, F(a+b) is also even.", "rule"),
        ("For all n >= 6, F(n) > n (Fibonacci strictly exceeds its index).", "rule"),
        ("If p is a prime and p divides F(p-1), then p ≡ ±1 (mod 5) (Lucas-Fermat).", "rule"),
        ("gcd(F(n), F(m)) = F(gcd(n,m)) holds for all positive integers n and m.", "rule"),
        # ── RULES — golden ratio powers ───────────────────────────────────────
        ("phi^2 = phi + 1, so phi^3 = 2*phi + 1 and phi^4 = 3*phi + 2.", "rule"),
        ("phi^n = F(n) * phi + F(n-1) for all integers n >= 1.", "rule"),
        ("L(n) = phi^n + psi^n yields an integer for all n >= 0 with no rounding needed.", "rule"),

        # ── FUNCTIONS — core recurrence ───────────────────────────────────────
        ("F(n) = F(n-1) + F(n-2) for all n >= 2, with F(0)=0, F(1)=1.", "function"),
        ("Binet's formula: F(n) = (phi^n - psi^n) / sqrt(5), where psi = (1 - sqrt(5)) / 2.", "function"),
        ("Matrix form: [[1,1],[1,0]]^n = [[F(n+1), F(n)], [F(n), F(n-1)]].", "function"),
        ("Fast doubling: F(2k) = F(k)(2F(k+1) - F(k)); F(2k+1) = F(k)^2 + F(k+1)^2.", "function"),
        ("Sum of squares identity: F(n)^2 + F(n+1)^2 = F(2n+1).", "function"),
        ("Negative Fibonacci: F(-n) = (-1)^(n+1) * F(n).", "function"),
        # ── FUNCTIONS — Lucas ─────────────────────────────────────────────────
        ("Lucas closed form: L(n) = phi^n + psi^n, phi=(1+sqrt(5))/2, psi=(1-sqrt(5))/2.", "function"),
        ("L(n) from Fibonacci: L(n) = F(n-1) + F(n+1) for n >= 1.", "function"),
        ("Fast Lucas doubling: L(2n) = L(n)^2 - 2*(-1)^n.", "function"),
        ("L(n) to Fibonacci: F(n) = (L(n-1) + L(n+1)) / 5.", "function"),
        # ── FUNCTIONS — Tribonacci ────────────────────────────────────────────
        ("Tribonacci recurrence: T(n) = T(n-1) + T(n-2) + T(n-3), T(0)=0, T(1)=0, T(2)=1.", "function"),
        # ── FUNCTIONS — extended identities ───────────────────────────────────
        ("Fibonacci addition: F(n+m) = F(n)*F(m+1) + F(n-1)*F(m).", "function"),
        ("F(2n) = F(n) * L(n) (product formula via Lucas).", "function"),
        ("To verify Cassini identity: compute F(n-1)*F(n+1) - F(n)^2 and confirm result is (-1)^n.", "function"),
        ("Odd-index doubling formula: F(2n+1) = F(n+1)^2 + F(n)^2 allows fast computation.", "function"),
        # ── FUNCTIONS — digit and size ────────────────────────────────────────
        ("Number of digits in F(n): floor(n * log10(phi)) + 1 for n >= 1.", "function"),
        ("Fibonacci approximation: F(n) = round(phi^n / sqrt(5)) for n >= 0.", "function"),
        # ── FUNCTIONS — parity ────────────────────────────────────────────────
        ("Fibonacci parity: F(n) is even iff n mod 3 == 0; else odd.", "function"),
        ("Lucas parity: L(n) is even iff n is a multiple of 3.", "function"),
        # ── FUNCTIONS — Pisano ────────────────────────────────────────────────
        ("Pisano period by prime power: pi(p^k) = p^(k-1) * pi(p) for prime p, k >= 1.", "function"),
        ("Entry point alpha(m): smallest k >= 1 such that m divides F(k).", "function"),
        # ── FUNCTIONS — checks ────────────────────────────────────────────────
        ("Fibonacci membership test: x is a Fibonacci number iff 5*x^2+4 or 5*x^2-4 is a perfect square.", "function"),
        ("To check if F(n) is prime: verify F(n) passes a primality test; n prime is necessary.", "function"),
        # ── FUNCTIONS — sums ─────────────────────────────────────────────────
        ("Sum of first n Fibonacci numbers: F(1)+F(2)+...+F(n) = F(n+2) - 1.", "function"),
        ("Sum of squares of first n Fibonacci numbers: sum F(i)^2 for i=1..n = F(n)*F(n+1).", "function"),
        ("Sum of odd-indexed Fibonacci: F(1)+F(3)+...+F(2n-1) = F(2n).", "function"),
        ("Sum of even-indexed Fibonacci: F(2)+F(4)+...+F(2n) = F(2n+1) - 1.", "function"),
        # ── FUNCTIONS — phi approximations ───────────────────────────────────
        ("phi from ratio: F(k+1)/F(k) gives increasingly close rational approximations to phi.", "function"),
        ("phi from continued fraction: truncating [1; 1, 1, ..., 1] at k steps gives F(k+1)/F(k).", "function"),
        # ── FUNCTIONS — Zeckendorf ────────────────────────────────────────────
        ("Zeckendorf decomposition: greedily subtract the largest Fibonacci number <= n, repeat.", "function"),
        ("Zeckendorf verification: confirm no two summands are consecutive Fibonacci numbers.", "function"),
        # ── FUNCTIONS — matrix / fast ─────────────────────────────────────────
        ("Matrix power: compute [[1,1],[1,0]]^n via fast exponentiation; top-right entry is F(n).", "function"),
        ("Fast doubling halves operations: O(log n) steps to compute F(n).", "function"),
        ("psi = (1-sqrt(5))/2; psi^n -> 0 so Binet's formula rounds correctly to integers.", "function"),
        ("Fibonacci search: use Fibonacci numbers as division indices for sorted-array lookup.", "function"),
        ("Golden spiral radius at step n is proportional to phi^(n/2), forming a logarithmic spiral.", "function"),
        ("To extend Fibonacci backward: F(-n) = (-1)^(n+1) * F(n) for positive integer n.", "function"),
        ("Ratio error bound: |F(n+1)/F(n) - phi| < 1 / (F(n) * sqrt(5)).", "function"),
        ("L(n) parity: L(n) is odd when n is not a multiple of 3, even when it is.", "function"),
        ("To compute F(n) for large n without full table: use fast doubling or matrix exponentiation.", "function"),

        # ── CONSTRAINTS — core ────────────────────────────────────────────────
        ("F(n) is always a positive integer for n >= 1.", "constraint"),
        ("The Fibonacci recurrence requires exactly two prior terms — it is second-order.", "constraint"),
        ("A sequence is not Fibonacci if any term does not equal the sum of the preceding two.", "constraint"),
        ("F(0) = 0 by definition; redefining F(0) changes all subsequent values.", "constraint"),
        ("Binet's formula gives exact integers only when rounded — it involves irrational numbers.", "constraint"),
        ("Two different indices n share the same value only at F(1)=F(2)=1.", "constraint"),
        # ── CONSTRAINTS — Lucas and Tribonacci ───────────────────────────────
        ("The Lucas sequence is distinct from Fibonacci despite sharing the same recurrence.", "constraint"),
        ("Lucas numbers L(n) differ from Fibonacci numbers for all n > 2: L(n) != F(n).", "constraint"),
        ("The Tribonacci sequence requires three initial conditions, not two.", "constraint"),
        ("No simple closed form like Binet's formula exists for Tribonacci in elementary terms.", "constraint"),
        ("The Tribonacci constant ~= 1.3247 is not the same as the golden ratio phi ~= 1.618.", "constraint"),
        # ── CONSTRAINTS — phi ────────────────────────────────────────────────
        ("phi cannot be expressed as a ratio of two integers — it is irrational.", "constraint"),
        ("psi = (1-sqrt(5))/2 is negative; psi^n alternates in sign for positive integer n.", "constraint"),
        ("phi^n for large n grows faster than any fixed polynomial in n.", "constraint"),
        # ── CONSTRAINTS — Zeckendorf ──────────────────────────────────────────
        ("Zeckendorf representation cannot include two consecutive Fibonacci numbers as summands.", "constraint"),
        ("No positive integer has two different Zeckendorf representations — uniqueness is guaranteed.", "constraint"),
        # ── CONSTRAINTS — primes and squares ─────────────────────────────────
        ("Fibonacci primes are rare; F(n) being prime requires n to be prime, but not vice versa.", "constraint"),
        ("F(n) is a perfect square only for F(0)=0, F(1)=1, F(2)=1, and F(12)=144.", "constraint"),
        ("No two consecutive Fibonacci numbers can both be even.", "constraint"),
        # ── CONSTRAINTS — Pisano ─────────────────────────────────────────────
        ("The Pisano period pi(m) is always >= 1; it is never zero.", "constraint"),
        ("pi(p^k) = p^(k-1)*pi(p) holds only for prime p — composite moduli need separate analysis.", "constraint"),
        # ── CONSTRAINTS — computation ─────────────────────────────────────────
        ("Fast doubling requires careful handling of integer overflow for large n.", "constraint"),
        ("Fibonacci numbers beyond F(25) cannot be recalled from memory — use the recurrence.", "constraint"),
        ("The golden spiral is an approximation; the true golden spiral is a logarithmic spiral.", "constraint"),
        ("Fibonacci heap trees are Fibonacci-bounded in size, not exactly Fibonacci-numbered.", "constraint"),
        # ── CONSTRAINTS — identities scope ────────────────────────────────────
        ("The Cassini identity holds only for the standard Fibonacci sequence, not arbitrary Lucas sequences.", "constraint"),
        ("L(n)^2 - 5*F(n)^2 = 4*(-1)^n holds only for standard Lucas and Fibonacci — not modified seeds.", "constraint"),
        ("The Fibonacci sequence is the unique sequence with F(0)=0, F(1)=1, and F(n)=F(n-1)+F(n-2).", "constraint"),
        ("Entry point alpha(m) must divide the Pisano period pi(m).", "constraint"),
        ("phi^n grows without bound; it is not bounded above by any constant.", "constraint"),
    ],

    "arithmetic": [
        # ── FACTS — basic operations (18) ──────────────────────────────────────
        ("Addition is commutative: a + b = b + a.", "fact"),
        ("Multiplication is commutative: a * b = b * a.", "fact"),
        ("Subtraction is not commutative: a - b != b - a in general.", "fact"),
        ("Division is not commutative: a / b != b / a in general.", "fact"),
        ("Multiplying any number by 0 gives 0.", "fact"),
        ("Adding 0 to any number leaves it unchanged.", "fact"),
        ("Multiplying any number by 1 leaves it unchanged.", "fact"),
        ("A negative times a negative equals a positive.", "fact"),
        ("A negative times a positive equals a negative.", "fact"),
        ("Division by zero is undefined.", "fact"),
        ("A prime number has exactly two divisors: 1 and itself.", "fact"),
        ("1 is not a prime number.", "fact"),
        ("2 is the only even prime number.", "fact"),
        ("Every integer greater than 1 is either prime or composite.", "fact"),
        ("The square root of 2 is irrational.", "fact"),
        ("pi ~= 3.14159 and is irrational.", "fact"),
        ("An even number is divisible by 2 with no remainder.", "fact"),
        ("An odd number leaves remainder 1 when divided by 2.", "fact"),
        # ── FACTS — number systems (9) ─────────────────────────────────────────
        ("Natural numbers are the positive counting integers: 1, 2, 3, and so on.", "fact"),
        ("Integers include all whole numbers and their negatives: ..., -2, -1, 0, 1, 2, ...", "fact"),
        ("Rational numbers can be expressed as p/q where p and q are integers and q != 0.", "fact"),
        ("Irrational numbers cannot be expressed as a ratio of two integers.", "fact"),
        ("Real numbers include all rational and irrational numbers on the number line.", "fact"),
        ("Complex numbers have the form a + b*i where i*i = -1.", "fact"),
        ("Zero is neither positive nor negative.", "fact"),
        ("Every rational number has a terminating or repeating decimal expansion.", "fact"),
        ("Every terminating or repeating decimal is a rational number.", "fact"),
        # ── FACTS — divisibility (7) ───────────────────────────────────────────
        ("A number is divisible by 3 if the sum of its digits is divisible by 3.", "fact"),
        ("A number is divisible by 9 if the sum of its digits is divisible by 9.", "fact"),
        ("A number is divisible by 4 if its last two digits form a multiple of 4.", "fact"),
        ("A number is divisible by 8 if its last three digits form a multiple of 8.", "fact"),
        ("A number is divisible by 11 if its alternating digit sum is divisible by 11.", "fact"),
        ("Every integer greater than 1 has a unique prime factorization (Fundamental Theorem of Arithmetic).", "fact"),
        ("There are infinitely many prime numbers.", "fact"),
        # ── FACTS — primes (4) ─────────────────────────────────────────────────
        ("The first ten primes are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29.", "fact"),
        ("Twin primes are pairs of primes differing by 2, such as (11, 13) and (17, 19).", "fact"),
        ("A composite number has more than two positive divisors.", "fact"),
        ("Every composite number is a product of prime factors.", "fact"),
        # ── FACTS — exponents (6) ─────────────────────────────────────────────
        ("Any nonzero number raised to the power 0 equals 1.", "fact"),
        ("A number raised to the power 1 equals itself.", "fact"),
        ("A negative exponent is the reciprocal: a^(-n) = 1 / a^n for a != 0.", "fact"),
        ("The square of any real number is non-negative.", "fact"),
        ("2^10 = 1024.", "fact"),
        ("Squaring and square-rooting are inverse operations for non-negative numbers.", "fact"),
        # ── FACTS — roots (4) ─────────────────────────────────────────────────
        ("The square root of a perfect square is an integer.", "fact"),
        ("sqrt(4) = 2; sqrt(9) = 3; sqrt(16) = 4; sqrt(25) = 5.", "fact"),
        ("Cube root of 8 = 2; cube root of 27 = 3; cube root of 64 = 4.", "fact"),
        ("The nth root of x equals x^(1/n) for x >= 0 and n > 0.", "fact"),
        # ── FACTS — logarithms (5) ────────────────────────────────────────────
        ("log base 10 of 10 is 1; log base 10 of 100 is 2; log base 10 of 1000 is 3.", "fact"),
        ("The natural logarithm ln(x) uses base e, where e ~= 2.71828.", "fact"),
        ("log_b(1) = 0 for any valid base b.", "fact"),
        ("ln(e) = 1.", "fact"),
        ("Logarithm and exponentiation are inverse operations: log_b(b^x) = x.", "fact"),
        # ── FACTS — fractions and decimals (5) ────────────────────────────────
        ("Multiplying a fraction's numerator and denominator by the same nonzero value preserves its value.", "fact"),
        ("A fraction equals 1 when its numerator equals its denominator (both nonzero).", "fact"),
        ("Adding fractions requires a common denominator.", "fact"),
        ("To convert a fraction to decimal form, divide the numerator by the denominator.", "fact"),
        ("The repeating decimal 0.333... equals 1/3 exactly.", "fact"),
        # ── FACTS — percentages and growth (4) ────────────────────────────────
        ("A 100% increase doubles a value.", "fact"),
        ("A 50% decrease halves a value.", "fact"),
        ("A 50% increase followed by a 50% decrease yields 75% of the original value.", "fact"),
        ("Compounding two 10% increases yields a 21% total increase, not 20%.", "fact"),
        # ── FACTS — statistics (8) ────────────────────────────────────────────
        ("The median is the middle value in a sorted dataset.", "fact"),
        ("The mode is the most frequently occurring value in a dataset.", "fact"),
        ("The range of a dataset is the maximum minus the minimum.", "fact"),
        ("Variance measures the average squared deviation from the mean.", "fact"),
        ("Standard deviation is the square root of the variance.", "fact"),
        ("For a normal distribution, about 68% of values fall within one standard deviation of the mean.", "fact"),
        ("The mean of a dataset is sensitive to extreme outliers.", "fact"),
        ("Multiplying every value in a dataset by a constant multiplies the mean by that constant.", "fact"),
        # ── FACTS — geometry (7) ──────────────────────────────────────────────
        ("A triangle has three sides and three angles summing to 180 degrees.", "fact"),
        ("An equilateral triangle has all three sides equal and all angles equal to 60 degrees.", "fact"),
        ("A right triangle has one angle of exactly 90 degrees.", "fact"),
        ("The hypotenuse is the longest side of a right triangle, opposite the right angle.", "fact"),
        ("The circumference of a circle is 2 * pi * r.", "fact"),
        ("The diameter of a circle is twice its radius.", "fact"),
        ("A square has four equal sides and four right angles.", "fact"),
        # ── FACTS — coordinate geometry (3) ───────────────────────────────────
        ("The origin of a coordinate plane is the point (0, 0).", "fact"),
        ("The slope of a horizontal line is 0.", "fact"),
        ("The slope of a vertical line is undefined.", "fact"),

        # ── RULES — basic (12) ────────────────────────────────────────────────
        ("If a = b and b = c, then a = c.", "rule"),
        ("If a > b and b > c, then a > c.", "rule"),
        ("If a + b = c, then c - b = a.", "rule"),
        ("If a * b = c and b != 0, then c / b = a.", "rule"),
        ("If two numbers are both even, their sum is even.", "rule"),
        ("If one number is even and one is odd, their sum is odd.", "rule"),
        ("If two numbers are both odd, their sum is even.", "rule"),
        ("If a number is divisible by 2 and by 3, it is divisible by 6.", "rule"),
        ("If n mod 2 = 0, then n is even.", "rule"),
        ("If n mod 2 = 1, then n is odd.", "rule"),
        ("If a^2 = b and a >= 0, then a = sqrt(b).", "rule"),
        ("If a number ends in 0 or 5, it is divisible by 5.", "rule"),
        # ── RULES — exponents (5) ─────────────────────────────────────────────
        ("If multiplying powers of the same base, add the exponents: a^m * a^n = a^(m+n).", "rule"),
        ("If dividing powers of the same base, subtract the exponents: a^m / a^n = a^(m-n) for a != 0.", "rule"),
        ("If raising a power to a power, multiply the exponents: (a^m)^n = a^(m*n).", "rule"),
        ("If a^x = a^y with a > 0 and a != 1, then x = y.", "rule"),
        ("(a * b)^n = a^n * b^n for any positive integer n.", "rule"),
        # ── RULES — logarithms (5) ────────────────────────────────────────────
        ("log_b(x * y) = log_b(x) + log_b(y) for positive x, y.", "rule"),
        ("log_b(x / y) = log_b(x) - log_b(y) for positive x, y.", "rule"),
        ("log_b(x^n) = n * log_b(x).", "rule"),
        ("If log_b(x) = y, then b^y = x.", "rule"),
        ("Change of base: log_b(x) = log(x) / log(b) for any common logarithm base.", "rule"),
        # ── RULES — fractions (3) ─────────────────────────────────────────────
        ("If a/b = c/d and b, d != 0, then a * d = b * c (cross multiplication).", "rule"),
        ("If a fraction's numerator is 0 and the denominator is nonzero, the fraction equals 0.", "rule"),
        ("If two fractions have the same value but different numerators and denominators, they are equivalent.", "rule"),
        # ── RULES — inequalities (6) ──────────────────────────────────────────
        ("If a > b, then a + c > b + c for any real c.", "rule"),
        ("If a > b and c > 0, then a * c > b * c.", "rule"),
        ("If a > b and c < 0, then a * c < b * c (inequality reverses).", "rule"),
        ("If a > b > 0, then 1/a < 1/b.", "rule"),
        ("If |x| < k and k > 0, then -k < x < k.", "rule"),
        ("If |x| > k and k > 0, then x > k or x < -k.", "rule"),
        # ── RULES — modular arithmetic (3) ────────────────────────────────────
        ("If a is congruent to b (mod m), then a + c is congruent to b + c (mod m).", "rule"),
        ("If a is congruent to b (mod m) and c is congruent to d (mod m), then a + c is congruent to b + d (mod m).", "rule"),
        ("If a is congruent to b (mod m), then a * c is congruent to b * c (mod m).", "rule"),
        # ── RULES — divisibility (4) ──────────────────────────────────────────
        ("If m divides a and m divides b, then m divides a + b.", "rule"),
        ("If m divides a and m divides b, then m divides a - b.", "rule"),
        ("If m divides a, then m divides a * k for any integer k.", "rule"),
        ("If p is prime and p divides a * b, then p divides a or p divides b.", "rule"),
        # ── RULES — absolute value (3) ────────────────────────────────────────
        ("If |a| = b and b >= 0, then a = b or a = -b.", "rule"),
        ("|a * b| = |a| * |b| for any real numbers a and b.", "rule"),
        ("|a + b| <= |a| + |b| for any real numbers a and b (triangle inequality).", "rule"),
        # ── RULES — algebraic identities (4) ──────────────────────────────────
        ("(a + b)^2 = a^2 + 2*a*b + b^2.", "rule"),
        ("(a - b)^2 = a^2 - 2*a*b + b^2.", "rule"),
        ("a^2 - b^2 = (a + b) * (a - b) (difference of squares).", "rule"),
        ("(a + b)^3 = a^3 + 3*a^2*b + 3*a*b^2 + b^3.", "rule"),
        # ── RULES — number theory (2) ─────────────────────────────────────────
        ("Euclid's algorithm: gcd(a, b) = gcd(b, a mod b); terminates when remainder is 0.", "rule"),
        ("If gcd(a, b) = 1 and a divides b * c, then a divides c.", "rule"),
        # ── RULES — geometry (3) ──────────────────────────────────────────────
        ("If two triangles have equal corresponding angles, they are similar.", "rule"),
        ("In a right triangle, the hypotenuse squared equals the sum of squares of the other two sides.", "rule"),
        ("The exterior angle of a triangle equals the sum of the two non-adjacent interior angles.", "rule"),

        # ── FUNCTIONS — basic (8) ─────────────────────────────────────────────
        ("Percentage: p% of x = (p / 100) * x.", "function"),
        ("Simple interest: I = P * r * t.", "function"),
        ("Area of a rectangle: A = length * width.", "function"),
        ("Area of a circle: A = pi * r^2.", "function"),
        ("Pythagorean theorem: a^2 + b^2 = c^2 for right triangles.", "function"),
        ("Quadratic formula: x = (-b +/- sqrt(b^2 - 4*a*c)) / (2*a).", "function"),
        ("Arithmetic mean of n values: sum / n.", "function"),
        ("Order of operations: parentheses -> exponents -> multiply/divide -> add/subtract.", "function"),
        # ── FUNCTIONS — series and sequences (9) ──────────────────────────────
        ("Sum of first n natural numbers: 1 + 2 + ... + n = n * (n + 1) / 2.", "function"),
        ("Sum of first n perfect squares: n * (n+1) * (2*n+1) / 6.", "function"),
        ("Sum of first n perfect cubes: (n * (n+1) / 2)^2.", "function"),
        ("Sum of arithmetic series: S_n = n / 2 * (first term + last term).", "function"),
        ("nth term of an arithmetic sequence: a_n = a_1 + (n - 1) * d.", "function"),
        ("nth term of a geometric sequence: a_n = a_1 * r^(n - 1).", "function"),
        ("Sum of a finite geometric series: S_n = a_1 * (1 - r^n) / (1 - r) for r != 1.", "function"),
        ("Sum of an infinite geometric series with |r| < 1: S = a_1 / (1 - r).", "function"),
        ("Arithmetic series step form: S_n = n / 2 * (2 * a_1 + (n - 1) * d).", "function"),
        # ── FUNCTIONS — number theory (2) ─────────────────────────────────────
        ("GCD via Euclidean algorithm: gcd(a, b) = gcd(b, a mod b); base case gcd(a, 0) = a.", "function"),
        ("LCM formula: lcm(a, b) = |a * b| / gcd(a, b).", "function"),
        # ── FUNCTIONS — combinatorics (4) ─────────────────────────────────────
        ("Factorial: n! = n * (n - 1) * ... * 2 * 1 for n >= 1; 0! = 1 by definition.", "function"),
        ("Combinations: C(n, r) = n! / (r! * (n - r)!).", "function"),
        ("Permutations: P(n, r) = n! / (n - r)!.", "function"),
        ("Binomial theorem: (a + b)^n = sum over k=0 to n of C(n, k) * a^(n-k) * b^k.", "function"),
        # ── FUNCTIONS — coordinate geometry (5) ───────────────────────────────
        ("Distance between two points: d = sqrt((x2 - x1)^2 + (y2 - y1)^2).", "function"),
        ("Midpoint of two points: M = ((x1 + x2) / 2, (y1 + y2) / 2).", "function"),
        ("Slope of a line through two points: m = (y2 - y1) / (x2 - x1) for x1 != x2.", "function"),
        ("Slope-intercept form of a line: y = m * x + b.", "function"),
        ("Point-slope form of a line: y - y1 = m * (x - x1).", "function"),
        # ── FUNCTIONS — geometry (6) ───────────────────────────────────────────
        ("Area of a triangle: A = (1/2) * base * height.", "function"),
        ("Heron's formula for triangle area: A = sqrt(s*(s-a)*(s-b)*(s-c)) where s = (a+b+c)/2.", "function"),
        ("Volume of a cylinder: V = pi * r^2 * h.", "function"),
        ("Volume of a cone: V = (1/3) * pi * r^2 * h.", "function"),
        ("Surface area of a sphere: SA = 4 * pi * r^2.", "function"),
        ("Volume of a sphere: V = (4/3) * pi * r^3.", "function"),
        # ── FUNCTIONS — finance and statistics (4) ────────────────────────────
        ("Compound interest: A = P * (1 + r/n)^(n*t).", "function"),
        ("Z-score formula: z = (x - mean) / standard_deviation.", "function"),
        ("Weighted average: sum of (weight_i * value_i) / sum of all weights.", "function"),
        ("Variance of a dataset: sum of (x_i - mean)^2 / n.", "function"),
        # ── FUNCTIONS — utility (2) ────────────────────────────────────────────
        ("Floor function floor(x): the greatest integer not exceeding x.", "function"),
        ("Ceiling function ceil(x): the smallest integer not less than x.", "function"),

        # ── CONSTRAINTS — basic (7) ───────────────────────────────────────────
        ("A number cannot be both even and odd.", "constraint"),
        ("The result of dividing two integers is not always an integer.", "constraint"),
        ("A prime number must be greater than 1.", "constraint"),
        ("Square roots of negative numbers are not real.", "constraint"),
        ("Two quantities cannot be simultaneously equal and unequal.", "constraint"),
        ("Probability must be between 0 and 1 inclusive.", "constraint"),
        ("A percentage cannot exceed 100% of a whole unless comparing to a baseline.", "constraint"),
        # ── CONSTRAINTS — logarithms and exponents (4) ────────────────────────
        ("The logarithm of a non-positive number is undefined in real arithmetic.", "constraint"),
        ("The base of a logarithm must be positive and not equal to 1.", "constraint"),
        ("A logarithm base of 1 is invalid because 1 raised to any power always equals 1.", "constraint"),
        ("The nth root of a negative number is not real when n is even.", "constraint"),
        # ── CONSTRAINTS — combinatorics (3) ───────────────────────────────────
        ("The factorial function is defined only for non-negative integers.", "constraint"),
        ("C(n, r) requires 0 <= r <= n; it equals 0 when r > n by convention.", "constraint"),
        ("C(n, 0) = 1 and C(n, n) = 1 for all n >= 0.", "constraint"),
        # ── CONSTRAINTS — number theory (3) ───────────────────────────────────
        ("gcd(0, 0) is undefined; gcd(a, 0) = a for nonzero a.", "constraint"),
        ("No two distinct values can both be the GCD of the same pair of numbers.", "constraint"),
        ("Modular arithmetic requires a positive integer modulus.", "constraint"),
        # ── FACTS — modular arithmetic examples (to anchor mod routing) ───────
        ("The modulo operation a mod b gives the remainder when integer a is divided by b.", "fact"),
        ("17 mod 5 = 2, because 17 = 3 * 5 + 2.", "fact"),
        ("Modular arithmetic (mod): 10 mod 3 = 1, 7 mod 4 = 3, 20 mod 6 = 2.", "fact"),
        # ── CONSTRAINTS — algebra and arithmetic (5) ──────────────────────────
        ("No real number satisfies x^2 < 0.", "constraint"),
        ("Multiplying an irrational number by a nonzero rational number yields an irrational number.", "constraint"),
        ("An inequality's direction reverses when both sides are multiplied by a negative number.", "constraint"),
        ("An integer divided by a larger positive integer has absolute value less than 1.", "constraint"),
        ("Floor and ceiling functions return the input unchanged when the input is already an integer.", "constraint"),
        # ── CONSTRAINTS — series and statistics (5) ───────────────────────────
        ("The sum of an infinite geometric series diverges when |r| >= 1.", "constraint"),
        ("Standard deviation is non-negative; it is 0 only when all values are identical.", "constraint"),
        ("The median is resistant to outliers; the mean is not.", "constraint"),
        ("A geometric series with first term 0 is trivially 0 for every partial sum.", "constraint"),
        ("The AM-GM inequality: the arithmetic mean of non-negative numbers is >= their geometric mean.", "constraint"),
        # ── CONSTRAINTS — geometry (3) ────────────────────────────────────────
        ("The sum of angles in any triangle is exactly 180 degrees in Euclidean geometry.", "constraint"),
        ("Two parallel lines in Euclidean geometry never intersect.", "constraint"),
        ("The slope of a vertical line is undefined due to division by zero in the slope formula.", "constraint"),
    ],

    "multihop_facts": [
        # ── FACTS — science: Curie, Einstein, quantum ─────────────────────────
        ("Marie Curie was awarded the Nobel Prize in Physics in 1903.", "fact"),
        ("Marie Curie was awarded the Nobel Prize in Chemistry in 1911.", "fact"),
        ("Marie Curie was the first woman to win a Nobel Prize.", "fact"),
        ("The Nobel Prize in Physics is awarded for outstanding contributions to physics.", "fact"),
        ("Physics is a natural science studying matter, energy, and their interactions.", "fact"),
        ("Albert Einstein received the Nobel Prize in Physics in 1921.", "fact"),
        ("Einstein's Nobel Prize was for his discovery of the photoelectric effect.", "fact"),
        ("The photoelectric effect is a quantum phenomenon.", "fact"),
        ("Quantum mechanics describes physical phenomena at the atomic scale.", "fact"),
        ("Marie Curie's research focused on radioactivity.", "fact"),
        ("Polonium and radium were discovered by Marie Curie.", "fact"),
        ("Radioactivity is the emission of radiation from unstable atomic nuclei.", "fact"),
        ("The nucleus of an atom contains protons and neutrons.", "fact"),
        ("Electrons occupy orbitals around the nucleus.", "fact"),
        # ── FACTS — geography and history (original) ──────────────────────────
        ("Paris is the capital of France.", "fact"),
        ("France is a country in Western Europe.", "fact"),
        ("The Eiffel Tower is located in Paris, France.", "fact"),
        ("The Eiffel Tower was completed in 1889.", "fact"),
        ("World War II ended in 1945.", "fact"),
        ("The United Nations was founded in 1945.", "fact"),
        ("The United States declared independence in 1776.", "fact"),
        ("Washington D.C. is the capital of the United States.", "fact"),
        ("London is the capital of the United Kingdom.", "fact"),
        ("The Berlin Wall fell in 1989.", "fact"),
        # ── FACTS — technology and inventions ────────────────────────────────
        ("Alexander Graham Bell invented the telephone in 1876.", "fact"),
        ("Thomas Edison developed a practical electric lightbulb in 1879.", "fact"),
        ("Orville and Wilbur Wright made the first powered airplane flight in 1903 at Kitty Hawk.", "fact"),
        ("Johannes Gutenberg invented movable type printing around 1440.", "fact"),
        ("James Watt improved the steam engine in 1769, spurring the Industrial Revolution.", "fact"),
        ("Karl Benz built the first practical gasoline-powered automobile in 1885.", "fact"),
        ("Alexander Fleming discovered penicillin in 1928.", "fact"),
        ("Watson and Crick described the double helix structure of DNA in 1953.", "fact"),
        ("Guglielmo Marconi demonstrated wireless radio transmission in 1895.", "fact"),
        ("ARPANET, the precursor to the modern internet, was created in 1969.", "fact"),
        ("Tim Berners-Lee invented the World Wide Web in 1989.", "fact"),
        ("The transistor was invented at Bell Labs in 1947.", "fact"),
        ("Wilhelm Roentgen discovered X-rays in 1895.", "fact"),
        ("Nikola Tesla developed the alternating current (AC) electrical system in the 1880s.", "fact"),
        ("Benjamin Franklin demonstrated that lightning is electrical in nature in 1752.", "fact"),
        ("Louis Pasteur developed germ theory and early vaccines for cholera and anthrax.", "fact"),
        ("Isaac Newton formulated three laws of motion and the law of universal gravitation.", "fact"),
        ("Charles Darwin published 'On the Origin of Species' in 1859, introducing natural selection.", "fact"),
        ("Dmitri Mendeleev created the periodic table of elements in 1869.", "fact"),
        ("Edward Jenner developed the first vaccine against smallpox in 1796.", "fact"),
        # ── FACTS — science (extended) ────────────────────────────────────────
        ("The speed of light in a vacuum is approximately 3 * 10^8 meters per second.", "fact"),
        ("Water boils at 100 degrees Celsius at standard atmospheric pressure.", "fact"),
        ("The human body has approximately 37 trillion cells.", "fact"),
        ("DNA is made of four chemical bases: adenine, thymine, guanine, and cytosine.", "fact"),
        ("The periodic table currently contains 118 confirmed chemical elements.", "fact"),
        ("Oxygen is the most abundant element in Earth's crust by mass.", "fact"),
        ("The Milky Way galaxy contains approximately 200 to 400 billion stars.", "fact"),
        ("The Earth is approximately 150 million kilometers from the Sun (1 astronomical unit).", "fact"),
        ("The Big Bang theory states the universe began approximately 13.8 billion years ago.", "fact"),
        ("Newton's First Law: an object stays at rest or in motion unless acted on by a net force.", "fact"),
        ("Newton's Third Law: for every action, there is an equal and opposite reaction.", "fact"),
        ("E = mc^2 is Einstein's equation relating mass and energy.", "fact"),
        ("Plate tectonics theory explains continental drift and the causes of earthquakes.", "fact"),
        ("The human genome contains approximately 3 billion base pairs of DNA.", "fact"),
        ("Blood circulation was first accurately described by William Harvey in 1628.", "fact"),
        # ── FACTS — history and geography (extended) ──────────────────────────
        ("Rome is the capital of Italy and was the center of the ancient Roman Empire.", "fact"),
        ("Julius Caesar was a Roman general and dictator assassinated on March 15, 44 BC.", "fact"),
        ("Napoleon Bonaparte became Emperor of France in 1804.", "fact"),
        ("The French Revolution began in 1789 and led to the abolition of the French monarchy.", "fact"),
        ("The American Civil War was fought from 1861 to 1865 between Union and Confederate states.", "fact"),
        ("World War I lasted from 1914 to 1918.", "fact"),
        ("The Industrial Revolution began in Britain in the late 18th century.", "fact"),
        ("The Great Wall of China was built over many centuries to protect against northern invasions.", "fact"),
        ("Cleopatra VII was the last active ruler of the Ptolemaic Kingdom of Egypt.", "fact"),
        ("Alexander the Great built one of history's largest empires, dying at age 32.", "fact"),
        ("The Ancient Olympic Games were first recorded in Olympia, Greece, in 776 BC.", "fact"),
        ("The modern Olympic Games were revived in Athens, Greece, in 1896.", "fact"),
        ("Tokyo is the capital of Japan.", "fact"),
        ("Beijing is the capital of China.", "fact"),
        ("Moscow is the capital of Russia.", "fact"),
        ("Ottawa is the capital of Canada.", "fact"),
        ("Canberra is the capital of Australia.", "fact"),
        ("Cairo is the capital of Egypt.", "fact"),
        ("The Amazon River is the world's largest river by water discharge.", "fact"),
        ("Mount Everest, at approximately 8849 meters, is the tallest mountain above sea level.", "fact"),
        ("The Pacific Ocean is the largest ocean on Earth by area.", "fact"),
        ("The Great Pyramid of Giza was built approximately 2560 BC as a tomb for Pharaoh Khufu.", "fact"),
        ("The Magna Carta, signed in 1215, limited the power of the English king.", "fact"),
        ("The Renaissance was a cultural movement that began in Italy in the 14th century.", "fact"),
        ("The Silk Road was an ancient trade network connecting China to the Mediterranean.", "fact"),
        # ── FACTS — arts and literature ───────────────────────────────────────
        ("William Shakespeare wrote approximately 37 plays, including Hamlet and Romeo and Juliet.", "fact"),
        ("Leonardo da Vinci painted the Mona Lisa and The Last Supper.", "fact"),
        ("Vincent van Gogh painted The Starry Night in 1889.", "fact"),
        ("Ludwig van Beethoven composed nine symphonies despite becoming deaf.", "fact"),
        ("Wolfgang Amadeus Mozart composed over 600 works before his death at age 35.", "fact"),
        ("Miguel de Cervantes wrote Don Quixote (1605), considered the first modern European novel.", "fact"),
        ("Homer's Iliad and Odyssey are ancient Greek epic poems foundational to Western literature.", "fact"),
        ("The Taj Mahal in Agra, India, was built by Mughal emperor Shah Jahan in the 17th century.", "fact"),
        ("Michelangelo painted the Sistine Chapel ceiling in Rome between 1508 and 1512.", "fact"),
        ("Pablo Picasso co-founded Cubism and painted Guernica in 1937.", "fact"),
        # ── FACTS — sports and culture ────────────────────────────────────────
        ("The FIFA World Cup is held every four years and is the world's most-watched sporting event.", "fact"),
        ("The first FIFA World Cup was held in Uruguay in 1930.", "fact"),
        ("Basketball was invented by James Naismith in 1891.", "fact"),
        ("A standard marathon distance is 26.2 miles (42.195 kilometers).", "fact"),
        ("Chess originated in India around the 6th century AD and spread to Persia and Europe.", "fact"),
        ("The Nobel Peace Prize is awarded to individuals or organizations that advance world peace.", "fact"),

        # ── RULES — attribution and prizes ────────────────────────────────────
        ("If X won the Nobel Prize in Physics, then X made a significant contribution to physics.", "rule"),
        ("If X discovered a chemical element, then X contributed to chemistry.", "rule"),
        ("If X is located in city Y, and Y is in country Z, then X is in country Z.", "rule"),
        ("If event A caused event B, and B caused C, then A indirectly caused C.", "rule"),
        ("If two events share the same year, they are contemporaneous.", "rule"),
        ("If X and Y are capitals, they belong to different countries unless specified.", "rule"),
        ("If a person won multiple Nobel Prizes, they worked across multiple fields.", "rule"),
        ("If X is the first person to achieve Y, no one preceded X in achieving Y.", "rule"),
        # ── RULES — invention and discovery ───────────────────────────────────
        ("If X invented Y in year Z, then X is credited with creating Y around year Z.", "rule"),
        ("If X won the Nobel Prize in field Y, X made an exceptional contribution to field Y.", "rule"),
        ("If X is the capital of country Y, X serves as the seat of government for Y.", "rule"),
        ("If X wrote work Y, then Y is attributed to X as its author or creator.", "rule"),
        ("If X is in continent Y and Z is a country in Y, X and Z are in the same continental region.", "rule"),
        ("If X is the largest Y in group Z, then all other members of group Z are smaller than X.", "rule"),
        ("If X's research led to discovery Y, and Y saved lives, X's work has humanitarian value.", "rule"),
        ("If X was assassinated in year Y, X's rule or influence ended in year Y.", "rule"),
        ("If X discovered element Y, element Y is associated with X in the history of chemistry.", "rule"),
        ("If X invented Y and Y led to Z, then X's invention contributed to Z.", "rule"),
        ("If X developed theory Y and Y was confirmed by experiment, Y is accepted scientific theory.", "rule"),
        ("If X ruled country Y from year A to year B, X governed Y during that period.", "rule"),
        ("If city X is in country Y and Y is in Europe, then X is a European city.", "rule"),
        ("If X painted Y and Y is displayed in museum Z, Y can be seen at Z.", "rule"),
        ("If X composed work Y for instrument Z, Y is typically performed on Z.", "rule"),
        ("If X won prize Y in year Z, X was the recipient of Y in Z.", "rule"),
        ("If X occurred in century Y, X is a historical event of the Y century.", "rule"),
        ("If X is adjacent to ocean Y, X has a coastline on Y.", "rule"),
        ("If country X borders country Y, X and Y share a geographic boundary.", "rule"),
        ("If X is a mountain and X has the highest elevation in region Y, X is the highest peak in Y.", "rule"),
        ("If X is recognized as a landmark of civilization, X has historical or cultural significance.", "rule"),
        ("If X depends on technology Y and Y was invented after X, there is a logical inconsistency.", "rule"),
        ("If X is the founding year of institution Y, Y did not exist before X.", "rule"),
        ("If person X died before event Y occurred, X had no direct involvement in Y.", "rule"),
        ("If two scientists independently discovered the same thing, the discovery is attributed to both.", "rule"),
        ("If X and Y both contributed to Z, Z is a collaborative achievement.", "rule"),
        ("If event X triggered event Y in the same year, X and Y are causally linked and contemporaneous.", "rule"),
        ("If X is the president of country Y, X is the executive head of Y's government.", "rule"),
        ("If X is an ancient structure still standing, X has survived for centuries.", "rule"),
        ("If city X served as the capital of empire Y, X had strategic importance to Y.", "rule"),
        ("If X's population exceeds Y's population, X is more populous than Y.", "rule"),
        ("If X is named after person Y, the naming is a tribute or recognition of Y.", "rule"),
        ("If X is translated from language Y to language Z, X is accessible to speakers of Z.", "rule"),
        ("If X and Y fought a war and Y won, Y gained strategic control or advantage.", "rule"),
        ("If X is the second-largest Y, then exactly one Y is larger than X.", "rule"),
        ("If X is both a scientist and an artist, X can contribute to both domains.", "rule"),
        ("If X won multiple awards in the same field, X is considered highly accomplished in that field.", "rule"),
        ("If X is a scientific law, X describes a consistent, testable regularity in nature.", "rule"),

        # ── FUNCTIONS — identification ─────────────────────────────────────────
        ("To find a person's scientific field: identify the category of their Nobel Prize.", "function"),
        ("To find which country a landmark belongs to: identify the city, then the country.", "function"),
        ("To chain two facts: find the shared entity and link their attributes.", "function"),
        ("To check temporal order: compare the years of two events.", "function"),
        ("To verify a multi-hop claim: trace each hop and confirm each link independently.", "function"),
        ("To identify who invented X: search for the first person credited with creating X.", "function"),
        ("To verify a historical date: cross-reference with independently documented events.", "function"),
        ("To determine the capital of a country: identify the officially designated seat of government.", "function"),
        ("To trace the origin of an invention: identify the problem it solved and the inventor credited.", "function"),
        ("To determine if X preceded Y: compare the year X occurred with the year Y occurred.", "function"),
        ("To find where a landmark is located: identify its city, then identify that city's country.", "function"),
        ("To identify someone's nationality: look for their birthplace, citizenship, or country of residence.", "function"),
        ("To attribute a discovery: find the first published or officially credited account.", "function"),
        ("To find the duration of a historical period: subtract the start year from the end year.", "function"),
        ("To find the influence of a historical figure: trace what changed after their actions.", "function"),
        ("To chain two facts about the same person: find the shared entity and link attributes.", "function"),
        ("To verify contemporaneous events: confirm they share the same year or overlapping dates.", "function"),
        ("To find the continent of a country: consult world geography knowledge.", "function"),
        ("To find the successor to a historical ruler: identify who assumed power after the reign ended.", "function"),
        ("To find the precursor to a modern technology: trace its technological lineage back to origin.", "function"),
        ("To determine if two things are related: find a shared attribute, ancestor, or causal link.", "function"),
        ("To verify a geographic fact: check the political or physical boundaries involved.", "function"),
        ("To identify the field of a scientist: look at the subject matter of their research.", "function"),
        ("To determine whether two events are causally linked: verify the proposed mechanism.", "function"),
        ("To find the location of a museum or artwork: trace its current recorded institutional home.", "function"),

        # ── CONSTRAINTS — logical ─────────────────────────────────────────────
        ("Two contradictory facts about the same subject cannot both be true.", "constraint"),
        ("A person cannot win a Nobel Prize in a field they did not contribute to.", "constraint"),
        ("A city can be the capital of only one country at a time.", "constraint"),
        ("An event cannot precede its own cause.", "constraint"),
        ("A person cannot be in two locations simultaneously.", "constraint"),
        ("Historical facts are fixed — they do not change based on interpretation.", "constraint"),
        # ── CONSTRAINTS — Nobel and awards ────────────────────────────────────
        ("Since 1974, the Nobel Prize cannot be awarded posthumously.", "constraint"),
        ("A Nobel Prize in a given year is shared by at most three co-recipients.", "constraint"),
        # ── CONSTRAINTS — geography ───────────────────────────────────────────
        ("No two sovereign countries can legally occupy the same geographic territory simultaneously.", "constraint"),
        ("An ocean is not located on a continent; oceans separate land masses.", "constraint"),
        ("A country's official capital is determined by its own laws, not external recognition alone.", "constraint"),
        # ── CONSTRAINTS — science ─────────────────────────────────────────────
        ("No information can travel faster than the speed of light in a vacuum.", "constraint"),
        ("A scientific theory cannot be simultaneously confirmed and definitively disproven.", "constraint"),
        # ── CONSTRAINTS — people and time ─────────────────────────────────────
        ("A person cannot be their own biological ancestor.", "constraint"),
        ("An invention cannot predate the inventor's birth.", "constraint"),
        ("An event that occurred cannot be said to have not occurred.", "constraint"),
        ("Two different people cannot both be the sole original inventor of the same technology.", "constraint"),
        ("Elements in the periodic table each have a unique atomic number.", "constraint"),
        ("A person cannot win multiple Nobel Prizes if their work spanned only one narrow subfield.", "constraint"),
        ("A historical figure cannot be both the inventor and the destroyer of the same technology.", "constraint"),
        # ── RULES — additional ────────────────────────────────────────────────
        ("If X is a body of water and X separates continents Y and Z, X lies between Y and Z.", "rule"),
        ("If X is a composer and Y is X's most famous symphony, Y is part of X's musical legacy.", "rule"),
        ("If X is an empire and Y is X's capital, Y was the political center of X.", "rule"),
        ("If X's theory replaced theory Y, X's theory is considered more accurate or general than Y.", "rule"),
        # ── FUNCTIONS — additional ────────────────────────────────────────────
        ("To determine what replaced an older theory: identify the later theory that explains the same phenomena.", "function"),
        ("To find a river's continent: identify the countries it flows through, then the continent.", "function"),
        ("To verify whether X is an ancient civilization: check if X predates the modern nation-state era.", "function"),
        ("To identify an empire's capital: find the city from which the empire was governed.", "function"),
        ("To compare the size of two oceans: use surface area measurements (Pacific > Atlantic > Indian).", "function"),
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# BUILD PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def build_harm_cluster(embedder: SentenceTransformer) -> np.ndarray:
    vecs = embedder.encode(HARM_TERMS, normalize_embeddings=True)
    cluster = vecs.mean(axis=0)
    return cluster / np.linalg.norm(cluster)


def build_nodes(
    primitives: List[Tuple[str, str]],
    embedder: SentenceTransformer,
    harm_cluster: np.ndarray
) -> List[ThoughtNode]:
    texts = [p[0] for p in primitives]
    types = [p[1] for p in primitives]

    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    nodes = []
    for i, (text, typ, emb) in enumerate(zip(texts, types, embeddings)):
        harm = float(max(0.0, np.dot(emb, harm_cluster)))
        nodes.append(ThoughtNode(
            id=i, text=text, type=typ,
            embed=emb.tolist(),
            strength=0.5, age=0.0, harm=harm
        ))
    return nodes


def _n_clusters(n_nodes: int) -> int:
    """
    Choose SHSRS cluster count so MiniBatchKMeans creates no ghost centroids.
    N // 20 gives ~20 nodes/cluster — dense enough that all centroids attract
    real points.  max(4, ...) keeps tiny domains stable.
    """
    return max(4, n_nodes // 20)


def build_index(nodes: List[ThoughtNode], domain: str) -> SHSRSEngine:
    vectors = np.array([n.embed for n in nodes], dtype=np.float32)
    n_clusters = _n_clusters(len(nodes))
    index_dir = f"thought_space_{domain}"

    engine = SHSRSEngine.build(
        vectors=vectors,
        index_dir=index_dir,
        n_clusters=n_clusters,
        M=8,
        ef_construction=100,
    )
    print(f"  Index built -> {index_dir}/  ({len(nodes)} nodes, {n_clusters} clusters)")
    return engine


def save_nodes(nodes: List[ThoughtNode], domain: str):
    path = Path(f"thought_space_{domain}/nodes.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    dist = dict(Counter(n.type for n in nodes))
    data = {
        "domain": domain,
        "count": len(nodes),
        "type_distribution": dist,
        "nodes": [asdict(n) for n in nodes]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Nodes saved -> {path}")
    print(f"  Distribution: {dist}")


def verify_index(nodes: List[ThoughtNode], engine: SHSRSEngine, sample_n: int = 5):
    import random
    print("\n" + "=" * 65)
    print("NEIGHBOR VERIFICATION")
    print("Semantically related neighbors -> index healthy")
    print("Random/unrelated neighbors     -> refine primitives")
    print("=" * 65)

    node_lookup  = {n.id: n for n in nodes}
    n_clusters   = _n_clusters(len(nodes))
    avg_cluster  = len(nodes) // n_clusters
    # argpartition in SHSRS requires k < total candidates collected.
    # With probe=1, we get ~avg_cluster candidates. Keep k well below that.
    safe_k    = max(1, min(3, avg_cluster - 2))
    safe_probe = 1
    for node in random.sample(nodes, min(sample_n, len(nodes))):
        qvec = np.array(node.embed, dtype=np.float32)
        results = engine.search(qvec, k=safe_k, probe=safe_probe)
        print(f"\n[{node.type}] {node.text}")
        for nid, sim in results:
            if nid == node.id:
                continue
            nb = node_lookup.get(nid)
            if nb:
                print(f"  {sim:.4f}  [{nb.type}]  {nb.text}")
    print("\n" + "=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="all",
                        help="fibonacci_reasoning | arithmetic | multihop_facts | all")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    print(f"Loading embedder: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    harm_cluster = build_harm_cluster(embedder)

    targets = list(DOMAINS.keys()) if args.domain == "all" else [args.domain]

    for domain in targets:
        if domain not in DOMAINS:
            print(f"[SKIP] Unknown domain: {domain}")
            continue

        primitives = DOMAINS[domain]
        print(f"\n{'='*60}")
        print(f"Domain: {domain}  ({len(primitives)} primitives)")
        print(f"{'='*60}")

        nodes  = build_nodes(primitives, embedder, harm_cluster)
        engine = build_index(nodes, domain)
        save_nodes(nodes, domain)

        if args.verify:
            verify_index(nodes, engine, sample_n=5)

    print("\n[OK] All thought spaces ready.")
    print("   Next: BAP propagation engine.")


if __name__ == "__main__":
    main()
