# agile-community-rules-classification documentation!

## Description

This project builds a binary classifier that, given a Reddit comment and available context (subreddit, parent, metadata), outputs a calibrated probability that the comment violated a specific subreddit rule using a large corpus of moderated comments plus a small labeled dev set; the emphasis is on capturing community-specific norms (not censoring speech), preserving raw data immutability, evaluating with column-averaged AUC, analyzing calibration and subgroup failure modes, and delivering probabilistic predictions, clear error analysis, and practical integration guidance for moderators while enforcing strict privacy/ethics rules (redaction, no sharing of PII, human-in-the-loop decisions).

## Commands

The Makefile contains the central entry points for common tasks related to this project.

