# Transformer From Scratch  
Rebuilding "Attention Is All You Need" step by step  

A complete from-scratch implementation of the Transformer architecture (Vaswani et al., 2017) using PyTorch and NumPy. This project aims to provide an educational, minimal, and fully transparent codebase that demystifies the core concepts of attention, multi-head attention, positional encoding, and Transformer blocks — with detailed explanations and visualizations.

## Table of Contents
- Background
- Goals
- Architecture Overview
- Implemented Components
- Installation
- Usage
- Results
- Visualizations
- References
- License

## Background  
The 2017 paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) introduced the Transformer architecture, revolutionizing natural language processing and enabling models such as BERT, GPT, and T5.  

This project is a faithful, fully manual implementation of the original paper — built step-by-step to promote deep understanding of:
- Self-Attention Mechanism
- Multi-Head Attention
- Positional Encodings
- Encoder & Decoder Blocks
- Transformer End-to-End Pipeline

## Goals
- Implement Transformer from scratch without using high-level APIs  
- Provide readable, well-documented code  
- Create visualizations for attention and weights  
- Train on a small translation dataset (English → German)  
- Understand every component at the algorithmic level  

## Architecture Overview  

